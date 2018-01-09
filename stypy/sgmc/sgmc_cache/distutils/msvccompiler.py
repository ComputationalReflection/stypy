
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.msvccompiler
2: 
3: Contains MSVCCompiler, an implementation of the abstract CCompiler class
4: for the Microsoft Visual Studio.
5: '''
6: 
7: # Written by Perry Stoll
8: # hacked by Robin Becker and Thomas Heller to do a better job of
9: #   finding DevStudio (through the registry)
10: 
11: __revision__ = "$Id$"
12: 
13: import sys
14: import os
15: import string
16: 
17: from distutils.errors import (DistutilsExecError, DistutilsPlatformError,
18:                               CompileError, LibError, LinkError)
19: from distutils.ccompiler import CCompiler, gen_lib_options
20: from distutils import log
21: 
22: _can_read_reg = 0
23: try:
24:     import _winreg
25: 
26:     _can_read_reg = 1
27:     hkey_mod = _winreg
28: 
29:     RegOpenKeyEx = _winreg.OpenKeyEx
30:     RegEnumKey = _winreg.EnumKey
31:     RegEnumValue = _winreg.EnumValue
32:     RegError = _winreg.error
33: 
34: except ImportError:
35:     try:
36:         import win32api
37:         import win32con
38:         _can_read_reg = 1
39:         hkey_mod = win32con
40: 
41:         RegOpenKeyEx = win32api.RegOpenKeyEx
42:         RegEnumKey = win32api.RegEnumKey
43:         RegEnumValue = win32api.RegEnumValue
44:         RegError = win32api.error
45: 
46:     except ImportError:
47:         log.info("Warning: Can't read registry to find the "
48:                  "necessary compiler setting\n"
49:                  "Make sure that Python modules _winreg, "
50:                  "win32api or win32con are installed.")
51:         pass
52: 
53: if _can_read_reg:
54:     HKEYS = (hkey_mod.HKEY_USERS,
55:              hkey_mod.HKEY_CURRENT_USER,
56:              hkey_mod.HKEY_LOCAL_MACHINE,
57:              hkey_mod.HKEY_CLASSES_ROOT)
58: 
59: def read_keys(base, key):
60:     '''Return list of registry keys.'''
61: 
62:     try:
63:         handle = RegOpenKeyEx(base, key)
64:     except RegError:
65:         return None
66:     L = []
67:     i = 0
68:     while 1:
69:         try:
70:             k = RegEnumKey(handle, i)
71:         except RegError:
72:             break
73:         L.append(k)
74:         i = i + 1
75:     return L
76: 
77: def read_values(base, key):
78:     '''Return dict of registry keys and values.
79: 
80:     All names are converted to lowercase.
81:     '''
82:     try:
83:         handle = RegOpenKeyEx(base, key)
84:     except RegError:
85:         return None
86:     d = {}
87:     i = 0
88:     while 1:
89:         try:
90:             name, value, type = RegEnumValue(handle, i)
91:         except RegError:
92:             break
93:         name = name.lower()
94:         d[convert_mbcs(name)] = convert_mbcs(value)
95:         i = i + 1
96:     return d
97: 
98: def convert_mbcs(s):
99:     enc = getattr(s, "encode", None)
100:     if enc is not None:
101:         try:
102:             s = enc("mbcs")
103:         except UnicodeError:
104:             pass
105:     return s
106: 
107: class MacroExpander:
108: 
109:     def __init__(self, version):
110:         self.macros = {}
111:         self.load_macros(version)
112: 
113:     def set_macro(self, macro, path, key):
114:         for base in HKEYS:
115:             d = read_values(base, path)
116:             if d:
117:                 self.macros["$(%s)" % macro] = d[key]
118:                 break
119: 
120:     def load_macros(self, version):
121:         vsbase = r"Software\Microsoft\VisualStudio\%0.1f" % version
122:         self.set_macro("VCInstallDir", vsbase + r"\Setup\VC", "productdir")
123:         self.set_macro("VSInstallDir", vsbase + r"\Setup\VS", "productdir")
124:         net = r"Software\Microsoft\.NETFramework"
125:         self.set_macro("FrameworkDir", net, "installroot")
126:         try:
127:             if version > 7.0:
128:                 self.set_macro("FrameworkSDKDir", net, "sdkinstallrootv1.1")
129:             else:
130:                 self.set_macro("FrameworkSDKDir", net, "sdkinstallroot")
131:         except KeyError:
132:             raise DistutilsPlatformError, \
133:                   ('''Python was built with Visual Studio 2003;
134: extensions must be built with a compiler than can generate compatible binaries.
135: Visual Studio 2003 was not found on this system. If you have Cygwin installed,
136: you can try compiling with MingW32, by passing "-c mingw32" to setup.py.''')
137: 
138:         p = r"Software\Microsoft\NET Framework Setup\Product"
139:         for base in HKEYS:
140:             try:
141:                 h = RegOpenKeyEx(base, p)
142:             except RegError:
143:                 continue
144:             key = RegEnumKey(h, 0)
145:             d = read_values(base, r"%s\%s" % (p, key))
146:             self.macros["$(FrameworkVersion)"] = d["version"]
147: 
148:     def sub(self, s):
149:         for k, v in self.macros.items():
150:             s = string.replace(s, k, v)
151:         return s
152: 
153: def get_build_version():
154:     '''Return the version of MSVC that was used to build Python.
155: 
156:     For Python 2.3 and up, the version number is included in
157:     sys.version.  For earlier versions, assume the compiler is MSVC 6.
158:     '''
159: 
160:     prefix = "MSC v."
161:     i = string.find(sys.version, prefix)
162:     if i == -1:
163:         return 6
164:     i = i + len(prefix)
165:     s, rest = sys.version[i:].split(" ", 1)
166:     majorVersion = int(s[:-2]) - 6
167:     minorVersion = int(s[2:3]) / 10.0
168:     # I don't think paths are affected by minor version in version 6
169:     if majorVersion == 6:
170:         minorVersion = 0
171:     if majorVersion >= 6:
172:         return majorVersion + minorVersion
173:     # else we don't know what version of the compiler this is
174:     return None
175: 
176: def get_build_architecture():
177:     '''Return the processor architecture.
178: 
179:     Possible results are "Intel", "Itanium", or "AMD64".
180:     '''
181: 
182:     prefix = " bit ("
183:     i = string.find(sys.version, prefix)
184:     if i == -1:
185:         return "Intel"
186:     j = string.find(sys.version, ")", i)
187:     return sys.version[i+len(prefix):j]
188: 
189: def normalize_and_reduce_paths(paths):
190:     '''Return a list of normalized paths with duplicates removed.
191: 
192:     The current order of paths is maintained.
193:     '''
194:     # Paths are normalized so things like:  /a and /a/ aren't both preserved.
195:     reduced_paths = []
196:     for p in paths:
197:         np = os.path.normpath(p)
198:         # XXX(nnorwitz): O(n**2), if reduced_paths gets long perhaps use a set.
199:         if np not in reduced_paths:
200:             reduced_paths.append(np)
201:     return reduced_paths
202: 
203: 
204: class MSVCCompiler (CCompiler) :
205:     '''Concrete class that implements an interface to Microsoft Visual C++,
206:        as defined by the CCompiler abstract class.'''
207: 
208:     compiler_type = 'msvc'
209: 
210:     # Just set this so CCompiler's constructor doesn't barf.  We currently
211:     # don't use the 'set_executables()' bureaucracy provided by CCompiler,
212:     # as it really isn't necessary for this sort of single-compiler class.
213:     # Would be nice to have a consistent interface with UnixCCompiler,
214:     # though, so it's worth thinking about.
215:     executables = {}
216: 
217:     # Private class data (need to distinguish C from C++ source for compiler)
218:     _c_extensions = ['.c']
219:     _cpp_extensions = ['.cc', '.cpp', '.cxx']
220:     _rc_extensions = ['.rc']
221:     _mc_extensions = ['.mc']
222: 
223:     # Needed for the filename generation methods provided by the
224:     # base class, CCompiler.
225:     src_extensions = (_c_extensions + _cpp_extensions +
226:                       _rc_extensions + _mc_extensions)
227:     res_extension = '.res'
228:     obj_extension = '.obj'
229:     static_lib_extension = '.lib'
230:     shared_lib_extension = '.dll'
231:     static_lib_format = shared_lib_format = '%s%s'
232:     exe_extension = '.exe'
233: 
234:     def __init__ (self, verbose=0, dry_run=0, force=0):
235:         CCompiler.__init__ (self, verbose, dry_run, force)
236:         self.__version = get_build_version()
237:         self.__arch = get_build_architecture()
238:         if self.__arch == "Intel":
239:             # x86
240:             if self.__version >= 7:
241:                 self.__root = r"Software\Microsoft\VisualStudio"
242:                 self.__macros = MacroExpander(self.__version)
243:             else:
244:                 self.__root = r"Software\Microsoft\Devstudio"
245:             self.__product = "Visual Studio version %s" % self.__version
246:         else:
247:             # Win64. Assume this was built with the platform SDK
248:             self.__product = "Microsoft SDK compiler %s" % (self.__version + 6)
249: 
250:         self.initialized = False
251: 
252:     def initialize(self):
253:         self.__paths = []
254:         if "DISTUTILS_USE_SDK" in os.environ and "MSSdk" in os.environ and self.find_exe("cl.exe"):
255:             # Assume that the SDK set up everything alright; don't try to be
256:             # smarter
257:             self.cc = "cl.exe"
258:             self.linker = "link.exe"
259:             self.lib = "lib.exe"
260:             self.rc = "rc.exe"
261:             self.mc = "mc.exe"
262:         else:
263:             self.__paths = self.get_msvc_paths("path")
264: 
265:             if len (self.__paths) == 0:
266:                 raise DistutilsPlatformError, \
267:                       ("Python was built with %s, "
268:                        "and extensions need to be built with the same "
269:                        "version of the compiler, but it isn't installed." % self.__product)
270: 
271:             self.cc = self.find_exe("cl.exe")
272:             self.linker = self.find_exe("link.exe")
273:             self.lib = self.find_exe("lib.exe")
274:             self.rc = self.find_exe("rc.exe")   # resource compiler
275:             self.mc = self.find_exe("mc.exe")   # message compiler
276:             self.set_path_env_var('lib')
277:             self.set_path_env_var('include')
278: 
279:         # extend the MSVC path with the current path
280:         try:
281:             for p in string.split(os.environ['path'], ';'):
282:                 self.__paths.append(p)
283:         except KeyError:
284:             pass
285:         self.__paths = normalize_and_reduce_paths(self.__paths)
286:         os.environ['path'] = string.join(self.__paths, ';')
287: 
288:         self.preprocess_options = None
289:         if self.__arch == "Intel":
290:             self.compile_options = [ '/nologo', '/Ox', '/MD', '/W3', '/GX' ,
291:                                      '/DNDEBUG']
292:             self.compile_options_debug = ['/nologo', '/Od', '/MDd', '/W3', '/GX',
293:                                           '/Z7', '/D_DEBUG']
294:         else:
295:             # Win64
296:             self.compile_options = [ '/nologo', '/Ox', '/MD', '/W3', '/GS-' ,
297:                                      '/DNDEBUG']
298:             self.compile_options_debug = ['/nologo', '/Od', '/MDd', '/W3', '/GS-',
299:                                           '/Z7', '/D_DEBUG']
300: 
301:         self.ldflags_shared = ['/DLL', '/nologo', '/INCREMENTAL:NO']
302:         if self.__version >= 7:
303:             self.ldflags_shared_debug = [
304:                 '/DLL', '/nologo', '/INCREMENTAL:no', '/DEBUG'
305:                 ]
306:         else:
307:             self.ldflags_shared_debug = [
308:                 '/DLL', '/nologo', '/INCREMENTAL:no', '/pdb:None', '/DEBUG'
309:                 ]
310:         self.ldflags_static = [ '/nologo']
311: 
312:         self.initialized = True
313: 
314:     # -- Worker methods ------------------------------------------------
315: 
316:     def object_filenames (self,
317:                           source_filenames,
318:                           strip_dir=0,
319:                           output_dir=''):
320:         # Copied from ccompiler.py, extended to return .res as 'object'-file
321:         # for .rc input file
322:         if output_dir is None: output_dir = ''
323:         obj_names = []
324:         for src_name in source_filenames:
325:             (base, ext) = os.path.splitext (src_name)
326:             base = os.path.splitdrive(base)[1] # Chop off the drive
327:             base = base[os.path.isabs(base):]  # If abs, chop off leading /
328:             if ext not in self.src_extensions:
329:                 # Better to raise an exception instead of silently continuing
330:                 # and later complain about sources and targets having
331:                 # different lengths
332:                 raise CompileError ("Don't know how to compile %s" % src_name)
333:             if strip_dir:
334:                 base = os.path.basename (base)
335:             if ext in self._rc_extensions:
336:                 obj_names.append (os.path.join (output_dir,
337:                                                 base + self.res_extension))
338:             elif ext in self._mc_extensions:
339:                 obj_names.append (os.path.join (output_dir,
340:                                                 base + self.res_extension))
341:             else:
342:                 obj_names.append (os.path.join (output_dir,
343:                                                 base + self.obj_extension))
344:         return obj_names
345: 
346:     # object_filenames ()
347: 
348: 
349:     def compile(self, sources,
350:                 output_dir=None, macros=None, include_dirs=None, debug=0,
351:                 extra_preargs=None, extra_postargs=None, depends=None):
352: 
353:         if not self.initialized: self.initialize()
354:         macros, objects, extra_postargs, pp_opts, build = \
355:                 self._setup_compile(output_dir, macros, include_dirs, sources,
356:                                     depends, extra_postargs)
357: 
358:         compile_opts = extra_preargs or []
359:         compile_opts.append ('/c')
360:         if debug:
361:             compile_opts.extend(self.compile_options_debug)
362:         else:
363:             compile_opts.extend(self.compile_options)
364: 
365:         for obj in objects:
366:             try:
367:                 src, ext = build[obj]
368:             except KeyError:
369:                 continue
370:             if debug:
371:                 # pass the full pathname to MSVC in debug mode,
372:                 # this allows the debugger to find the source file
373:                 # without asking the user to browse for it
374:                 src = os.path.abspath(src)
375: 
376:             if ext in self._c_extensions:
377:                 input_opt = "/Tc" + src
378:             elif ext in self._cpp_extensions:
379:                 input_opt = "/Tp" + src
380:             elif ext in self._rc_extensions:
381:                 # compile .RC to .RES file
382:                 input_opt = src
383:                 output_opt = "/fo" + obj
384:                 try:
385:                     self.spawn ([self.rc] + pp_opts +
386:                                 [output_opt] + [input_opt])
387:                 except DistutilsExecError, msg:
388:                     raise CompileError, msg
389:                 continue
390:             elif ext in self._mc_extensions:
391: 
392:                 # Compile .MC to .RC file to .RES file.
393:                 #   * '-h dir' specifies the directory for the
394:                 #     generated include file
395:                 #   * '-r dir' specifies the target directory of the
396:                 #     generated RC file and the binary message resource
397:                 #     it includes
398:                 #
399:                 # For now (since there are no options to change this),
400:                 # we use the source-directory for the include file and
401:                 # the build directory for the RC file and message
402:                 # resources. This works at least for win32all.
403: 
404:                 h_dir = os.path.dirname (src)
405:                 rc_dir = os.path.dirname (obj)
406:                 try:
407:                     # first compile .MC to .RC and .H file
408:                     self.spawn ([self.mc] +
409:                                 ['-h', h_dir, '-r', rc_dir] + [src])
410:                     base, _ = os.path.splitext (os.path.basename (src))
411:                     rc_file = os.path.join (rc_dir, base + '.rc')
412:                     # then compile .RC to .RES file
413:                     self.spawn ([self.rc] +
414:                                 ["/fo" + obj] + [rc_file])
415: 
416:                 except DistutilsExecError, msg:
417:                     raise CompileError, msg
418:                 continue
419:             else:
420:                 # how to handle this file?
421:                 raise CompileError (
422:                     "Don't know how to compile %s to %s" % \
423:                     (src, obj))
424: 
425:             output_opt = "/Fo" + obj
426:             try:
427:                 self.spawn ([self.cc] + compile_opts + pp_opts +
428:                             [input_opt, output_opt] +
429:                             extra_postargs)
430:             except DistutilsExecError, msg:
431:                 raise CompileError, msg
432: 
433:         return objects
434: 
435:     # compile ()
436: 
437: 
438:     def create_static_lib (self,
439:                            objects,
440:                            output_libname,
441:                            output_dir=None,
442:                            debug=0,
443:                            target_lang=None):
444: 
445:         if not self.initialized: self.initialize()
446:         (objects, output_dir) = self._fix_object_args (objects, output_dir)
447:         output_filename = \
448:             self.library_filename (output_libname, output_dir=output_dir)
449: 
450:         if self._need_link (objects, output_filename):
451:             lib_args = objects + ['/OUT:' + output_filename]
452:             if debug:
453:                 pass                    # XXX what goes here?
454:             try:
455:                 self.spawn ([self.lib] + lib_args)
456:             except DistutilsExecError, msg:
457:                 raise LibError, msg
458: 
459:         else:
460:             log.debug("skipping %s (up-to-date)", output_filename)
461: 
462:     # create_static_lib ()
463: 
464:     def link (self,
465:               target_desc,
466:               objects,
467:               output_filename,
468:               output_dir=None,
469:               libraries=None,
470:               library_dirs=None,
471:               runtime_library_dirs=None,
472:               export_symbols=None,
473:               debug=0,
474:               extra_preargs=None,
475:               extra_postargs=None,
476:               build_temp=None,
477:               target_lang=None):
478: 
479:         if not self.initialized: self.initialize()
480:         (objects, output_dir) = self._fix_object_args (objects, output_dir)
481:         (libraries, library_dirs, runtime_library_dirs) = \
482:             self._fix_lib_args (libraries, library_dirs, runtime_library_dirs)
483: 
484:         if runtime_library_dirs:
485:             self.warn ("I don't know what to do with 'runtime_library_dirs': "
486:                        + str (runtime_library_dirs))
487: 
488:         lib_opts = gen_lib_options (self,
489:                                     library_dirs, runtime_library_dirs,
490:                                     libraries)
491:         if output_dir is not None:
492:             output_filename = os.path.join (output_dir, output_filename)
493: 
494:         if self._need_link (objects, output_filename):
495: 
496:             if target_desc == CCompiler.EXECUTABLE:
497:                 if debug:
498:                     ldflags = self.ldflags_shared_debug[1:]
499:                 else:
500:                     ldflags = self.ldflags_shared[1:]
501:             else:
502:                 if debug:
503:                     ldflags = self.ldflags_shared_debug
504:                 else:
505:                     ldflags = self.ldflags_shared
506: 
507:             export_opts = []
508:             for sym in (export_symbols or []):
509:                 export_opts.append("/EXPORT:" + sym)
510: 
511:             ld_args = (ldflags + lib_opts + export_opts +
512:                        objects + ['/OUT:' + output_filename])
513: 
514:             # The MSVC linker generates .lib and .exp files, which cannot be
515:             # suppressed by any linker switches. The .lib files may even be
516:             # needed! Make sure they are generated in the temporary build
517:             # directory. Since they have different names for debug and release
518:             # builds, they can go into the same directory.
519:             if export_symbols is not None:
520:                 (dll_name, dll_ext) = os.path.splitext(
521:                     os.path.basename(output_filename))
522:                 implib_file = os.path.join(
523:                     os.path.dirname(objects[0]),
524:                     self.library_filename(dll_name))
525:                 ld_args.append ('/IMPLIB:' + implib_file)
526: 
527:             if extra_preargs:
528:                 ld_args[:0] = extra_preargs
529:             if extra_postargs:
530:                 ld_args.extend(extra_postargs)
531: 
532:             self.mkpath (os.path.dirname (output_filename))
533:             try:
534:                 self.spawn ([self.linker] + ld_args)
535:             except DistutilsExecError, msg:
536:                 raise LinkError, msg
537: 
538:         else:
539:             log.debug("skipping %s (up-to-date)", output_filename)
540: 
541:     # link ()
542: 
543: 
544:     # -- Miscellaneous methods -----------------------------------------
545:     # These are all used by the 'gen_lib_options() function, in
546:     # ccompiler.py.
547: 
548:     def library_dir_option (self, dir):
549:         return "/LIBPATH:" + dir
550: 
551:     def runtime_library_dir_option (self, dir):
552:         raise DistutilsPlatformError, \
553:               "don't know how to set runtime library search path for MSVC++"
554: 
555:     def library_option (self, lib):
556:         return self.library_filename (lib)
557: 
558: 
559:     def find_library_file (self, dirs, lib, debug=0):
560:         # Prefer a debugging library if found (and requested), but deal
561:         # with it if we don't have one.
562:         if debug:
563:             try_names = [lib + "_d", lib]
564:         else:
565:             try_names = [lib]
566:         for dir in dirs:
567:             for name in try_names:
568:                 libfile = os.path.join(dir, self.library_filename (name))
569:                 if os.path.exists(libfile):
570:                     return libfile
571:         else:
572:             # Oops, didn't find it in *any* of 'dirs'
573:             return None
574: 
575:     # find_library_file ()
576: 
577:     # Helper methods for using the MSVC registry settings
578: 
579:     def find_exe(self, exe):
580:         '''Return path to an MSVC executable program.
581: 
582:         Tries to find the program in several places: first, one of the
583:         MSVC program search paths from the registry; next, the directories
584:         in the PATH environment variable.  If any of those work, return an
585:         absolute path that is known to exist.  If none of them work, just
586:         return the original program name, 'exe'.
587:         '''
588: 
589:         for p in self.__paths:
590:             fn = os.path.join(os.path.abspath(p), exe)
591:             if os.path.isfile(fn):
592:                 return fn
593: 
594:         # didn't find it; try existing path
595:         for p in string.split(os.environ['Path'],';'):
596:             fn = os.path.join(os.path.abspath(p),exe)
597:             if os.path.isfile(fn):
598:                 return fn
599: 
600:         return exe
601: 
602:     def get_msvc_paths(self, path, platform='x86'):
603:         '''Get a list of devstudio directories (include, lib or path).
604: 
605:         Return a list of strings.  The list will be empty if unable to
606:         access the registry or appropriate registry keys not found.
607:         '''
608: 
609:         if not _can_read_reg:
610:             return []
611: 
612:         path = path + " dirs"
613:         if self.__version >= 7:
614:             key = (r"%s\%0.1f\VC\VC_OBJECTS_PLATFORM_INFO\Win32\Directories"
615:                    % (self.__root, self.__version))
616:         else:
617:             key = (r"%s\6.0\Build System\Components\Platforms"
618:                    r"\Win32 (%s)\Directories" % (self.__root, platform))
619: 
620:         for base in HKEYS:
621:             d = read_values(base, key)
622:             if d:
623:                 if self.__version >= 7:
624:                     return string.split(self.__macros.sub(d[path]), ";")
625:                 else:
626:                     return string.split(d[path], ";")
627:         # MSVC 6 seems to create the registry entries we need only when
628:         # the GUI is run.
629:         if self.__version == 6:
630:             for base in HKEYS:
631:                 if read_values(base, r"%s\6.0" % self.__root) is not None:
632:                     self.warn("It seems you have Visual Studio 6 installed, "
633:                         "but the expected registry settings are not present.\n"
634:                         "You must at least run the Visual Studio GUI once "
635:                         "so that these entries are created.")
636:                     break
637:         return []
638: 
639:     def set_path_env_var(self, name):
640:         '''Set environment variable 'name' to an MSVC path type value.
641: 
642:         This is equivalent to a SET command prior to execution of spawned
643:         commands.
644:         '''
645: 
646:         if name == "lib":
647:             p = self.get_msvc_paths("library")
648:         else:
649:             p = self.get_msvc_paths(name)
650:         if p:
651:             os.environ[name] = string.join(p, ';')
652: 
653: 
654: if get_build_version() >= 8.0:
655:     log.debug("Importing new compiler from distutils.msvc9compiler")
656:     OldMSVCCompiler = MSVCCompiler
657:     from distutils.msvc9compiler import MSVCCompiler
658:     # get_build_architecture not really relevant now we support cross-compile
659:     from distutils.msvc9compiler import MacroExpander
660: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_5039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'str', 'distutils.msvccompiler\n\nContains MSVCCompiler, an implementation of the abstract CCompiler class\nfor the Microsoft Visual Studio.\n')

# Assigning a Str to a Name (line 11):

# Assigning a Str to a Name (line 11):
str_5040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), '__revision__', str_5040)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import sys' statement (line 13)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import os' statement (line 14)
import os

import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'import string' statement (line 15)
import string

import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'string', string, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from distutils.errors import DistutilsExecError, DistutilsPlatformError, CompileError, LibError, LinkError' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_5041 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.errors')

if (type(import_5041) is not StypyTypeError):

    if (import_5041 != 'pyd_module'):
        __import__(import_5041)
        sys_modules_5042 = sys.modules[import_5041]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.errors', sys_modules_5042.module_type_store, module_type_store, ['DistutilsExecError', 'DistutilsPlatformError', 'CompileError', 'LibError', 'LinkError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_5042, sys_modules_5042.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsExecError, DistutilsPlatformError, CompileError, LibError, LinkError

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.errors', None, module_type_store, ['DistutilsExecError', 'DistutilsPlatformError', 'CompileError', 'LibError', 'LinkError'], [DistutilsExecError, DistutilsPlatformError, CompileError, LibError, LinkError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.errors', import_5041)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from distutils.ccompiler import CCompiler, gen_lib_options' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_5043 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.ccompiler')

if (type(import_5043) is not StypyTypeError):

    if (import_5043 != 'pyd_module'):
        __import__(import_5043)
        sys_modules_5044 = sys.modules[import_5043]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.ccompiler', sys_modules_5044.module_type_store, module_type_store, ['CCompiler', 'gen_lib_options'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_5044, sys_modules_5044.module_type_store, module_type_store)
    else:
        from distutils.ccompiler import CCompiler, gen_lib_options

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.ccompiler', None, module_type_store, ['CCompiler', 'gen_lib_options'], [CCompiler, gen_lib_options])

else:
    # Assigning a type to the variable 'distutils.ccompiler' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.ccompiler', import_5043)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from distutils import log' statement (line 20)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils', None, module_type_store, ['log'], [log])


# Assigning a Num to a Name (line 22):

# Assigning a Num to a Name (line 22):
int_5045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'int')
# Assigning a type to the variable '_can_read_reg' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), '_can_read_reg', int_5045)


# SSA begins for try-except statement (line 23)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 4))

# 'import _winreg' statement (line 24)
import _winreg

import_module(stypy.reporting.localization.Localization(__file__, 24, 4), '_winreg', _winreg, module_type_store)


# Assigning a Num to a Name (line 26):

# Assigning a Num to a Name (line 26):
int_5046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 20), 'int')
# Assigning a type to the variable '_can_read_reg' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), '_can_read_reg', int_5046)

# Assigning a Name to a Name (line 27):

# Assigning a Name to a Name (line 27):
# Getting the type of '_winreg' (line 27)
_winreg_5047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 15), '_winreg')
# Assigning a type to the variable 'hkey_mod' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'hkey_mod', _winreg_5047)

# Assigning a Attribute to a Name (line 29):

# Assigning a Attribute to a Name (line 29):
# Getting the type of '_winreg' (line 29)
_winreg_5048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 19), '_winreg')
# Obtaining the member 'OpenKeyEx' of a type (line 29)
OpenKeyEx_5049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 19), _winreg_5048, 'OpenKeyEx')
# Assigning a type to the variable 'RegOpenKeyEx' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'RegOpenKeyEx', OpenKeyEx_5049)

# Assigning a Attribute to a Name (line 30):

# Assigning a Attribute to a Name (line 30):
# Getting the type of '_winreg' (line 30)
_winreg_5050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 17), '_winreg')
# Obtaining the member 'EnumKey' of a type (line 30)
EnumKey_5051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 17), _winreg_5050, 'EnumKey')
# Assigning a type to the variable 'RegEnumKey' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'RegEnumKey', EnumKey_5051)

# Assigning a Attribute to a Name (line 31):

# Assigning a Attribute to a Name (line 31):
# Getting the type of '_winreg' (line 31)
_winreg_5052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 19), '_winreg')
# Obtaining the member 'EnumValue' of a type (line 31)
EnumValue_5053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 19), _winreg_5052, 'EnumValue')
# Assigning a type to the variable 'RegEnumValue' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'RegEnumValue', EnumValue_5053)

# Assigning a Attribute to a Name (line 32):

# Assigning a Attribute to a Name (line 32):
# Getting the type of '_winreg' (line 32)
_winreg_5054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), '_winreg')
# Obtaining the member 'error' of a type (line 32)
error_5055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 15), _winreg_5054, 'error')
# Assigning a type to the variable 'RegError' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'RegError', error_5055)
# SSA branch for the except part of a try statement (line 23)
# SSA branch for the except 'ImportError' branch of a try statement (line 23)
module_type_store.open_ssa_branch('except')


# SSA begins for try-except statement (line 35)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 36, 8))

# 'import win32api' statement (line 36)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_5056 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 36, 8), 'win32api')

if (type(import_5056) is not StypyTypeError):

    if (import_5056 != 'pyd_module'):
        __import__(import_5056)
        sys_modules_5057 = sys.modules[import_5056]
        import_module(stypy.reporting.localization.Localization(__file__, 36, 8), 'win32api', sys_modules_5057.module_type_store, module_type_store)
    else:
        import win32api

        import_module(stypy.reporting.localization.Localization(__file__, 36, 8), 'win32api', win32api, module_type_store)

else:
    # Assigning a type to the variable 'win32api' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'win32api', import_5056)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 37, 8))

# 'import win32con' statement (line 37)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_5058 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 37, 8), 'win32con')

if (type(import_5058) is not StypyTypeError):

    if (import_5058 != 'pyd_module'):
        __import__(import_5058)
        sys_modules_5059 = sys.modules[import_5058]
        import_module(stypy.reporting.localization.Localization(__file__, 37, 8), 'win32con', sys_modules_5059.module_type_store, module_type_store)
    else:
        import win32con

        import_module(stypy.reporting.localization.Localization(__file__, 37, 8), 'win32con', win32con, module_type_store)

else:
    # Assigning a type to the variable 'win32con' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'win32con', import_5058)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')


# Assigning a Num to a Name (line 38):

# Assigning a Num to a Name (line 38):
int_5060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 24), 'int')
# Assigning a type to the variable '_can_read_reg' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), '_can_read_reg', int_5060)

# Assigning a Name to a Name (line 39):

# Assigning a Name to a Name (line 39):
# Getting the type of 'win32con' (line 39)
win32con_5061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 19), 'win32con')
# Assigning a type to the variable 'hkey_mod' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'hkey_mod', win32con_5061)

# Assigning a Attribute to a Name (line 41):

# Assigning a Attribute to a Name (line 41):
# Getting the type of 'win32api' (line 41)
win32api_5062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 23), 'win32api')
# Obtaining the member 'RegOpenKeyEx' of a type (line 41)
RegOpenKeyEx_5063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 23), win32api_5062, 'RegOpenKeyEx')
# Assigning a type to the variable 'RegOpenKeyEx' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'RegOpenKeyEx', RegOpenKeyEx_5063)

# Assigning a Attribute to a Name (line 42):

# Assigning a Attribute to a Name (line 42):
# Getting the type of 'win32api' (line 42)
win32api_5064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 21), 'win32api')
# Obtaining the member 'RegEnumKey' of a type (line 42)
RegEnumKey_5065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 21), win32api_5064, 'RegEnumKey')
# Assigning a type to the variable 'RegEnumKey' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'RegEnumKey', RegEnumKey_5065)

# Assigning a Attribute to a Name (line 43):

# Assigning a Attribute to a Name (line 43):
# Getting the type of 'win32api' (line 43)
win32api_5066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 23), 'win32api')
# Obtaining the member 'RegEnumValue' of a type (line 43)
RegEnumValue_5067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 23), win32api_5066, 'RegEnumValue')
# Assigning a type to the variable 'RegEnumValue' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'RegEnumValue', RegEnumValue_5067)

# Assigning a Attribute to a Name (line 44):

# Assigning a Attribute to a Name (line 44):
# Getting the type of 'win32api' (line 44)
win32api_5068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 19), 'win32api')
# Obtaining the member 'error' of a type (line 44)
error_5069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 19), win32api_5068, 'error')
# Assigning a type to the variable 'RegError' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'RegError', error_5069)
# SSA branch for the except part of a try statement (line 35)
# SSA branch for the except 'ImportError' branch of a try statement (line 35)
module_type_store.open_ssa_branch('except')

# Call to info(...): (line 47)
# Processing the call arguments (line 47)
str_5072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 17), 'str', "Warning: Can't read registry to find the necessary compiler setting\nMake sure that Python modules _winreg, win32api or win32con are installed.")
# Processing the call keyword arguments (line 47)
kwargs_5073 = {}
# Getting the type of 'log' (line 47)
log_5070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'log', False)
# Obtaining the member 'info' of a type (line 47)
info_5071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), log_5070, 'info')
# Calling info(args, kwargs) (line 47)
info_call_result_5074 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), info_5071, *[str_5072], **kwargs_5073)

pass
# SSA join for try-except statement (line 35)
module_type_store = module_type_store.join_ssa_context()

# SSA join for try-except statement (line 23)
module_type_store = module_type_store.join_ssa_context()


# Getting the type of '_can_read_reg' (line 53)
_can_read_reg_5075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 3), '_can_read_reg')
# Testing the type of an if condition (line 53)
if_condition_5076 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 53, 0), _can_read_reg_5075)
# Assigning a type to the variable 'if_condition_5076' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'if_condition_5076', if_condition_5076)
# SSA begins for if statement (line 53)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Tuple to a Name (line 54):

# Assigning a Tuple to a Name (line 54):

# Obtaining an instance of the builtin type 'tuple' (line 54)
tuple_5077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 54)
# Adding element type (line 54)
# Getting the type of 'hkey_mod' (line 54)
hkey_mod_5078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 13), 'hkey_mod')
# Obtaining the member 'HKEY_USERS' of a type (line 54)
HKEY_USERS_5079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 13), hkey_mod_5078, 'HKEY_USERS')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 13), tuple_5077, HKEY_USERS_5079)
# Adding element type (line 54)
# Getting the type of 'hkey_mod' (line 55)
hkey_mod_5080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 13), 'hkey_mod')
# Obtaining the member 'HKEY_CURRENT_USER' of a type (line 55)
HKEY_CURRENT_USER_5081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 13), hkey_mod_5080, 'HKEY_CURRENT_USER')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 13), tuple_5077, HKEY_CURRENT_USER_5081)
# Adding element type (line 54)
# Getting the type of 'hkey_mod' (line 56)
hkey_mod_5082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 13), 'hkey_mod')
# Obtaining the member 'HKEY_LOCAL_MACHINE' of a type (line 56)
HKEY_LOCAL_MACHINE_5083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 13), hkey_mod_5082, 'HKEY_LOCAL_MACHINE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 13), tuple_5077, HKEY_LOCAL_MACHINE_5083)
# Adding element type (line 54)
# Getting the type of 'hkey_mod' (line 57)
hkey_mod_5084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 13), 'hkey_mod')
# Obtaining the member 'HKEY_CLASSES_ROOT' of a type (line 57)
HKEY_CLASSES_ROOT_5085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 13), hkey_mod_5084, 'HKEY_CLASSES_ROOT')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 13), tuple_5077, HKEY_CLASSES_ROOT_5085)

# Assigning a type to the variable 'HKEYS' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'HKEYS', tuple_5077)
# SSA join for if statement (line 53)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def read_keys(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'read_keys'
    module_type_store = module_type_store.open_function_context('read_keys', 59, 0, False)
    
    # Passed parameters checking function
    read_keys.stypy_localization = localization
    read_keys.stypy_type_of_self = None
    read_keys.stypy_type_store = module_type_store
    read_keys.stypy_function_name = 'read_keys'
    read_keys.stypy_param_names_list = ['base', 'key']
    read_keys.stypy_varargs_param_name = None
    read_keys.stypy_kwargs_param_name = None
    read_keys.stypy_call_defaults = defaults
    read_keys.stypy_call_varargs = varargs
    read_keys.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'read_keys', ['base', 'key'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'read_keys', localization, ['base', 'key'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'read_keys(...)' code ##################

    str_5086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 4), 'str', 'Return list of registry keys.')
    
    
    # SSA begins for try-except statement (line 62)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 63):
    
    # Assigning a Call to a Name (line 63):
    
    # Call to RegOpenKeyEx(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'base' (line 63)
    base_5088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 30), 'base', False)
    # Getting the type of 'key' (line 63)
    key_5089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 36), 'key', False)
    # Processing the call keyword arguments (line 63)
    kwargs_5090 = {}
    # Getting the type of 'RegOpenKeyEx' (line 63)
    RegOpenKeyEx_5087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 17), 'RegOpenKeyEx', False)
    # Calling RegOpenKeyEx(args, kwargs) (line 63)
    RegOpenKeyEx_call_result_5091 = invoke(stypy.reporting.localization.Localization(__file__, 63, 17), RegOpenKeyEx_5087, *[base_5088, key_5089], **kwargs_5090)
    
    # Assigning a type to the variable 'handle' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'handle', RegOpenKeyEx_call_result_5091)
    # SSA branch for the except part of a try statement (line 62)
    # SSA branch for the except 'RegError' branch of a try statement (line 62)
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'None' (line 65)
    None_5092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'stypy_return_type', None_5092)
    # SSA join for try-except statement (line 62)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 66):
    
    # Assigning a List to a Name (line 66):
    
    # Obtaining an instance of the builtin type 'list' (line 66)
    list_5093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 66)
    
    # Assigning a type to the variable 'L' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'L', list_5093)
    
    # Assigning a Num to a Name (line 67):
    
    # Assigning a Num to a Name (line 67):
    int_5094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 8), 'int')
    # Assigning a type to the variable 'i' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'i', int_5094)
    
    int_5095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 10), 'int')
    # Testing the type of an if condition (line 68)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 4), int_5095)
    # SSA begins for while statement (line 68)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    
    # SSA begins for try-except statement (line 69)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 70):
    
    # Assigning a Call to a Name (line 70):
    
    # Call to RegEnumKey(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'handle' (line 70)
    handle_5097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 27), 'handle', False)
    # Getting the type of 'i' (line 70)
    i_5098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 35), 'i', False)
    # Processing the call keyword arguments (line 70)
    kwargs_5099 = {}
    # Getting the type of 'RegEnumKey' (line 70)
    RegEnumKey_5096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'RegEnumKey', False)
    # Calling RegEnumKey(args, kwargs) (line 70)
    RegEnumKey_call_result_5100 = invoke(stypy.reporting.localization.Localization(__file__, 70, 16), RegEnumKey_5096, *[handle_5097, i_5098], **kwargs_5099)
    
    # Assigning a type to the variable 'k' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'k', RegEnumKey_call_result_5100)
    # SSA branch for the except part of a try statement (line 69)
    # SSA branch for the except 'RegError' branch of a try statement (line 69)
    module_type_store.open_ssa_branch('except')
    # SSA join for try-except statement (line 69)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'k' (line 73)
    k_5103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 17), 'k', False)
    # Processing the call keyword arguments (line 73)
    kwargs_5104 = {}
    # Getting the type of 'L' (line 73)
    L_5101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'L', False)
    # Obtaining the member 'append' of a type (line 73)
    append_5102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), L_5101, 'append')
    # Calling append(args, kwargs) (line 73)
    append_call_result_5105 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), append_5102, *[k_5103], **kwargs_5104)
    
    
    # Assigning a BinOp to a Name (line 74):
    
    # Assigning a BinOp to a Name (line 74):
    # Getting the type of 'i' (line 74)
    i_5106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'i')
    int_5107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 16), 'int')
    # Applying the binary operator '+' (line 74)
    result_add_5108 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 12), '+', i_5106, int_5107)
    
    # Assigning a type to the variable 'i' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'i', result_add_5108)
    # SSA join for while statement (line 68)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'L' (line 75)
    L_5109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 11), 'L')
    # Assigning a type to the variable 'stypy_return_type' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type', L_5109)
    
    # ################# End of 'read_keys(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'read_keys' in the type store
    # Getting the type of 'stypy_return_type' (line 59)
    stypy_return_type_5110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5110)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'read_keys'
    return stypy_return_type_5110

# Assigning a type to the variable 'read_keys' (line 59)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'read_keys', read_keys)

@norecursion
def read_values(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'read_values'
    module_type_store = module_type_store.open_function_context('read_values', 77, 0, False)
    
    # Passed parameters checking function
    read_values.stypy_localization = localization
    read_values.stypy_type_of_self = None
    read_values.stypy_type_store = module_type_store
    read_values.stypy_function_name = 'read_values'
    read_values.stypy_param_names_list = ['base', 'key']
    read_values.stypy_varargs_param_name = None
    read_values.stypy_kwargs_param_name = None
    read_values.stypy_call_defaults = defaults
    read_values.stypy_call_varargs = varargs
    read_values.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'read_values', ['base', 'key'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'read_values', localization, ['base', 'key'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'read_values(...)' code ##################

    str_5111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, (-1)), 'str', 'Return dict of registry keys and values.\n\n    All names are converted to lowercase.\n    ')
    
    
    # SSA begins for try-except statement (line 82)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 83):
    
    # Assigning a Call to a Name (line 83):
    
    # Call to RegOpenKeyEx(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'base' (line 83)
    base_5113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 30), 'base', False)
    # Getting the type of 'key' (line 83)
    key_5114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 36), 'key', False)
    # Processing the call keyword arguments (line 83)
    kwargs_5115 = {}
    # Getting the type of 'RegOpenKeyEx' (line 83)
    RegOpenKeyEx_5112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 17), 'RegOpenKeyEx', False)
    # Calling RegOpenKeyEx(args, kwargs) (line 83)
    RegOpenKeyEx_call_result_5116 = invoke(stypy.reporting.localization.Localization(__file__, 83, 17), RegOpenKeyEx_5112, *[base_5113, key_5114], **kwargs_5115)
    
    # Assigning a type to the variable 'handle' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'handle', RegOpenKeyEx_call_result_5116)
    # SSA branch for the except part of a try statement (line 82)
    # SSA branch for the except 'RegError' branch of a try statement (line 82)
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'None' (line 85)
    None_5117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'stypy_return_type', None_5117)
    # SSA join for try-except statement (line 82)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Dict to a Name (line 86):
    
    # Assigning a Dict to a Name (line 86):
    
    # Obtaining an instance of the builtin type 'dict' (line 86)
    dict_5118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 8), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 86)
    
    # Assigning a type to the variable 'd' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'd', dict_5118)
    
    # Assigning a Num to a Name (line 87):
    
    # Assigning a Num to a Name (line 87):
    int_5119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 8), 'int')
    # Assigning a type to the variable 'i' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'i', int_5119)
    
    int_5120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 10), 'int')
    # Testing the type of an if condition (line 88)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 4), int_5120)
    # SSA begins for while statement (line 88)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    
    # SSA begins for try-except statement (line 89)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Tuple (line 90):
    
    # Assigning a Subscript to a Name (line 90):
    
    # Obtaining the type of the subscript
    int_5121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 12), 'int')
    
    # Call to RegEnumValue(...): (line 90)
    # Processing the call arguments (line 90)
    # Getting the type of 'handle' (line 90)
    handle_5123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 45), 'handle', False)
    # Getting the type of 'i' (line 90)
    i_5124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 53), 'i', False)
    # Processing the call keyword arguments (line 90)
    kwargs_5125 = {}
    # Getting the type of 'RegEnumValue' (line 90)
    RegEnumValue_5122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 32), 'RegEnumValue', False)
    # Calling RegEnumValue(args, kwargs) (line 90)
    RegEnumValue_call_result_5126 = invoke(stypy.reporting.localization.Localization(__file__, 90, 32), RegEnumValue_5122, *[handle_5123, i_5124], **kwargs_5125)
    
    # Obtaining the member '__getitem__' of a type (line 90)
    getitem___5127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), RegEnumValue_call_result_5126, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 90)
    subscript_call_result_5128 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), getitem___5127, int_5121)
    
    # Assigning a type to the variable 'tuple_var_assignment_5014' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'tuple_var_assignment_5014', subscript_call_result_5128)
    
    # Assigning a Subscript to a Name (line 90):
    
    # Obtaining the type of the subscript
    int_5129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 12), 'int')
    
    # Call to RegEnumValue(...): (line 90)
    # Processing the call arguments (line 90)
    # Getting the type of 'handle' (line 90)
    handle_5131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 45), 'handle', False)
    # Getting the type of 'i' (line 90)
    i_5132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 53), 'i', False)
    # Processing the call keyword arguments (line 90)
    kwargs_5133 = {}
    # Getting the type of 'RegEnumValue' (line 90)
    RegEnumValue_5130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 32), 'RegEnumValue', False)
    # Calling RegEnumValue(args, kwargs) (line 90)
    RegEnumValue_call_result_5134 = invoke(stypy.reporting.localization.Localization(__file__, 90, 32), RegEnumValue_5130, *[handle_5131, i_5132], **kwargs_5133)
    
    # Obtaining the member '__getitem__' of a type (line 90)
    getitem___5135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), RegEnumValue_call_result_5134, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 90)
    subscript_call_result_5136 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), getitem___5135, int_5129)
    
    # Assigning a type to the variable 'tuple_var_assignment_5015' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'tuple_var_assignment_5015', subscript_call_result_5136)
    
    # Assigning a Subscript to a Name (line 90):
    
    # Obtaining the type of the subscript
    int_5137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 12), 'int')
    
    # Call to RegEnumValue(...): (line 90)
    # Processing the call arguments (line 90)
    # Getting the type of 'handle' (line 90)
    handle_5139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 45), 'handle', False)
    # Getting the type of 'i' (line 90)
    i_5140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 53), 'i', False)
    # Processing the call keyword arguments (line 90)
    kwargs_5141 = {}
    # Getting the type of 'RegEnumValue' (line 90)
    RegEnumValue_5138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 32), 'RegEnumValue', False)
    # Calling RegEnumValue(args, kwargs) (line 90)
    RegEnumValue_call_result_5142 = invoke(stypy.reporting.localization.Localization(__file__, 90, 32), RegEnumValue_5138, *[handle_5139, i_5140], **kwargs_5141)
    
    # Obtaining the member '__getitem__' of a type (line 90)
    getitem___5143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), RegEnumValue_call_result_5142, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 90)
    subscript_call_result_5144 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), getitem___5143, int_5137)
    
    # Assigning a type to the variable 'tuple_var_assignment_5016' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'tuple_var_assignment_5016', subscript_call_result_5144)
    
    # Assigning a Name to a Name (line 90):
    # Getting the type of 'tuple_var_assignment_5014' (line 90)
    tuple_var_assignment_5014_5145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'tuple_var_assignment_5014')
    # Assigning a type to the variable 'name' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'name', tuple_var_assignment_5014_5145)
    
    # Assigning a Name to a Name (line 90):
    # Getting the type of 'tuple_var_assignment_5015' (line 90)
    tuple_var_assignment_5015_5146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'tuple_var_assignment_5015')
    # Assigning a type to the variable 'value' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 18), 'value', tuple_var_assignment_5015_5146)
    
    # Assigning a Name to a Name (line 90):
    # Getting the type of 'tuple_var_assignment_5016' (line 90)
    tuple_var_assignment_5016_5147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'tuple_var_assignment_5016')
    # Assigning a type to the variable 'type' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), 'type', tuple_var_assignment_5016_5147)
    # SSA branch for the except part of a try statement (line 89)
    # SSA branch for the except 'RegError' branch of a try statement (line 89)
    module_type_store.open_ssa_branch('except')
    # SSA join for try-except statement (line 89)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 93):
    
    # Assigning a Call to a Name (line 93):
    
    # Call to lower(...): (line 93)
    # Processing the call keyword arguments (line 93)
    kwargs_5150 = {}
    # Getting the type of 'name' (line 93)
    name_5148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 15), 'name', False)
    # Obtaining the member 'lower' of a type (line 93)
    lower_5149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 15), name_5148, 'lower')
    # Calling lower(args, kwargs) (line 93)
    lower_call_result_5151 = invoke(stypy.reporting.localization.Localization(__file__, 93, 15), lower_5149, *[], **kwargs_5150)
    
    # Assigning a type to the variable 'name' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'name', lower_call_result_5151)
    
    # Assigning a Call to a Subscript (line 94):
    
    # Assigning a Call to a Subscript (line 94):
    
    # Call to convert_mbcs(...): (line 94)
    # Processing the call arguments (line 94)
    # Getting the type of 'value' (line 94)
    value_5153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 45), 'value', False)
    # Processing the call keyword arguments (line 94)
    kwargs_5154 = {}
    # Getting the type of 'convert_mbcs' (line 94)
    convert_mbcs_5152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 32), 'convert_mbcs', False)
    # Calling convert_mbcs(args, kwargs) (line 94)
    convert_mbcs_call_result_5155 = invoke(stypy.reporting.localization.Localization(__file__, 94, 32), convert_mbcs_5152, *[value_5153], **kwargs_5154)
    
    # Getting the type of 'd' (line 94)
    d_5156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'd')
    
    # Call to convert_mbcs(...): (line 94)
    # Processing the call arguments (line 94)
    # Getting the type of 'name' (line 94)
    name_5158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 23), 'name', False)
    # Processing the call keyword arguments (line 94)
    kwargs_5159 = {}
    # Getting the type of 'convert_mbcs' (line 94)
    convert_mbcs_5157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 10), 'convert_mbcs', False)
    # Calling convert_mbcs(args, kwargs) (line 94)
    convert_mbcs_call_result_5160 = invoke(stypy.reporting.localization.Localization(__file__, 94, 10), convert_mbcs_5157, *[name_5158], **kwargs_5159)
    
    # Storing an element on a container (line 94)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 8), d_5156, (convert_mbcs_call_result_5160, convert_mbcs_call_result_5155))
    
    # Assigning a BinOp to a Name (line 95):
    
    # Assigning a BinOp to a Name (line 95):
    # Getting the type of 'i' (line 95)
    i_5161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'i')
    int_5162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 16), 'int')
    # Applying the binary operator '+' (line 95)
    result_add_5163 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 12), '+', i_5161, int_5162)
    
    # Assigning a type to the variable 'i' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'i', result_add_5163)
    # SSA join for while statement (line 88)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'd' (line 96)
    d_5164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 11), 'd')
    # Assigning a type to the variable 'stypy_return_type' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'stypy_return_type', d_5164)
    
    # ################# End of 'read_values(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'read_values' in the type store
    # Getting the type of 'stypy_return_type' (line 77)
    stypy_return_type_5165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5165)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'read_values'
    return stypy_return_type_5165

# Assigning a type to the variable 'read_values' (line 77)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'read_values', read_values)

@norecursion
def convert_mbcs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'convert_mbcs'
    module_type_store = module_type_store.open_function_context('convert_mbcs', 98, 0, False)
    
    # Passed parameters checking function
    convert_mbcs.stypy_localization = localization
    convert_mbcs.stypy_type_of_self = None
    convert_mbcs.stypy_type_store = module_type_store
    convert_mbcs.stypy_function_name = 'convert_mbcs'
    convert_mbcs.stypy_param_names_list = ['s']
    convert_mbcs.stypy_varargs_param_name = None
    convert_mbcs.stypy_kwargs_param_name = None
    convert_mbcs.stypy_call_defaults = defaults
    convert_mbcs.stypy_call_varargs = varargs
    convert_mbcs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'convert_mbcs', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'convert_mbcs', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'convert_mbcs(...)' code ##################

    
    # Assigning a Call to a Name (line 99):
    
    # Assigning a Call to a Name (line 99):
    
    # Call to getattr(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 's' (line 99)
    s_5167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 18), 's', False)
    str_5168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 21), 'str', 'encode')
    # Getting the type of 'None' (line 99)
    None_5169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 31), 'None', False)
    # Processing the call keyword arguments (line 99)
    kwargs_5170 = {}
    # Getting the type of 'getattr' (line 99)
    getattr_5166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 10), 'getattr', False)
    # Calling getattr(args, kwargs) (line 99)
    getattr_call_result_5171 = invoke(stypy.reporting.localization.Localization(__file__, 99, 10), getattr_5166, *[s_5167, str_5168, None_5169], **kwargs_5170)
    
    # Assigning a type to the variable 'enc' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'enc', getattr_call_result_5171)
    
    # Type idiom detected: calculating its left and rigth part (line 100)
    # Getting the type of 'enc' (line 100)
    enc_5172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'enc')
    # Getting the type of 'None' (line 100)
    None_5173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 18), 'None')
    
    (may_be_5174, more_types_in_union_5175) = may_not_be_none(enc_5172, None_5173)

    if may_be_5174:

        if more_types_in_union_5175:
            # Runtime conditional SSA (line 100)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # SSA begins for try-except statement (line 101)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 102):
        
        # Assigning a Call to a Name (line 102):
        
        # Call to enc(...): (line 102)
        # Processing the call arguments (line 102)
        str_5177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 20), 'str', 'mbcs')
        # Processing the call keyword arguments (line 102)
        kwargs_5178 = {}
        # Getting the type of 'enc' (line 102)
        enc_5176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'enc', False)
        # Calling enc(args, kwargs) (line 102)
        enc_call_result_5179 = invoke(stypy.reporting.localization.Localization(__file__, 102, 16), enc_5176, *[str_5177], **kwargs_5178)
        
        # Assigning a type to the variable 's' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 's', enc_call_result_5179)
        # SSA branch for the except part of a try statement (line 101)
        # SSA branch for the except 'UnicodeError' branch of a try statement (line 101)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 101)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_5175:
            # SSA join for if statement (line 100)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 's' (line 105)
    s_5180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 11), 's')
    # Assigning a type to the variable 'stypy_return_type' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type', s_5180)
    
    # ################# End of 'convert_mbcs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'convert_mbcs' in the type store
    # Getting the type of 'stypy_return_type' (line 98)
    stypy_return_type_5181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5181)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'convert_mbcs'
    return stypy_return_type_5181

# Assigning a type to the variable 'convert_mbcs' (line 98)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'convert_mbcs', convert_mbcs)
# Declaration of the 'MacroExpander' class

class MacroExpander:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 109, 4, False)
        # Assigning a type to the variable 'self' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MacroExpander.__init__', ['version'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['version'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Dict to a Attribute (line 110):
        
        # Assigning a Dict to a Attribute (line 110):
        
        # Obtaining an instance of the builtin type 'dict' (line 110)
        dict_5182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 22), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 110)
        
        # Getting the type of 'self' (line 110)
        self_5183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'self')
        # Setting the type of the member 'macros' of a type (line 110)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), self_5183, 'macros', dict_5182)
        
        # Call to load_macros(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'version' (line 111)
        version_5186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 25), 'version', False)
        # Processing the call keyword arguments (line 111)
        kwargs_5187 = {}
        # Getting the type of 'self' (line 111)
        self_5184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'self', False)
        # Obtaining the member 'load_macros' of a type (line 111)
        load_macros_5185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), self_5184, 'load_macros')
        # Calling load_macros(args, kwargs) (line 111)
        load_macros_call_result_5188 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), load_macros_5185, *[version_5186], **kwargs_5187)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_macro(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_macro'
        module_type_store = module_type_store.open_function_context('set_macro', 113, 4, False)
        # Assigning a type to the variable 'self' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MacroExpander.set_macro.__dict__.__setitem__('stypy_localization', localization)
        MacroExpander.set_macro.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MacroExpander.set_macro.__dict__.__setitem__('stypy_type_store', module_type_store)
        MacroExpander.set_macro.__dict__.__setitem__('stypy_function_name', 'MacroExpander.set_macro')
        MacroExpander.set_macro.__dict__.__setitem__('stypy_param_names_list', ['macro', 'path', 'key'])
        MacroExpander.set_macro.__dict__.__setitem__('stypy_varargs_param_name', None)
        MacroExpander.set_macro.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MacroExpander.set_macro.__dict__.__setitem__('stypy_call_defaults', defaults)
        MacroExpander.set_macro.__dict__.__setitem__('stypy_call_varargs', varargs)
        MacroExpander.set_macro.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MacroExpander.set_macro.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MacroExpander.set_macro', ['macro', 'path', 'key'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_macro', localization, ['macro', 'path', 'key'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_macro(...)' code ##################

        
        # Getting the type of 'HKEYS' (line 114)
        HKEYS_5189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 20), 'HKEYS')
        # Testing the type of a for loop iterable (line 114)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 114, 8), HKEYS_5189)
        # Getting the type of the for loop variable (line 114)
        for_loop_var_5190 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 114, 8), HKEYS_5189)
        # Assigning a type to the variable 'base' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'base', for_loop_var_5190)
        # SSA begins for a for statement (line 114)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 115):
        
        # Assigning a Call to a Name (line 115):
        
        # Call to read_values(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'base' (line 115)
        base_5192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 28), 'base', False)
        # Getting the type of 'path' (line 115)
        path_5193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 34), 'path', False)
        # Processing the call keyword arguments (line 115)
        kwargs_5194 = {}
        # Getting the type of 'read_values' (line 115)
        read_values_5191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'read_values', False)
        # Calling read_values(args, kwargs) (line 115)
        read_values_call_result_5195 = invoke(stypy.reporting.localization.Localization(__file__, 115, 16), read_values_5191, *[base_5192, path_5193], **kwargs_5194)
        
        # Assigning a type to the variable 'd' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'd', read_values_call_result_5195)
        
        # Getting the type of 'd' (line 116)
        d_5196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 15), 'd')
        # Testing the type of an if condition (line 116)
        if_condition_5197 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 12), d_5196)
        # Assigning a type to the variable 'if_condition_5197' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'if_condition_5197', if_condition_5197)
        # SSA begins for if statement (line 116)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Subscript (line 117):
        
        # Assigning a Subscript to a Subscript (line 117):
        
        # Obtaining the type of the subscript
        # Getting the type of 'key' (line 117)
        key_5198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 49), 'key')
        # Getting the type of 'd' (line 117)
        d_5199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 47), 'd')
        # Obtaining the member '__getitem__' of a type (line 117)
        getitem___5200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 47), d_5199, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 117)
        subscript_call_result_5201 = invoke(stypy.reporting.localization.Localization(__file__, 117, 47), getitem___5200, key_5198)
        
        # Getting the type of 'self' (line 117)
        self_5202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'self')
        # Obtaining the member 'macros' of a type (line 117)
        macros_5203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 16), self_5202, 'macros')
        str_5204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 28), 'str', '$(%s)')
        # Getting the type of 'macro' (line 117)
        macro_5205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 38), 'macro')
        # Applying the binary operator '%' (line 117)
        result_mod_5206 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 28), '%', str_5204, macro_5205)
        
        # Storing an element on a container (line 117)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 16), macros_5203, (result_mod_5206, subscript_call_result_5201))
        # SSA join for if statement (line 116)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'set_macro(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_macro' in the type store
        # Getting the type of 'stypy_return_type' (line 113)
        stypy_return_type_5207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5207)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_macro'
        return stypy_return_type_5207


    @norecursion
    def load_macros(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'load_macros'
        module_type_store = module_type_store.open_function_context('load_macros', 120, 4, False)
        # Assigning a type to the variable 'self' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MacroExpander.load_macros.__dict__.__setitem__('stypy_localization', localization)
        MacroExpander.load_macros.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MacroExpander.load_macros.__dict__.__setitem__('stypy_type_store', module_type_store)
        MacroExpander.load_macros.__dict__.__setitem__('stypy_function_name', 'MacroExpander.load_macros')
        MacroExpander.load_macros.__dict__.__setitem__('stypy_param_names_list', ['version'])
        MacroExpander.load_macros.__dict__.__setitem__('stypy_varargs_param_name', None)
        MacroExpander.load_macros.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MacroExpander.load_macros.__dict__.__setitem__('stypy_call_defaults', defaults)
        MacroExpander.load_macros.__dict__.__setitem__('stypy_call_varargs', varargs)
        MacroExpander.load_macros.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MacroExpander.load_macros.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MacroExpander.load_macros', ['version'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'load_macros', localization, ['version'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'load_macros(...)' code ##################

        
        # Assigning a BinOp to a Name (line 121):
        
        # Assigning a BinOp to a Name (line 121):
        str_5208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 17), 'str', 'Software\\Microsoft\\VisualStudio\\%0.1f')
        # Getting the type of 'version' (line 121)
        version_5209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 60), 'version')
        # Applying the binary operator '%' (line 121)
        result_mod_5210 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 17), '%', str_5208, version_5209)
        
        # Assigning a type to the variable 'vsbase' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'vsbase', result_mod_5210)
        
        # Call to set_macro(...): (line 122)
        # Processing the call arguments (line 122)
        str_5213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 23), 'str', 'VCInstallDir')
        # Getting the type of 'vsbase' (line 122)
        vsbase_5214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 39), 'vsbase', False)
        str_5215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 48), 'str', '\\Setup\\VC')
        # Applying the binary operator '+' (line 122)
        result_add_5216 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 39), '+', vsbase_5214, str_5215)
        
        str_5217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 62), 'str', 'productdir')
        # Processing the call keyword arguments (line 122)
        kwargs_5218 = {}
        # Getting the type of 'self' (line 122)
        self_5211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'self', False)
        # Obtaining the member 'set_macro' of a type (line 122)
        set_macro_5212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), self_5211, 'set_macro')
        # Calling set_macro(args, kwargs) (line 122)
        set_macro_call_result_5219 = invoke(stypy.reporting.localization.Localization(__file__, 122, 8), set_macro_5212, *[str_5213, result_add_5216, str_5217], **kwargs_5218)
        
        
        # Call to set_macro(...): (line 123)
        # Processing the call arguments (line 123)
        str_5222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 23), 'str', 'VSInstallDir')
        # Getting the type of 'vsbase' (line 123)
        vsbase_5223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 39), 'vsbase', False)
        str_5224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 48), 'str', '\\Setup\\VS')
        # Applying the binary operator '+' (line 123)
        result_add_5225 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 39), '+', vsbase_5223, str_5224)
        
        str_5226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 62), 'str', 'productdir')
        # Processing the call keyword arguments (line 123)
        kwargs_5227 = {}
        # Getting the type of 'self' (line 123)
        self_5220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'self', False)
        # Obtaining the member 'set_macro' of a type (line 123)
        set_macro_5221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), self_5220, 'set_macro')
        # Calling set_macro(args, kwargs) (line 123)
        set_macro_call_result_5228 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), set_macro_5221, *[str_5222, result_add_5225, str_5226], **kwargs_5227)
        
        
        # Assigning a Str to a Name (line 124):
        
        # Assigning a Str to a Name (line 124):
        str_5229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 14), 'str', 'Software\\Microsoft\\.NETFramework')
        # Assigning a type to the variable 'net' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'net', str_5229)
        
        # Call to set_macro(...): (line 125)
        # Processing the call arguments (line 125)
        str_5232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 23), 'str', 'FrameworkDir')
        # Getting the type of 'net' (line 125)
        net_5233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 39), 'net', False)
        str_5234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 44), 'str', 'installroot')
        # Processing the call keyword arguments (line 125)
        kwargs_5235 = {}
        # Getting the type of 'self' (line 125)
        self_5230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'self', False)
        # Obtaining the member 'set_macro' of a type (line 125)
        set_macro_5231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), self_5230, 'set_macro')
        # Calling set_macro(args, kwargs) (line 125)
        set_macro_call_result_5236 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), set_macro_5231, *[str_5232, net_5233, str_5234], **kwargs_5235)
        
        
        
        # SSA begins for try-except statement (line 126)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        
        # Getting the type of 'version' (line 127)
        version_5237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'version')
        float_5238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 25), 'float')
        # Applying the binary operator '>' (line 127)
        result_gt_5239 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 15), '>', version_5237, float_5238)
        
        # Testing the type of an if condition (line 127)
        if_condition_5240 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 127, 12), result_gt_5239)
        # Assigning a type to the variable 'if_condition_5240' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'if_condition_5240', if_condition_5240)
        # SSA begins for if statement (line 127)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_macro(...): (line 128)
        # Processing the call arguments (line 128)
        str_5243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 31), 'str', 'FrameworkSDKDir')
        # Getting the type of 'net' (line 128)
        net_5244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 50), 'net', False)
        str_5245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 55), 'str', 'sdkinstallrootv1.1')
        # Processing the call keyword arguments (line 128)
        kwargs_5246 = {}
        # Getting the type of 'self' (line 128)
        self_5241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'self', False)
        # Obtaining the member 'set_macro' of a type (line 128)
        set_macro_5242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 16), self_5241, 'set_macro')
        # Calling set_macro(args, kwargs) (line 128)
        set_macro_call_result_5247 = invoke(stypy.reporting.localization.Localization(__file__, 128, 16), set_macro_5242, *[str_5243, net_5244, str_5245], **kwargs_5246)
        
        # SSA branch for the else part of an if statement (line 127)
        module_type_store.open_ssa_branch('else')
        
        # Call to set_macro(...): (line 130)
        # Processing the call arguments (line 130)
        str_5250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 31), 'str', 'FrameworkSDKDir')
        # Getting the type of 'net' (line 130)
        net_5251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 50), 'net', False)
        str_5252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 55), 'str', 'sdkinstallroot')
        # Processing the call keyword arguments (line 130)
        kwargs_5253 = {}
        # Getting the type of 'self' (line 130)
        self_5248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'self', False)
        # Obtaining the member 'set_macro' of a type (line 130)
        set_macro_5249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 16), self_5248, 'set_macro')
        # Calling set_macro(args, kwargs) (line 130)
        set_macro_call_result_5254 = invoke(stypy.reporting.localization.Localization(__file__, 130, 16), set_macro_5249, *[str_5250, net_5251, str_5252], **kwargs_5253)
        
        # SSA join for if statement (line 127)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the except part of a try statement (line 126)
        # SSA branch for the except 'KeyError' branch of a try statement (line 126)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsPlatformError' (line 132)
        DistutilsPlatformError_5255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 18), 'DistutilsPlatformError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 132, 12), DistutilsPlatformError_5255, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 126)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Str to a Name (line 138):
        
        # Assigning a Str to a Name (line 138):
        str_5256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 12), 'str', 'Software\\Microsoft\\NET Framework Setup\\Product')
        # Assigning a type to the variable 'p' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'p', str_5256)
        
        # Getting the type of 'HKEYS' (line 139)
        HKEYS_5257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 20), 'HKEYS')
        # Testing the type of a for loop iterable (line 139)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 139, 8), HKEYS_5257)
        # Getting the type of the for loop variable (line 139)
        for_loop_var_5258 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 139, 8), HKEYS_5257)
        # Assigning a type to the variable 'base' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'base', for_loop_var_5258)
        # SSA begins for a for statement (line 139)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # SSA begins for try-except statement (line 140)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 141):
        
        # Assigning a Call to a Name (line 141):
        
        # Call to RegOpenKeyEx(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'base' (line 141)
        base_5260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 33), 'base', False)
        # Getting the type of 'p' (line 141)
        p_5261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 39), 'p', False)
        # Processing the call keyword arguments (line 141)
        kwargs_5262 = {}
        # Getting the type of 'RegOpenKeyEx' (line 141)
        RegOpenKeyEx_5259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 20), 'RegOpenKeyEx', False)
        # Calling RegOpenKeyEx(args, kwargs) (line 141)
        RegOpenKeyEx_call_result_5263 = invoke(stypy.reporting.localization.Localization(__file__, 141, 20), RegOpenKeyEx_5259, *[base_5260, p_5261], **kwargs_5262)
        
        # Assigning a type to the variable 'h' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 'h', RegOpenKeyEx_call_result_5263)
        # SSA branch for the except part of a try statement (line 140)
        # SSA branch for the except 'RegError' branch of a try statement (line 140)
        module_type_store.open_ssa_branch('except')
        # SSA join for try-except statement (line 140)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 144):
        
        # Assigning a Call to a Name (line 144):
        
        # Call to RegEnumKey(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'h' (line 144)
        h_5265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 29), 'h', False)
        int_5266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 32), 'int')
        # Processing the call keyword arguments (line 144)
        kwargs_5267 = {}
        # Getting the type of 'RegEnumKey' (line 144)
        RegEnumKey_5264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 18), 'RegEnumKey', False)
        # Calling RegEnumKey(args, kwargs) (line 144)
        RegEnumKey_call_result_5268 = invoke(stypy.reporting.localization.Localization(__file__, 144, 18), RegEnumKey_5264, *[h_5265, int_5266], **kwargs_5267)
        
        # Assigning a type to the variable 'key' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'key', RegEnumKey_call_result_5268)
        
        # Assigning a Call to a Name (line 145):
        
        # Assigning a Call to a Name (line 145):
        
        # Call to read_values(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'base' (line 145)
        base_5270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 28), 'base', False)
        str_5271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 34), 'str', '%s\\%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 145)
        tuple_5272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 145)
        # Adding element type (line 145)
        # Getting the type of 'p' (line 145)
        p_5273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 46), 'p', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 46), tuple_5272, p_5273)
        # Adding element type (line 145)
        # Getting the type of 'key' (line 145)
        key_5274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 49), 'key', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 46), tuple_5272, key_5274)
        
        # Applying the binary operator '%' (line 145)
        result_mod_5275 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 34), '%', str_5271, tuple_5272)
        
        # Processing the call keyword arguments (line 145)
        kwargs_5276 = {}
        # Getting the type of 'read_values' (line 145)
        read_values_5269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 16), 'read_values', False)
        # Calling read_values(args, kwargs) (line 145)
        read_values_call_result_5277 = invoke(stypy.reporting.localization.Localization(__file__, 145, 16), read_values_5269, *[base_5270, result_mod_5275], **kwargs_5276)
        
        # Assigning a type to the variable 'd' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'd', read_values_call_result_5277)
        
        # Assigning a Subscript to a Subscript (line 146):
        
        # Assigning a Subscript to a Subscript (line 146):
        
        # Obtaining the type of the subscript
        str_5278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 51), 'str', 'version')
        # Getting the type of 'd' (line 146)
        d_5279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 49), 'd')
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___5280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 49), d_5279, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_5281 = invoke(stypy.reporting.localization.Localization(__file__, 146, 49), getitem___5280, str_5278)
        
        # Getting the type of 'self' (line 146)
        self_5282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'self')
        # Obtaining the member 'macros' of a type (line 146)
        macros_5283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 12), self_5282, 'macros')
        str_5284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 24), 'str', '$(FrameworkVersion)')
        # Storing an element on a container (line 146)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 12), macros_5283, (str_5284, subscript_call_result_5281))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'load_macros(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'load_macros' in the type store
        # Getting the type of 'stypy_return_type' (line 120)
        stypy_return_type_5285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5285)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'load_macros'
        return stypy_return_type_5285


    @norecursion
    def sub(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'sub'
        module_type_store = module_type_store.open_function_context('sub', 148, 4, False)
        # Assigning a type to the variable 'self' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MacroExpander.sub.__dict__.__setitem__('stypy_localization', localization)
        MacroExpander.sub.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MacroExpander.sub.__dict__.__setitem__('stypy_type_store', module_type_store)
        MacroExpander.sub.__dict__.__setitem__('stypy_function_name', 'MacroExpander.sub')
        MacroExpander.sub.__dict__.__setitem__('stypy_param_names_list', ['s'])
        MacroExpander.sub.__dict__.__setitem__('stypy_varargs_param_name', None)
        MacroExpander.sub.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MacroExpander.sub.__dict__.__setitem__('stypy_call_defaults', defaults)
        MacroExpander.sub.__dict__.__setitem__('stypy_call_varargs', varargs)
        MacroExpander.sub.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MacroExpander.sub.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MacroExpander.sub', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'sub', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'sub(...)' code ##################

        
        
        # Call to items(...): (line 149)
        # Processing the call keyword arguments (line 149)
        kwargs_5289 = {}
        # Getting the type of 'self' (line 149)
        self_5286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 20), 'self', False)
        # Obtaining the member 'macros' of a type (line 149)
        macros_5287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 20), self_5286, 'macros')
        # Obtaining the member 'items' of a type (line 149)
        items_5288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 20), macros_5287, 'items')
        # Calling items(args, kwargs) (line 149)
        items_call_result_5290 = invoke(stypy.reporting.localization.Localization(__file__, 149, 20), items_5288, *[], **kwargs_5289)
        
        # Testing the type of a for loop iterable (line 149)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 149, 8), items_call_result_5290)
        # Getting the type of the for loop variable (line 149)
        for_loop_var_5291 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 149, 8), items_call_result_5290)
        # Assigning a type to the variable 'k' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 8), for_loop_var_5291))
        # Assigning a type to the variable 'v' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 8), for_loop_var_5291))
        # SSA begins for a for statement (line 149)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 150):
        
        # Assigning a Call to a Name (line 150):
        
        # Call to replace(...): (line 150)
        # Processing the call arguments (line 150)
        # Getting the type of 's' (line 150)
        s_5294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 31), 's', False)
        # Getting the type of 'k' (line 150)
        k_5295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 34), 'k', False)
        # Getting the type of 'v' (line 150)
        v_5296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 37), 'v', False)
        # Processing the call keyword arguments (line 150)
        kwargs_5297 = {}
        # Getting the type of 'string' (line 150)
        string_5292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'string', False)
        # Obtaining the member 'replace' of a type (line 150)
        replace_5293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 16), string_5292, 'replace')
        # Calling replace(args, kwargs) (line 150)
        replace_call_result_5298 = invoke(stypy.reporting.localization.Localization(__file__, 150, 16), replace_5293, *[s_5294, k_5295, v_5296], **kwargs_5297)
        
        # Assigning a type to the variable 's' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 's', replace_call_result_5298)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 's' (line 151)
        s_5299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 15), 's')
        # Assigning a type to the variable 'stypy_return_type' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'stypy_return_type', s_5299)
        
        # ################# End of 'sub(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sub' in the type store
        # Getting the type of 'stypy_return_type' (line 148)
        stypy_return_type_5300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5300)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sub'
        return stypy_return_type_5300


# Assigning a type to the variable 'MacroExpander' (line 107)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'MacroExpander', MacroExpander)

@norecursion
def get_build_version(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_build_version'
    module_type_store = module_type_store.open_function_context('get_build_version', 153, 0, False)
    
    # Passed parameters checking function
    get_build_version.stypy_localization = localization
    get_build_version.stypy_type_of_self = None
    get_build_version.stypy_type_store = module_type_store
    get_build_version.stypy_function_name = 'get_build_version'
    get_build_version.stypy_param_names_list = []
    get_build_version.stypy_varargs_param_name = None
    get_build_version.stypy_kwargs_param_name = None
    get_build_version.stypy_call_defaults = defaults
    get_build_version.stypy_call_varargs = varargs
    get_build_version.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_build_version', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_build_version', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_build_version(...)' code ##################

    str_5301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, (-1)), 'str', 'Return the version of MSVC that was used to build Python.\n\n    For Python 2.3 and up, the version number is included in\n    sys.version.  For earlier versions, assume the compiler is MSVC 6.\n    ')
    
    # Assigning a Str to a Name (line 160):
    
    # Assigning a Str to a Name (line 160):
    str_5302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 13), 'str', 'MSC v.')
    # Assigning a type to the variable 'prefix' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'prefix', str_5302)
    
    # Assigning a Call to a Name (line 161):
    
    # Assigning a Call to a Name (line 161):
    
    # Call to find(...): (line 161)
    # Processing the call arguments (line 161)
    # Getting the type of 'sys' (line 161)
    sys_5305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 20), 'sys', False)
    # Obtaining the member 'version' of a type (line 161)
    version_5306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 20), sys_5305, 'version')
    # Getting the type of 'prefix' (line 161)
    prefix_5307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 33), 'prefix', False)
    # Processing the call keyword arguments (line 161)
    kwargs_5308 = {}
    # Getting the type of 'string' (line 161)
    string_5303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'string', False)
    # Obtaining the member 'find' of a type (line 161)
    find_5304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 8), string_5303, 'find')
    # Calling find(args, kwargs) (line 161)
    find_call_result_5309 = invoke(stypy.reporting.localization.Localization(__file__, 161, 8), find_5304, *[version_5306, prefix_5307], **kwargs_5308)
    
    # Assigning a type to the variable 'i' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'i', find_call_result_5309)
    
    
    # Getting the type of 'i' (line 162)
    i_5310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 7), 'i')
    int_5311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 12), 'int')
    # Applying the binary operator '==' (line 162)
    result_eq_5312 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 7), '==', i_5310, int_5311)
    
    # Testing the type of an if condition (line 162)
    if_condition_5313 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 4), result_eq_5312)
    # Assigning a type to the variable 'if_condition_5313' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'if_condition_5313', if_condition_5313)
    # SSA begins for if statement (line 162)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_5314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'stypy_return_type', int_5314)
    # SSA join for if statement (line 162)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 164):
    
    # Assigning a BinOp to a Name (line 164):
    # Getting the type of 'i' (line 164)
    i_5315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'i')
    
    # Call to len(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'prefix' (line 164)
    prefix_5317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'prefix', False)
    # Processing the call keyword arguments (line 164)
    kwargs_5318 = {}
    # Getting the type of 'len' (line 164)
    len_5316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'len', False)
    # Calling len(args, kwargs) (line 164)
    len_call_result_5319 = invoke(stypy.reporting.localization.Localization(__file__, 164, 12), len_5316, *[prefix_5317], **kwargs_5318)
    
    # Applying the binary operator '+' (line 164)
    result_add_5320 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 8), '+', i_5315, len_call_result_5319)
    
    # Assigning a type to the variable 'i' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'i', result_add_5320)
    
    # Assigning a Call to a Tuple (line 165):
    
    # Assigning a Subscript to a Name (line 165):
    
    # Obtaining the type of the subscript
    int_5321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 4), 'int')
    
    # Call to split(...): (line 165)
    # Processing the call arguments (line 165)
    str_5329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 36), 'str', ' ')
    int_5330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 41), 'int')
    # Processing the call keyword arguments (line 165)
    kwargs_5331 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 165)
    i_5322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 26), 'i', False)
    slice_5323 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 165, 14), i_5322, None, None)
    # Getting the type of 'sys' (line 165)
    sys_5324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 14), 'sys', False)
    # Obtaining the member 'version' of a type (line 165)
    version_5325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 14), sys_5324, 'version')
    # Obtaining the member '__getitem__' of a type (line 165)
    getitem___5326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 14), version_5325, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 165)
    subscript_call_result_5327 = invoke(stypy.reporting.localization.Localization(__file__, 165, 14), getitem___5326, slice_5323)
    
    # Obtaining the member 'split' of a type (line 165)
    split_5328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 14), subscript_call_result_5327, 'split')
    # Calling split(args, kwargs) (line 165)
    split_call_result_5332 = invoke(stypy.reporting.localization.Localization(__file__, 165, 14), split_5328, *[str_5329, int_5330], **kwargs_5331)
    
    # Obtaining the member '__getitem__' of a type (line 165)
    getitem___5333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 4), split_call_result_5332, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 165)
    subscript_call_result_5334 = invoke(stypy.reporting.localization.Localization(__file__, 165, 4), getitem___5333, int_5321)
    
    # Assigning a type to the variable 'tuple_var_assignment_5017' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'tuple_var_assignment_5017', subscript_call_result_5334)
    
    # Assigning a Subscript to a Name (line 165):
    
    # Obtaining the type of the subscript
    int_5335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 4), 'int')
    
    # Call to split(...): (line 165)
    # Processing the call arguments (line 165)
    str_5343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 36), 'str', ' ')
    int_5344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 41), 'int')
    # Processing the call keyword arguments (line 165)
    kwargs_5345 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 165)
    i_5336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 26), 'i', False)
    slice_5337 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 165, 14), i_5336, None, None)
    # Getting the type of 'sys' (line 165)
    sys_5338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 14), 'sys', False)
    # Obtaining the member 'version' of a type (line 165)
    version_5339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 14), sys_5338, 'version')
    # Obtaining the member '__getitem__' of a type (line 165)
    getitem___5340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 14), version_5339, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 165)
    subscript_call_result_5341 = invoke(stypy.reporting.localization.Localization(__file__, 165, 14), getitem___5340, slice_5337)
    
    # Obtaining the member 'split' of a type (line 165)
    split_5342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 14), subscript_call_result_5341, 'split')
    # Calling split(args, kwargs) (line 165)
    split_call_result_5346 = invoke(stypy.reporting.localization.Localization(__file__, 165, 14), split_5342, *[str_5343, int_5344], **kwargs_5345)
    
    # Obtaining the member '__getitem__' of a type (line 165)
    getitem___5347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 4), split_call_result_5346, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 165)
    subscript_call_result_5348 = invoke(stypy.reporting.localization.Localization(__file__, 165, 4), getitem___5347, int_5335)
    
    # Assigning a type to the variable 'tuple_var_assignment_5018' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'tuple_var_assignment_5018', subscript_call_result_5348)
    
    # Assigning a Name to a Name (line 165):
    # Getting the type of 'tuple_var_assignment_5017' (line 165)
    tuple_var_assignment_5017_5349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'tuple_var_assignment_5017')
    # Assigning a type to the variable 's' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 's', tuple_var_assignment_5017_5349)
    
    # Assigning a Name to a Name (line 165):
    # Getting the type of 'tuple_var_assignment_5018' (line 165)
    tuple_var_assignment_5018_5350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'tuple_var_assignment_5018')
    # Assigning a type to the variable 'rest' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 7), 'rest', tuple_var_assignment_5018_5350)
    
    # Assigning a BinOp to a Name (line 166):
    
    # Assigning a BinOp to a Name (line 166):
    
    # Call to int(...): (line 166)
    # Processing the call arguments (line 166)
    
    # Obtaining the type of the subscript
    int_5352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 26), 'int')
    slice_5353 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 166, 23), None, int_5352, None)
    # Getting the type of 's' (line 166)
    s_5354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 23), 's', False)
    # Obtaining the member '__getitem__' of a type (line 166)
    getitem___5355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 23), s_5354, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 166)
    subscript_call_result_5356 = invoke(stypy.reporting.localization.Localization(__file__, 166, 23), getitem___5355, slice_5353)
    
    # Processing the call keyword arguments (line 166)
    kwargs_5357 = {}
    # Getting the type of 'int' (line 166)
    int_5351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 19), 'int', False)
    # Calling int(args, kwargs) (line 166)
    int_call_result_5358 = invoke(stypy.reporting.localization.Localization(__file__, 166, 19), int_5351, *[subscript_call_result_5356], **kwargs_5357)
    
    int_5359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 33), 'int')
    # Applying the binary operator '-' (line 166)
    result_sub_5360 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 19), '-', int_call_result_5358, int_5359)
    
    # Assigning a type to the variable 'majorVersion' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'majorVersion', result_sub_5360)
    
    # Assigning a BinOp to a Name (line 167):
    
    # Assigning a BinOp to a Name (line 167):
    
    # Call to int(...): (line 167)
    # Processing the call arguments (line 167)
    
    # Obtaining the type of the subscript
    int_5362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 25), 'int')
    int_5363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 27), 'int')
    slice_5364 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 167, 23), int_5362, int_5363, None)
    # Getting the type of 's' (line 167)
    s_5365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 23), 's', False)
    # Obtaining the member '__getitem__' of a type (line 167)
    getitem___5366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 23), s_5365, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 167)
    subscript_call_result_5367 = invoke(stypy.reporting.localization.Localization(__file__, 167, 23), getitem___5366, slice_5364)
    
    # Processing the call keyword arguments (line 167)
    kwargs_5368 = {}
    # Getting the type of 'int' (line 167)
    int_5361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 19), 'int', False)
    # Calling int(args, kwargs) (line 167)
    int_call_result_5369 = invoke(stypy.reporting.localization.Localization(__file__, 167, 19), int_5361, *[subscript_call_result_5367], **kwargs_5368)
    
    float_5370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 33), 'float')
    # Applying the binary operator 'div' (line 167)
    result_div_5371 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 19), 'div', int_call_result_5369, float_5370)
    
    # Assigning a type to the variable 'minorVersion' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'minorVersion', result_div_5371)
    
    
    # Getting the type of 'majorVersion' (line 169)
    majorVersion_5372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 7), 'majorVersion')
    int_5373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 23), 'int')
    # Applying the binary operator '==' (line 169)
    result_eq_5374 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 7), '==', majorVersion_5372, int_5373)
    
    # Testing the type of an if condition (line 169)
    if_condition_5375 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 4), result_eq_5374)
    # Assigning a type to the variable 'if_condition_5375' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'if_condition_5375', if_condition_5375)
    # SSA begins for if statement (line 169)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 170):
    
    # Assigning a Num to a Name (line 170):
    int_5376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 23), 'int')
    # Assigning a type to the variable 'minorVersion' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'minorVersion', int_5376)
    # SSA join for if statement (line 169)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'majorVersion' (line 171)
    majorVersion_5377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 7), 'majorVersion')
    int_5378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 23), 'int')
    # Applying the binary operator '>=' (line 171)
    result_ge_5379 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 7), '>=', majorVersion_5377, int_5378)
    
    # Testing the type of an if condition (line 171)
    if_condition_5380 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 4), result_ge_5379)
    # Assigning a type to the variable 'if_condition_5380' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'if_condition_5380', if_condition_5380)
    # SSA begins for if statement (line 171)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'majorVersion' (line 172)
    majorVersion_5381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 15), 'majorVersion')
    # Getting the type of 'minorVersion' (line 172)
    minorVersion_5382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 30), 'minorVersion')
    # Applying the binary operator '+' (line 172)
    result_add_5383 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 15), '+', majorVersion_5381, minorVersion_5382)
    
    # Assigning a type to the variable 'stypy_return_type' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'stypy_return_type', result_add_5383)
    # SSA join for if statement (line 171)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'None' (line 174)
    None_5384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 11), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type', None_5384)
    
    # ################# End of 'get_build_version(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_build_version' in the type store
    # Getting the type of 'stypy_return_type' (line 153)
    stypy_return_type_5385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5385)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_build_version'
    return stypy_return_type_5385

# Assigning a type to the variable 'get_build_version' (line 153)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 0), 'get_build_version', get_build_version)

@norecursion
def get_build_architecture(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_build_architecture'
    module_type_store = module_type_store.open_function_context('get_build_architecture', 176, 0, False)
    
    # Passed parameters checking function
    get_build_architecture.stypy_localization = localization
    get_build_architecture.stypy_type_of_self = None
    get_build_architecture.stypy_type_store = module_type_store
    get_build_architecture.stypy_function_name = 'get_build_architecture'
    get_build_architecture.stypy_param_names_list = []
    get_build_architecture.stypy_varargs_param_name = None
    get_build_architecture.stypy_kwargs_param_name = None
    get_build_architecture.stypy_call_defaults = defaults
    get_build_architecture.stypy_call_varargs = varargs
    get_build_architecture.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_build_architecture', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_build_architecture', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_build_architecture(...)' code ##################

    str_5386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, (-1)), 'str', 'Return the processor architecture.\n\n    Possible results are "Intel", "Itanium", or "AMD64".\n    ')
    
    # Assigning a Str to a Name (line 182):
    
    # Assigning a Str to a Name (line 182):
    str_5387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 13), 'str', ' bit (')
    # Assigning a type to the variable 'prefix' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'prefix', str_5387)
    
    # Assigning a Call to a Name (line 183):
    
    # Assigning a Call to a Name (line 183):
    
    # Call to find(...): (line 183)
    # Processing the call arguments (line 183)
    # Getting the type of 'sys' (line 183)
    sys_5390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 20), 'sys', False)
    # Obtaining the member 'version' of a type (line 183)
    version_5391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 20), sys_5390, 'version')
    # Getting the type of 'prefix' (line 183)
    prefix_5392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 33), 'prefix', False)
    # Processing the call keyword arguments (line 183)
    kwargs_5393 = {}
    # Getting the type of 'string' (line 183)
    string_5388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'string', False)
    # Obtaining the member 'find' of a type (line 183)
    find_5389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 8), string_5388, 'find')
    # Calling find(args, kwargs) (line 183)
    find_call_result_5394 = invoke(stypy.reporting.localization.Localization(__file__, 183, 8), find_5389, *[version_5391, prefix_5392], **kwargs_5393)
    
    # Assigning a type to the variable 'i' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'i', find_call_result_5394)
    
    
    # Getting the type of 'i' (line 184)
    i_5395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 7), 'i')
    int_5396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 12), 'int')
    # Applying the binary operator '==' (line 184)
    result_eq_5397 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 7), '==', i_5395, int_5396)
    
    # Testing the type of an if condition (line 184)
    if_condition_5398 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 184, 4), result_eq_5397)
    # Assigning a type to the variable 'if_condition_5398' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'if_condition_5398', if_condition_5398)
    # SSA begins for if statement (line 184)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_5399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 15), 'str', 'Intel')
    # Assigning a type to the variable 'stypy_return_type' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'stypy_return_type', str_5399)
    # SSA join for if statement (line 184)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 186):
    
    # Assigning a Call to a Name (line 186):
    
    # Call to find(...): (line 186)
    # Processing the call arguments (line 186)
    # Getting the type of 'sys' (line 186)
    sys_5402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 20), 'sys', False)
    # Obtaining the member 'version' of a type (line 186)
    version_5403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 20), sys_5402, 'version')
    str_5404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 33), 'str', ')')
    # Getting the type of 'i' (line 186)
    i_5405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 38), 'i', False)
    # Processing the call keyword arguments (line 186)
    kwargs_5406 = {}
    # Getting the type of 'string' (line 186)
    string_5400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'string', False)
    # Obtaining the member 'find' of a type (line 186)
    find_5401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), string_5400, 'find')
    # Calling find(args, kwargs) (line 186)
    find_call_result_5407 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), find_5401, *[version_5403, str_5404, i_5405], **kwargs_5406)
    
    # Assigning a type to the variable 'j' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'j', find_call_result_5407)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 187)
    i_5408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 23), 'i')
    
    # Call to len(...): (line 187)
    # Processing the call arguments (line 187)
    # Getting the type of 'prefix' (line 187)
    prefix_5410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 29), 'prefix', False)
    # Processing the call keyword arguments (line 187)
    kwargs_5411 = {}
    # Getting the type of 'len' (line 187)
    len_5409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 25), 'len', False)
    # Calling len(args, kwargs) (line 187)
    len_call_result_5412 = invoke(stypy.reporting.localization.Localization(__file__, 187, 25), len_5409, *[prefix_5410], **kwargs_5411)
    
    # Applying the binary operator '+' (line 187)
    result_add_5413 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 23), '+', i_5408, len_call_result_5412)
    
    # Getting the type of 'j' (line 187)
    j_5414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 37), 'j')
    slice_5415 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 187, 11), result_add_5413, j_5414, None)
    # Getting the type of 'sys' (line 187)
    sys_5416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 11), 'sys')
    # Obtaining the member 'version' of a type (line 187)
    version_5417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 11), sys_5416, 'version')
    # Obtaining the member '__getitem__' of a type (line 187)
    getitem___5418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 11), version_5417, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 187)
    subscript_call_result_5419 = invoke(stypy.reporting.localization.Localization(__file__, 187, 11), getitem___5418, slice_5415)
    
    # Assigning a type to the variable 'stypy_return_type' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'stypy_return_type', subscript_call_result_5419)
    
    # ################# End of 'get_build_architecture(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_build_architecture' in the type store
    # Getting the type of 'stypy_return_type' (line 176)
    stypy_return_type_5420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5420)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_build_architecture'
    return stypy_return_type_5420

# Assigning a type to the variable 'get_build_architecture' (line 176)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'get_build_architecture', get_build_architecture)

@norecursion
def normalize_and_reduce_paths(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'normalize_and_reduce_paths'
    module_type_store = module_type_store.open_function_context('normalize_and_reduce_paths', 189, 0, False)
    
    # Passed parameters checking function
    normalize_and_reduce_paths.stypy_localization = localization
    normalize_and_reduce_paths.stypy_type_of_self = None
    normalize_and_reduce_paths.stypy_type_store = module_type_store
    normalize_and_reduce_paths.stypy_function_name = 'normalize_and_reduce_paths'
    normalize_and_reduce_paths.stypy_param_names_list = ['paths']
    normalize_and_reduce_paths.stypy_varargs_param_name = None
    normalize_and_reduce_paths.stypy_kwargs_param_name = None
    normalize_and_reduce_paths.stypy_call_defaults = defaults
    normalize_and_reduce_paths.stypy_call_varargs = varargs
    normalize_and_reduce_paths.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'normalize_and_reduce_paths', ['paths'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'normalize_and_reduce_paths', localization, ['paths'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'normalize_and_reduce_paths(...)' code ##################

    str_5421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, (-1)), 'str', 'Return a list of normalized paths with duplicates removed.\n\n    The current order of paths is maintained.\n    ')
    
    # Assigning a List to a Name (line 195):
    
    # Assigning a List to a Name (line 195):
    
    # Obtaining an instance of the builtin type 'list' (line 195)
    list_5422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 195)
    
    # Assigning a type to the variable 'reduced_paths' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'reduced_paths', list_5422)
    
    # Getting the type of 'paths' (line 196)
    paths_5423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 13), 'paths')
    # Testing the type of a for loop iterable (line 196)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 196, 4), paths_5423)
    # Getting the type of the for loop variable (line 196)
    for_loop_var_5424 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 196, 4), paths_5423)
    # Assigning a type to the variable 'p' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'p', for_loop_var_5424)
    # SSA begins for a for statement (line 196)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 197):
    
    # Assigning a Call to a Name (line 197):
    
    # Call to normpath(...): (line 197)
    # Processing the call arguments (line 197)
    # Getting the type of 'p' (line 197)
    p_5428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 30), 'p', False)
    # Processing the call keyword arguments (line 197)
    kwargs_5429 = {}
    # Getting the type of 'os' (line 197)
    os_5425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 13), 'os', False)
    # Obtaining the member 'path' of a type (line 197)
    path_5426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 13), os_5425, 'path')
    # Obtaining the member 'normpath' of a type (line 197)
    normpath_5427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 13), path_5426, 'normpath')
    # Calling normpath(args, kwargs) (line 197)
    normpath_call_result_5430 = invoke(stypy.reporting.localization.Localization(__file__, 197, 13), normpath_5427, *[p_5428], **kwargs_5429)
    
    # Assigning a type to the variable 'np' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'np', normpath_call_result_5430)
    
    
    # Getting the type of 'np' (line 199)
    np_5431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 11), 'np')
    # Getting the type of 'reduced_paths' (line 199)
    reduced_paths_5432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 21), 'reduced_paths')
    # Applying the binary operator 'notin' (line 199)
    result_contains_5433 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 11), 'notin', np_5431, reduced_paths_5432)
    
    # Testing the type of an if condition (line 199)
    if_condition_5434 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 199, 8), result_contains_5433)
    # Assigning a type to the variable 'if_condition_5434' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'if_condition_5434', if_condition_5434)
    # SSA begins for if statement (line 199)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 200)
    # Processing the call arguments (line 200)
    # Getting the type of 'np' (line 200)
    np_5437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 33), 'np', False)
    # Processing the call keyword arguments (line 200)
    kwargs_5438 = {}
    # Getting the type of 'reduced_paths' (line 200)
    reduced_paths_5435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'reduced_paths', False)
    # Obtaining the member 'append' of a type (line 200)
    append_5436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 12), reduced_paths_5435, 'append')
    # Calling append(args, kwargs) (line 200)
    append_call_result_5439 = invoke(stypy.reporting.localization.Localization(__file__, 200, 12), append_5436, *[np_5437], **kwargs_5438)
    
    # SSA join for if statement (line 199)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'reduced_paths' (line 201)
    reduced_paths_5440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 11), 'reduced_paths')
    # Assigning a type to the variable 'stypy_return_type' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'stypy_return_type', reduced_paths_5440)
    
    # ################# End of 'normalize_and_reduce_paths(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'normalize_and_reduce_paths' in the type store
    # Getting the type of 'stypy_return_type' (line 189)
    stypy_return_type_5441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5441)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'normalize_and_reduce_paths'
    return stypy_return_type_5441

# Assigning a type to the variable 'normalize_and_reduce_paths' (line 189)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 0), 'normalize_and_reduce_paths', normalize_and_reduce_paths)
# Declaration of the 'MSVCCompiler' class
# Getting the type of 'CCompiler' (line 204)
CCompiler_5442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 20), 'CCompiler')

class MSVCCompiler(CCompiler_5442, ):
    str_5443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, (-1)), 'str', 'Concrete class that implements an interface to Microsoft Visual C++,\n       as defined by the CCompiler abstract class.')
    
    # Assigning a Str to a Name (line 208):
    
    # Assigning a Dict to a Name (line 215):
    
    # Assigning a List to a Name (line 218):
    
    # Assigning a List to a Name (line 219):
    
    # Assigning a List to a Name (line 220):
    
    # Assigning a List to a Name (line 221):
    
    # Assigning a BinOp to a Name (line 225):
    
    # Assigning a Str to a Name (line 227):
    
    # Assigning a Str to a Name (line 228):
    
    # Assigning a Str to a Name (line 229):
    
    # Assigning a Str to a Name (line 230):
    
    # Multiple assignment of 2 elements.
    
    # Assigning a Str to a Name (line 232):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_5444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 32), 'int')
        int_5445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 43), 'int')
        int_5446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 52), 'int')
        defaults = [int_5444, int_5445, int_5446]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 234, 4, False)
        # Assigning a type to the variable 'self' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MSVCCompiler.__init__', ['verbose', 'dry_run', 'force'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['verbose', 'dry_run', 'force'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'self' (line 235)
        self_5449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 28), 'self', False)
        # Getting the type of 'verbose' (line 235)
        verbose_5450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 34), 'verbose', False)
        # Getting the type of 'dry_run' (line 235)
        dry_run_5451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 43), 'dry_run', False)
        # Getting the type of 'force' (line 235)
        force_5452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 52), 'force', False)
        # Processing the call keyword arguments (line 235)
        kwargs_5453 = {}
        # Getting the type of 'CCompiler' (line 235)
        CCompiler_5447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'CCompiler', False)
        # Obtaining the member '__init__' of a type (line 235)
        init___5448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), CCompiler_5447, '__init__')
        # Calling __init__(args, kwargs) (line 235)
        init___call_result_5454 = invoke(stypy.reporting.localization.Localization(__file__, 235, 8), init___5448, *[self_5449, verbose_5450, dry_run_5451, force_5452], **kwargs_5453)
        
        
        # Assigning a Call to a Attribute (line 236):
        
        # Assigning a Call to a Attribute (line 236):
        
        # Call to get_build_version(...): (line 236)
        # Processing the call keyword arguments (line 236)
        kwargs_5456 = {}
        # Getting the type of 'get_build_version' (line 236)
        get_build_version_5455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 25), 'get_build_version', False)
        # Calling get_build_version(args, kwargs) (line 236)
        get_build_version_call_result_5457 = invoke(stypy.reporting.localization.Localization(__file__, 236, 25), get_build_version_5455, *[], **kwargs_5456)
        
        # Getting the type of 'self' (line 236)
        self_5458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'self')
        # Setting the type of the member '__version' of a type (line 236)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 8), self_5458, '__version', get_build_version_call_result_5457)
        
        # Assigning a Call to a Attribute (line 237):
        
        # Assigning a Call to a Attribute (line 237):
        
        # Call to get_build_architecture(...): (line 237)
        # Processing the call keyword arguments (line 237)
        kwargs_5460 = {}
        # Getting the type of 'get_build_architecture' (line 237)
        get_build_architecture_5459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 22), 'get_build_architecture', False)
        # Calling get_build_architecture(args, kwargs) (line 237)
        get_build_architecture_call_result_5461 = invoke(stypy.reporting.localization.Localization(__file__, 237, 22), get_build_architecture_5459, *[], **kwargs_5460)
        
        # Getting the type of 'self' (line 237)
        self_5462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'self')
        # Setting the type of the member '__arch' of a type (line 237)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 8), self_5462, '__arch', get_build_architecture_call_result_5461)
        
        
        # Getting the type of 'self' (line 238)
        self_5463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 11), 'self')
        # Obtaining the member '__arch' of a type (line 238)
        arch_5464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 11), self_5463, '__arch')
        str_5465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 26), 'str', 'Intel')
        # Applying the binary operator '==' (line 238)
        result_eq_5466 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 11), '==', arch_5464, str_5465)
        
        # Testing the type of an if condition (line 238)
        if_condition_5467 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 238, 8), result_eq_5466)
        # Assigning a type to the variable 'if_condition_5467' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'if_condition_5467', if_condition_5467)
        # SSA begins for if statement (line 238)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'self' (line 240)
        self_5468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 15), 'self')
        # Obtaining the member '__version' of a type (line 240)
        version_5469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 15), self_5468, '__version')
        int_5470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 33), 'int')
        # Applying the binary operator '>=' (line 240)
        result_ge_5471 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 15), '>=', version_5469, int_5470)
        
        # Testing the type of an if condition (line 240)
        if_condition_5472 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 12), result_ge_5471)
        # Assigning a type to the variable 'if_condition_5472' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'if_condition_5472', if_condition_5472)
        # SSA begins for if statement (line 240)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Attribute (line 241):
        
        # Assigning a Str to a Attribute (line 241):
        str_5473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 30), 'str', 'Software\\Microsoft\\VisualStudio')
        # Getting the type of 'self' (line 241)
        self_5474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 16), 'self')
        # Setting the type of the member '__root' of a type (line 241)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 16), self_5474, '__root', str_5473)
        
        # Assigning a Call to a Attribute (line 242):
        
        # Assigning a Call to a Attribute (line 242):
        
        # Call to MacroExpander(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'self' (line 242)
        self_5476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 46), 'self', False)
        # Obtaining the member '__version' of a type (line 242)
        version_5477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 46), self_5476, '__version')
        # Processing the call keyword arguments (line 242)
        kwargs_5478 = {}
        # Getting the type of 'MacroExpander' (line 242)
        MacroExpander_5475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 32), 'MacroExpander', False)
        # Calling MacroExpander(args, kwargs) (line 242)
        MacroExpander_call_result_5479 = invoke(stypy.reporting.localization.Localization(__file__, 242, 32), MacroExpander_5475, *[version_5477], **kwargs_5478)
        
        # Getting the type of 'self' (line 242)
        self_5480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'self')
        # Setting the type of the member '__macros' of a type (line 242)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 16), self_5480, '__macros', MacroExpander_call_result_5479)
        # SSA branch for the else part of an if statement (line 240)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Attribute (line 244):
        
        # Assigning a Str to a Attribute (line 244):
        str_5481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 30), 'str', 'Software\\Microsoft\\Devstudio')
        # Getting the type of 'self' (line 244)
        self_5482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 16), 'self')
        # Setting the type of the member '__root' of a type (line 244)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 16), self_5482, '__root', str_5481)
        # SSA join for if statement (line 240)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Attribute (line 245):
        
        # Assigning a BinOp to a Attribute (line 245):
        str_5483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 29), 'str', 'Visual Studio version %s')
        # Getting the type of 'self' (line 245)
        self_5484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 58), 'self')
        # Obtaining the member '__version' of a type (line 245)
        version_5485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 58), self_5484, '__version')
        # Applying the binary operator '%' (line 245)
        result_mod_5486 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 29), '%', str_5483, version_5485)
        
        # Getting the type of 'self' (line 245)
        self_5487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'self')
        # Setting the type of the member '__product' of a type (line 245)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 12), self_5487, '__product', result_mod_5486)
        # SSA branch for the else part of an if statement (line 238)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Attribute (line 248):
        
        # Assigning a BinOp to a Attribute (line 248):
        str_5488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 29), 'str', 'Microsoft SDK compiler %s')
        # Getting the type of 'self' (line 248)
        self_5489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 60), 'self')
        # Obtaining the member '__version' of a type (line 248)
        version_5490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 60), self_5489, '__version')
        int_5491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 77), 'int')
        # Applying the binary operator '+' (line 248)
        result_add_5492 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 60), '+', version_5490, int_5491)
        
        # Applying the binary operator '%' (line 248)
        result_mod_5493 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 29), '%', str_5488, result_add_5492)
        
        # Getting the type of 'self' (line 248)
        self_5494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'self')
        # Setting the type of the member '__product' of a type (line 248)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 12), self_5494, '__product', result_mod_5493)
        # SSA join for if statement (line 238)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 250):
        
        # Assigning a Name to a Attribute (line 250):
        # Getting the type of 'False' (line 250)
        False_5495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 27), 'False')
        # Getting the type of 'self' (line 250)
        self_5496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'self')
        # Setting the type of the member 'initialized' of a type (line 250)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), self_5496, 'initialized', False_5495)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def initialize(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize'
        module_type_store = module_type_store.open_function_context('initialize', 252, 4, False)
        # Assigning a type to the variable 'self' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_localization', localization)
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_type_store', module_type_store)
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_function_name', 'MSVCCompiler.initialize')
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_param_names_list', [])
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_varargs_param_name', None)
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_call_defaults', defaults)
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_call_varargs', varargs)
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MSVCCompiler.initialize', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'initialize', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'initialize(...)' code ##################

        
        # Assigning a List to a Attribute (line 253):
        
        # Assigning a List to a Attribute (line 253):
        
        # Obtaining an instance of the builtin type 'list' (line 253)
        list_5497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 253)
        
        # Getting the type of 'self' (line 253)
        self_5498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'self')
        # Setting the type of the member '__paths' of a type (line 253)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), self_5498, '__paths', list_5497)
        
        
        # Evaluating a boolean operation
        
        str_5499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 11), 'str', 'DISTUTILS_USE_SDK')
        # Getting the type of 'os' (line 254)
        os_5500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 34), 'os')
        # Obtaining the member 'environ' of a type (line 254)
        environ_5501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 34), os_5500, 'environ')
        # Applying the binary operator 'in' (line 254)
        result_contains_5502 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 11), 'in', str_5499, environ_5501)
        
        
        str_5503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 49), 'str', 'MSSdk')
        # Getting the type of 'os' (line 254)
        os_5504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 60), 'os')
        # Obtaining the member 'environ' of a type (line 254)
        environ_5505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 60), os_5504, 'environ')
        # Applying the binary operator 'in' (line 254)
        result_contains_5506 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 49), 'in', str_5503, environ_5505)
        
        # Applying the binary operator 'and' (line 254)
        result_and_keyword_5507 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 11), 'and', result_contains_5502, result_contains_5506)
        
        # Call to find_exe(...): (line 254)
        # Processing the call arguments (line 254)
        str_5510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 89), 'str', 'cl.exe')
        # Processing the call keyword arguments (line 254)
        kwargs_5511 = {}
        # Getting the type of 'self' (line 254)
        self_5508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 75), 'self', False)
        # Obtaining the member 'find_exe' of a type (line 254)
        find_exe_5509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 75), self_5508, 'find_exe')
        # Calling find_exe(args, kwargs) (line 254)
        find_exe_call_result_5512 = invoke(stypy.reporting.localization.Localization(__file__, 254, 75), find_exe_5509, *[str_5510], **kwargs_5511)
        
        # Applying the binary operator 'and' (line 254)
        result_and_keyword_5513 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 11), 'and', result_and_keyword_5507, find_exe_call_result_5512)
        
        # Testing the type of an if condition (line 254)
        if_condition_5514 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 254, 8), result_and_keyword_5513)
        # Assigning a type to the variable 'if_condition_5514' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'if_condition_5514', if_condition_5514)
        # SSA begins for if statement (line 254)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Attribute (line 257):
        
        # Assigning a Str to a Attribute (line 257):
        str_5515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 22), 'str', 'cl.exe')
        # Getting the type of 'self' (line 257)
        self_5516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'self')
        # Setting the type of the member 'cc' of a type (line 257)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 12), self_5516, 'cc', str_5515)
        
        # Assigning a Str to a Attribute (line 258):
        
        # Assigning a Str to a Attribute (line 258):
        str_5517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 26), 'str', 'link.exe')
        # Getting the type of 'self' (line 258)
        self_5518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'self')
        # Setting the type of the member 'linker' of a type (line 258)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 12), self_5518, 'linker', str_5517)
        
        # Assigning a Str to a Attribute (line 259):
        
        # Assigning a Str to a Attribute (line 259):
        str_5519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 23), 'str', 'lib.exe')
        # Getting the type of 'self' (line 259)
        self_5520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'self')
        # Setting the type of the member 'lib' of a type (line 259)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 12), self_5520, 'lib', str_5519)
        
        # Assigning a Str to a Attribute (line 260):
        
        # Assigning a Str to a Attribute (line 260):
        str_5521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 22), 'str', 'rc.exe')
        # Getting the type of 'self' (line 260)
        self_5522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'self')
        # Setting the type of the member 'rc' of a type (line 260)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 12), self_5522, 'rc', str_5521)
        
        # Assigning a Str to a Attribute (line 261):
        
        # Assigning a Str to a Attribute (line 261):
        str_5523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 22), 'str', 'mc.exe')
        # Getting the type of 'self' (line 261)
        self_5524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 12), 'self')
        # Setting the type of the member 'mc' of a type (line 261)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 12), self_5524, 'mc', str_5523)
        # SSA branch for the else part of an if statement (line 254)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Attribute (line 263):
        
        # Assigning a Call to a Attribute (line 263):
        
        # Call to get_msvc_paths(...): (line 263)
        # Processing the call arguments (line 263)
        str_5527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 47), 'str', 'path')
        # Processing the call keyword arguments (line 263)
        kwargs_5528 = {}
        # Getting the type of 'self' (line 263)
        self_5525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 27), 'self', False)
        # Obtaining the member 'get_msvc_paths' of a type (line 263)
        get_msvc_paths_5526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 27), self_5525, 'get_msvc_paths')
        # Calling get_msvc_paths(args, kwargs) (line 263)
        get_msvc_paths_call_result_5529 = invoke(stypy.reporting.localization.Localization(__file__, 263, 27), get_msvc_paths_5526, *[str_5527], **kwargs_5528)
        
        # Getting the type of 'self' (line 263)
        self_5530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'self')
        # Setting the type of the member '__paths' of a type (line 263)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 12), self_5530, '__paths', get_msvc_paths_call_result_5529)
        
        
        
        # Call to len(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'self' (line 265)
        self_5532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 20), 'self', False)
        # Obtaining the member '__paths' of a type (line 265)
        paths_5533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 20), self_5532, '__paths')
        # Processing the call keyword arguments (line 265)
        kwargs_5534 = {}
        # Getting the type of 'len' (line 265)
        len_5531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 15), 'len', False)
        # Calling len(args, kwargs) (line 265)
        len_call_result_5535 = invoke(stypy.reporting.localization.Localization(__file__, 265, 15), len_5531, *[paths_5533], **kwargs_5534)
        
        int_5536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 37), 'int')
        # Applying the binary operator '==' (line 265)
        result_eq_5537 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 15), '==', len_call_result_5535, int_5536)
        
        # Testing the type of an if condition (line 265)
        if_condition_5538 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 265, 12), result_eq_5537)
        # Assigning a type to the variable 'if_condition_5538' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'if_condition_5538', if_condition_5538)
        # SSA begins for if statement (line 265)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsPlatformError' (line 266)
        DistutilsPlatformError_5539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 22), 'DistutilsPlatformError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 266, 16), DistutilsPlatformError_5539, 'raise parameter', BaseException)
        # SSA join for if statement (line 265)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 271):
        
        # Assigning a Call to a Attribute (line 271):
        
        # Call to find_exe(...): (line 271)
        # Processing the call arguments (line 271)
        str_5542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 36), 'str', 'cl.exe')
        # Processing the call keyword arguments (line 271)
        kwargs_5543 = {}
        # Getting the type of 'self' (line 271)
        self_5540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 22), 'self', False)
        # Obtaining the member 'find_exe' of a type (line 271)
        find_exe_5541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 22), self_5540, 'find_exe')
        # Calling find_exe(args, kwargs) (line 271)
        find_exe_call_result_5544 = invoke(stypy.reporting.localization.Localization(__file__, 271, 22), find_exe_5541, *[str_5542], **kwargs_5543)
        
        # Getting the type of 'self' (line 271)
        self_5545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'self')
        # Setting the type of the member 'cc' of a type (line 271)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 12), self_5545, 'cc', find_exe_call_result_5544)
        
        # Assigning a Call to a Attribute (line 272):
        
        # Assigning a Call to a Attribute (line 272):
        
        # Call to find_exe(...): (line 272)
        # Processing the call arguments (line 272)
        str_5548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 40), 'str', 'link.exe')
        # Processing the call keyword arguments (line 272)
        kwargs_5549 = {}
        # Getting the type of 'self' (line 272)
        self_5546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 26), 'self', False)
        # Obtaining the member 'find_exe' of a type (line 272)
        find_exe_5547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 26), self_5546, 'find_exe')
        # Calling find_exe(args, kwargs) (line 272)
        find_exe_call_result_5550 = invoke(stypy.reporting.localization.Localization(__file__, 272, 26), find_exe_5547, *[str_5548], **kwargs_5549)
        
        # Getting the type of 'self' (line 272)
        self_5551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'self')
        # Setting the type of the member 'linker' of a type (line 272)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 12), self_5551, 'linker', find_exe_call_result_5550)
        
        # Assigning a Call to a Attribute (line 273):
        
        # Assigning a Call to a Attribute (line 273):
        
        # Call to find_exe(...): (line 273)
        # Processing the call arguments (line 273)
        str_5554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 37), 'str', 'lib.exe')
        # Processing the call keyword arguments (line 273)
        kwargs_5555 = {}
        # Getting the type of 'self' (line 273)
        self_5552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 23), 'self', False)
        # Obtaining the member 'find_exe' of a type (line 273)
        find_exe_5553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 23), self_5552, 'find_exe')
        # Calling find_exe(args, kwargs) (line 273)
        find_exe_call_result_5556 = invoke(stypy.reporting.localization.Localization(__file__, 273, 23), find_exe_5553, *[str_5554], **kwargs_5555)
        
        # Getting the type of 'self' (line 273)
        self_5557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'self')
        # Setting the type of the member 'lib' of a type (line 273)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 12), self_5557, 'lib', find_exe_call_result_5556)
        
        # Assigning a Call to a Attribute (line 274):
        
        # Assigning a Call to a Attribute (line 274):
        
        # Call to find_exe(...): (line 274)
        # Processing the call arguments (line 274)
        str_5560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 36), 'str', 'rc.exe')
        # Processing the call keyword arguments (line 274)
        kwargs_5561 = {}
        # Getting the type of 'self' (line 274)
        self_5558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 22), 'self', False)
        # Obtaining the member 'find_exe' of a type (line 274)
        find_exe_5559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 22), self_5558, 'find_exe')
        # Calling find_exe(args, kwargs) (line 274)
        find_exe_call_result_5562 = invoke(stypy.reporting.localization.Localization(__file__, 274, 22), find_exe_5559, *[str_5560], **kwargs_5561)
        
        # Getting the type of 'self' (line 274)
        self_5563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'self')
        # Setting the type of the member 'rc' of a type (line 274)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 12), self_5563, 'rc', find_exe_call_result_5562)
        
        # Assigning a Call to a Attribute (line 275):
        
        # Assigning a Call to a Attribute (line 275):
        
        # Call to find_exe(...): (line 275)
        # Processing the call arguments (line 275)
        str_5566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 36), 'str', 'mc.exe')
        # Processing the call keyword arguments (line 275)
        kwargs_5567 = {}
        # Getting the type of 'self' (line 275)
        self_5564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 22), 'self', False)
        # Obtaining the member 'find_exe' of a type (line 275)
        find_exe_5565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 22), self_5564, 'find_exe')
        # Calling find_exe(args, kwargs) (line 275)
        find_exe_call_result_5568 = invoke(stypy.reporting.localization.Localization(__file__, 275, 22), find_exe_5565, *[str_5566], **kwargs_5567)
        
        # Getting the type of 'self' (line 275)
        self_5569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'self')
        # Setting the type of the member 'mc' of a type (line 275)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 12), self_5569, 'mc', find_exe_call_result_5568)
        
        # Call to set_path_env_var(...): (line 276)
        # Processing the call arguments (line 276)
        str_5572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 34), 'str', 'lib')
        # Processing the call keyword arguments (line 276)
        kwargs_5573 = {}
        # Getting the type of 'self' (line 276)
        self_5570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'self', False)
        # Obtaining the member 'set_path_env_var' of a type (line 276)
        set_path_env_var_5571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 12), self_5570, 'set_path_env_var')
        # Calling set_path_env_var(args, kwargs) (line 276)
        set_path_env_var_call_result_5574 = invoke(stypy.reporting.localization.Localization(__file__, 276, 12), set_path_env_var_5571, *[str_5572], **kwargs_5573)
        
        
        # Call to set_path_env_var(...): (line 277)
        # Processing the call arguments (line 277)
        str_5577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 34), 'str', 'include')
        # Processing the call keyword arguments (line 277)
        kwargs_5578 = {}
        # Getting the type of 'self' (line 277)
        self_5575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'self', False)
        # Obtaining the member 'set_path_env_var' of a type (line 277)
        set_path_env_var_5576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 12), self_5575, 'set_path_env_var')
        # Calling set_path_env_var(args, kwargs) (line 277)
        set_path_env_var_call_result_5579 = invoke(stypy.reporting.localization.Localization(__file__, 277, 12), set_path_env_var_5576, *[str_5577], **kwargs_5578)
        
        # SSA join for if statement (line 254)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 280)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        
        # Call to split(...): (line 281)
        # Processing the call arguments (line 281)
        
        # Obtaining the type of the subscript
        str_5582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 45), 'str', 'path')
        # Getting the type of 'os' (line 281)
        os_5583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 34), 'os', False)
        # Obtaining the member 'environ' of a type (line 281)
        environ_5584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 34), os_5583, 'environ')
        # Obtaining the member '__getitem__' of a type (line 281)
        getitem___5585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 34), environ_5584, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 281)
        subscript_call_result_5586 = invoke(stypy.reporting.localization.Localization(__file__, 281, 34), getitem___5585, str_5582)
        
        str_5587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 54), 'str', ';')
        # Processing the call keyword arguments (line 281)
        kwargs_5588 = {}
        # Getting the type of 'string' (line 281)
        string_5580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 21), 'string', False)
        # Obtaining the member 'split' of a type (line 281)
        split_5581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 21), string_5580, 'split')
        # Calling split(args, kwargs) (line 281)
        split_call_result_5589 = invoke(stypy.reporting.localization.Localization(__file__, 281, 21), split_5581, *[subscript_call_result_5586, str_5587], **kwargs_5588)
        
        # Testing the type of a for loop iterable (line 281)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 281, 12), split_call_result_5589)
        # Getting the type of the for loop variable (line 281)
        for_loop_var_5590 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 281, 12), split_call_result_5589)
        # Assigning a type to the variable 'p' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'p', for_loop_var_5590)
        # SSA begins for a for statement (line 281)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'p' (line 282)
        p_5594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 36), 'p', False)
        # Processing the call keyword arguments (line 282)
        kwargs_5595 = {}
        # Getting the type of 'self' (line 282)
        self_5591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 16), 'self', False)
        # Obtaining the member '__paths' of a type (line 282)
        paths_5592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 16), self_5591, '__paths')
        # Obtaining the member 'append' of a type (line 282)
        append_5593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 16), paths_5592, 'append')
        # Calling append(args, kwargs) (line 282)
        append_call_result_5596 = invoke(stypy.reporting.localization.Localization(__file__, 282, 16), append_5593, *[p_5594], **kwargs_5595)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the except part of a try statement (line 280)
        # SSA branch for the except 'KeyError' branch of a try statement (line 280)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 280)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 285):
        
        # Assigning a Call to a Attribute (line 285):
        
        # Call to normalize_and_reduce_paths(...): (line 285)
        # Processing the call arguments (line 285)
        # Getting the type of 'self' (line 285)
        self_5598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 50), 'self', False)
        # Obtaining the member '__paths' of a type (line 285)
        paths_5599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 50), self_5598, '__paths')
        # Processing the call keyword arguments (line 285)
        kwargs_5600 = {}
        # Getting the type of 'normalize_and_reduce_paths' (line 285)
        normalize_and_reduce_paths_5597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 23), 'normalize_and_reduce_paths', False)
        # Calling normalize_and_reduce_paths(args, kwargs) (line 285)
        normalize_and_reduce_paths_call_result_5601 = invoke(stypy.reporting.localization.Localization(__file__, 285, 23), normalize_and_reduce_paths_5597, *[paths_5599], **kwargs_5600)
        
        # Getting the type of 'self' (line 285)
        self_5602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'self')
        # Setting the type of the member '__paths' of a type (line 285)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 8), self_5602, '__paths', normalize_and_reduce_paths_call_result_5601)
        
        # Assigning a Call to a Subscript (line 286):
        
        # Assigning a Call to a Subscript (line 286):
        
        # Call to join(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 'self' (line 286)
        self_5605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 41), 'self', False)
        # Obtaining the member '__paths' of a type (line 286)
        paths_5606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 41), self_5605, '__paths')
        str_5607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 55), 'str', ';')
        # Processing the call keyword arguments (line 286)
        kwargs_5608 = {}
        # Getting the type of 'string' (line 286)
        string_5603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 29), 'string', False)
        # Obtaining the member 'join' of a type (line 286)
        join_5604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 29), string_5603, 'join')
        # Calling join(args, kwargs) (line 286)
        join_call_result_5609 = invoke(stypy.reporting.localization.Localization(__file__, 286, 29), join_5604, *[paths_5606, str_5607], **kwargs_5608)
        
        # Getting the type of 'os' (line 286)
        os_5610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'os')
        # Obtaining the member 'environ' of a type (line 286)
        environ_5611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 8), os_5610, 'environ')
        str_5612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 19), 'str', 'path')
        # Storing an element on a container (line 286)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 8), environ_5611, (str_5612, join_call_result_5609))
        
        # Assigning a Name to a Attribute (line 288):
        
        # Assigning a Name to a Attribute (line 288):
        # Getting the type of 'None' (line 288)
        None_5613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 34), 'None')
        # Getting the type of 'self' (line 288)
        self_5614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'self')
        # Setting the type of the member 'preprocess_options' of a type (line 288)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 8), self_5614, 'preprocess_options', None_5613)
        
        
        # Getting the type of 'self' (line 289)
        self_5615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 11), 'self')
        # Obtaining the member '__arch' of a type (line 289)
        arch_5616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 11), self_5615, '__arch')
        str_5617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 26), 'str', 'Intel')
        # Applying the binary operator '==' (line 289)
        result_eq_5618 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 11), '==', arch_5616, str_5617)
        
        # Testing the type of an if condition (line 289)
        if_condition_5619 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 289, 8), result_eq_5618)
        # Assigning a type to the variable 'if_condition_5619' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'if_condition_5619', if_condition_5619)
        # SSA begins for if statement (line 289)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Attribute (line 290):
        
        # Assigning a List to a Attribute (line 290):
        
        # Obtaining an instance of the builtin type 'list' (line 290)
        list_5620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 290)
        # Adding element type (line 290)
        str_5621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 37), 'str', '/nologo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 35), list_5620, str_5621)
        # Adding element type (line 290)
        str_5622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 48), 'str', '/Ox')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 35), list_5620, str_5622)
        # Adding element type (line 290)
        str_5623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 55), 'str', '/MD')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 35), list_5620, str_5623)
        # Adding element type (line 290)
        str_5624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 62), 'str', '/W3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 35), list_5620, str_5624)
        # Adding element type (line 290)
        str_5625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 69), 'str', '/GX')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 35), list_5620, str_5625)
        # Adding element type (line 290)
        str_5626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 37), 'str', '/DNDEBUG')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 35), list_5620, str_5626)
        
        # Getting the type of 'self' (line 290)
        self_5627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'self')
        # Setting the type of the member 'compile_options' of a type (line 290)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 12), self_5627, 'compile_options', list_5620)
        
        # Assigning a List to a Attribute (line 292):
        
        # Assigning a List to a Attribute (line 292):
        
        # Obtaining an instance of the builtin type 'list' (line 292)
        list_5628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 292)
        # Adding element type (line 292)
        str_5629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 42), 'str', '/nologo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 41), list_5628, str_5629)
        # Adding element type (line 292)
        str_5630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 53), 'str', '/Od')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 41), list_5628, str_5630)
        # Adding element type (line 292)
        str_5631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 60), 'str', '/MDd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 41), list_5628, str_5631)
        # Adding element type (line 292)
        str_5632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 68), 'str', '/W3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 41), list_5628, str_5632)
        # Adding element type (line 292)
        str_5633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 75), 'str', '/GX')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 41), list_5628, str_5633)
        # Adding element type (line 292)
        str_5634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 42), 'str', '/Z7')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 41), list_5628, str_5634)
        # Adding element type (line 292)
        str_5635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 49), 'str', '/D_DEBUG')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 41), list_5628, str_5635)
        
        # Getting the type of 'self' (line 292)
        self_5636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'self')
        # Setting the type of the member 'compile_options_debug' of a type (line 292)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 12), self_5636, 'compile_options_debug', list_5628)
        # SSA branch for the else part of an if statement (line 289)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Attribute (line 296):
        
        # Assigning a List to a Attribute (line 296):
        
        # Obtaining an instance of the builtin type 'list' (line 296)
        list_5637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 296)
        # Adding element type (line 296)
        str_5638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 37), 'str', '/nologo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 35), list_5637, str_5638)
        # Adding element type (line 296)
        str_5639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 48), 'str', '/Ox')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 35), list_5637, str_5639)
        # Adding element type (line 296)
        str_5640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 55), 'str', '/MD')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 35), list_5637, str_5640)
        # Adding element type (line 296)
        str_5641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 62), 'str', '/W3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 35), list_5637, str_5641)
        # Adding element type (line 296)
        str_5642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 69), 'str', '/GS-')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 35), list_5637, str_5642)
        # Adding element type (line 296)
        str_5643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 37), 'str', '/DNDEBUG')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 35), list_5637, str_5643)
        
        # Getting the type of 'self' (line 296)
        self_5644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'self')
        # Setting the type of the member 'compile_options' of a type (line 296)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 12), self_5644, 'compile_options', list_5637)
        
        # Assigning a List to a Attribute (line 298):
        
        # Assigning a List to a Attribute (line 298):
        
        # Obtaining an instance of the builtin type 'list' (line 298)
        list_5645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 298)
        # Adding element type (line 298)
        str_5646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 42), 'str', '/nologo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 41), list_5645, str_5646)
        # Adding element type (line 298)
        str_5647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 53), 'str', '/Od')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 41), list_5645, str_5647)
        # Adding element type (line 298)
        str_5648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 60), 'str', '/MDd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 41), list_5645, str_5648)
        # Adding element type (line 298)
        str_5649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 68), 'str', '/W3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 41), list_5645, str_5649)
        # Adding element type (line 298)
        str_5650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 75), 'str', '/GS-')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 41), list_5645, str_5650)
        # Adding element type (line 298)
        str_5651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 42), 'str', '/Z7')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 41), list_5645, str_5651)
        # Adding element type (line 298)
        str_5652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 49), 'str', '/D_DEBUG')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 41), list_5645, str_5652)
        
        # Getting the type of 'self' (line 298)
        self_5653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'self')
        # Setting the type of the member 'compile_options_debug' of a type (line 298)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 12), self_5653, 'compile_options_debug', list_5645)
        # SSA join for if statement (line 289)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Attribute (line 301):
        
        # Assigning a List to a Attribute (line 301):
        
        # Obtaining an instance of the builtin type 'list' (line 301)
        list_5654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 301)
        # Adding element type (line 301)
        str_5655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 31), 'str', '/DLL')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 30), list_5654, str_5655)
        # Adding element type (line 301)
        str_5656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 39), 'str', '/nologo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 30), list_5654, str_5656)
        # Adding element type (line 301)
        str_5657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 50), 'str', '/INCREMENTAL:NO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 30), list_5654, str_5657)
        
        # Getting the type of 'self' (line 301)
        self_5658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'self')
        # Setting the type of the member 'ldflags_shared' of a type (line 301)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 8), self_5658, 'ldflags_shared', list_5654)
        
        
        # Getting the type of 'self' (line 302)
        self_5659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 11), 'self')
        # Obtaining the member '__version' of a type (line 302)
        version_5660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 11), self_5659, '__version')
        int_5661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 29), 'int')
        # Applying the binary operator '>=' (line 302)
        result_ge_5662 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 11), '>=', version_5660, int_5661)
        
        # Testing the type of an if condition (line 302)
        if_condition_5663 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 302, 8), result_ge_5662)
        # Assigning a type to the variable 'if_condition_5663' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'if_condition_5663', if_condition_5663)
        # SSA begins for if statement (line 302)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Attribute (line 303):
        
        # Assigning a List to a Attribute (line 303):
        
        # Obtaining an instance of the builtin type 'list' (line 303)
        list_5664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 303)
        # Adding element type (line 303)
        str_5665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 16), 'str', '/DLL')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 40), list_5664, str_5665)
        # Adding element type (line 303)
        str_5666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 24), 'str', '/nologo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 40), list_5664, str_5666)
        # Adding element type (line 303)
        str_5667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 35), 'str', '/INCREMENTAL:no')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 40), list_5664, str_5667)
        # Adding element type (line 303)
        str_5668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 54), 'str', '/DEBUG')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 40), list_5664, str_5668)
        
        # Getting the type of 'self' (line 303)
        self_5669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 'self')
        # Setting the type of the member 'ldflags_shared_debug' of a type (line 303)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 12), self_5669, 'ldflags_shared_debug', list_5664)
        # SSA branch for the else part of an if statement (line 302)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Attribute (line 307):
        
        # Assigning a List to a Attribute (line 307):
        
        # Obtaining an instance of the builtin type 'list' (line 307)
        list_5670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 307)
        # Adding element type (line 307)
        str_5671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 16), 'str', '/DLL')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 40), list_5670, str_5671)
        # Adding element type (line 307)
        str_5672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 24), 'str', '/nologo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 40), list_5670, str_5672)
        # Adding element type (line 307)
        str_5673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 35), 'str', '/INCREMENTAL:no')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 40), list_5670, str_5673)
        # Adding element type (line 307)
        str_5674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 54), 'str', '/pdb:None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 40), list_5670, str_5674)
        # Adding element type (line 307)
        str_5675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 67), 'str', '/DEBUG')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 40), list_5670, str_5675)
        
        # Getting the type of 'self' (line 307)
        self_5676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'self')
        # Setting the type of the member 'ldflags_shared_debug' of a type (line 307)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 12), self_5676, 'ldflags_shared_debug', list_5670)
        # SSA join for if statement (line 302)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Attribute (line 310):
        
        # Assigning a List to a Attribute (line 310):
        
        # Obtaining an instance of the builtin type 'list' (line 310)
        list_5677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 310)
        # Adding element type (line 310)
        str_5678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 32), 'str', '/nologo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 30), list_5677, str_5678)
        
        # Getting the type of 'self' (line 310)
        self_5679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'self')
        # Setting the type of the member 'ldflags_static' of a type (line 310)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 8), self_5679, 'ldflags_static', list_5677)
        
        # Assigning a Name to a Attribute (line 312):
        
        # Assigning a Name to a Attribute (line 312):
        # Getting the type of 'True' (line 312)
        True_5680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 27), 'True')
        # Getting the type of 'self' (line 312)
        self_5681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'self')
        # Setting the type of the member 'initialized' of a type (line 312)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 8), self_5681, 'initialized', True_5680)
        
        # ################# End of 'initialize(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize' in the type store
        # Getting the type of 'stypy_return_type' (line 252)
        stypy_return_type_5682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5682)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize'
        return stypy_return_type_5682


    @norecursion
    def object_filenames(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_5683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 36), 'int')
        str_5684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 37), 'str', '')
        defaults = [int_5683, str_5684]
        # Create a new context for function 'object_filenames'
        module_type_store = module_type_store.open_function_context('object_filenames', 316, 4, False)
        # Assigning a type to the variable 'self' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MSVCCompiler.object_filenames.__dict__.__setitem__('stypy_localization', localization)
        MSVCCompiler.object_filenames.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MSVCCompiler.object_filenames.__dict__.__setitem__('stypy_type_store', module_type_store)
        MSVCCompiler.object_filenames.__dict__.__setitem__('stypy_function_name', 'MSVCCompiler.object_filenames')
        MSVCCompiler.object_filenames.__dict__.__setitem__('stypy_param_names_list', ['source_filenames', 'strip_dir', 'output_dir'])
        MSVCCompiler.object_filenames.__dict__.__setitem__('stypy_varargs_param_name', None)
        MSVCCompiler.object_filenames.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MSVCCompiler.object_filenames.__dict__.__setitem__('stypy_call_defaults', defaults)
        MSVCCompiler.object_filenames.__dict__.__setitem__('stypy_call_varargs', varargs)
        MSVCCompiler.object_filenames.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MSVCCompiler.object_filenames.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MSVCCompiler.object_filenames', ['source_filenames', 'strip_dir', 'output_dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'object_filenames', localization, ['source_filenames', 'strip_dir', 'output_dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'object_filenames(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 322)
        # Getting the type of 'output_dir' (line 322)
        output_dir_5685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 11), 'output_dir')
        # Getting the type of 'None' (line 322)
        None_5686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 25), 'None')
        
        (may_be_5687, more_types_in_union_5688) = may_be_none(output_dir_5685, None_5686)

        if may_be_5687:

            if more_types_in_union_5688:
                # Runtime conditional SSA (line 322)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Str to a Name (line 322):
            
            # Assigning a Str to a Name (line 322):
            str_5689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 44), 'str', '')
            # Assigning a type to the variable 'output_dir' (line 322)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 31), 'output_dir', str_5689)

            if more_types_in_union_5688:
                # SSA join for if statement (line 322)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a List to a Name (line 323):
        
        # Assigning a List to a Name (line 323):
        
        # Obtaining an instance of the builtin type 'list' (line 323)
        list_5690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 323)
        
        # Assigning a type to the variable 'obj_names' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'obj_names', list_5690)
        
        # Getting the type of 'source_filenames' (line 324)
        source_filenames_5691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 24), 'source_filenames')
        # Testing the type of a for loop iterable (line 324)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 324, 8), source_filenames_5691)
        # Getting the type of the for loop variable (line 324)
        for_loop_var_5692 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 324, 8), source_filenames_5691)
        # Assigning a type to the variable 'src_name' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'src_name', for_loop_var_5692)
        # SSA begins for a for statement (line 324)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 325):
        
        # Assigning a Subscript to a Name (line 325):
        
        # Obtaining the type of the subscript
        int_5693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 12), 'int')
        
        # Call to splitext(...): (line 325)
        # Processing the call arguments (line 325)
        # Getting the type of 'src_name' (line 325)
        src_name_5697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 44), 'src_name', False)
        # Processing the call keyword arguments (line 325)
        kwargs_5698 = {}
        # Getting the type of 'os' (line 325)
        os_5694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 325)
        path_5695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 26), os_5694, 'path')
        # Obtaining the member 'splitext' of a type (line 325)
        splitext_5696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 26), path_5695, 'splitext')
        # Calling splitext(args, kwargs) (line 325)
        splitext_call_result_5699 = invoke(stypy.reporting.localization.Localization(__file__, 325, 26), splitext_5696, *[src_name_5697], **kwargs_5698)
        
        # Obtaining the member '__getitem__' of a type (line 325)
        getitem___5700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 12), splitext_call_result_5699, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 325)
        subscript_call_result_5701 = invoke(stypy.reporting.localization.Localization(__file__, 325, 12), getitem___5700, int_5693)
        
        # Assigning a type to the variable 'tuple_var_assignment_5019' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'tuple_var_assignment_5019', subscript_call_result_5701)
        
        # Assigning a Subscript to a Name (line 325):
        
        # Obtaining the type of the subscript
        int_5702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 12), 'int')
        
        # Call to splitext(...): (line 325)
        # Processing the call arguments (line 325)
        # Getting the type of 'src_name' (line 325)
        src_name_5706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 44), 'src_name', False)
        # Processing the call keyword arguments (line 325)
        kwargs_5707 = {}
        # Getting the type of 'os' (line 325)
        os_5703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 325)
        path_5704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 26), os_5703, 'path')
        # Obtaining the member 'splitext' of a type (line 325)
        splitext_5705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 26), path_5704, 'splitext')
        # Calling splitext(args, kwargs) (line 325)
        splitext_call_result_5708 = invoke(stypy.reporting.localization.Localization(__file__, 325, 26), splitext_5705, *[src_name_5706], **kwargs_5707)
        
        # Obtaining the member '__getitem__' of a type (line 325)
        getitem___5709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 12), splitext_call_result_5708, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 325)
        subscript_call_result_5710 = invoke(stypy.reporting.localization.Localization(__file__, 325, 12), getitem___5709, int_5702)
        
        # Assigning a type to the variable 'tuple_var_assignment_5020' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'tuple_var_assignment_5020', subscript_call_result_5710)
        
        # Assigning a Name to a Name (line 325):
        # Getting the type of 'tuple_var_assignment_5019' (line 325)
        tuple_var_assignment_5019_5711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'tuple_var_assignment_5019')
        # Assigning a type to the variable 'base' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 13), 'base', tuple_var_assignment_5019_5711)
        
        # Assigning a Name to a Name (line 325):
        # Getting the type of 'tuple_var_assignment_5020' (line 325)
        tuple_var_assignment_5020_5712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'tuple_var_assignment_5020')
        # Assigning a type to the variable 'ext' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 19), 'ext', tuple_var_assignment_5020_5712)
        
        # Assigning a Subscript to a Name (line 326):
        
        # Assigning a Subscript to a Name (line 326):
        
        # Obtaining the type of the subscript
        int_5713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 44), 'int')
        
        # Call to splitdrive(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'base' (line 326)
        base_5717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 38), 'base', False)
        # Processing the call keyword arguments (line 326)
        kwargs_5718 = {}
        # Getting the type of 'os' (line 326)
        os_5714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 326)
        path_5715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 19), os_5714, 'path')
        # Obtaining the member 'splitdrive' of a type (line 326)
        splitdrive_5716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 19), path_5715, 'splitdrive')
        # Calling splitdrive(args, kwargs) (line 326)
        splitdrive_call_result_5719 = invoke(stypy.reporting.localization.Localization(__file__, 326, 19), splitdrive_5716, *[base_5717], **kwargs_5718)
        
        # Obtaining the member '__getitem__' of a type (line 326)
        getitem___5720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 19), splitdrive_call_result_5719, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 326)
        subscript_call_result_5721 = invoke(stypy.reporting.localization.Localization(__file__, 326, 19), getitem___5720, int_5713)
        
        # Assigning a type to the variable 'base' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'base', subscript_call_result_5721)
        
        # Assigning a Subscript to a Name (line 327):
        
        # Assigning a Subscript to a Name (line 327):
        
        # Obtaining the type of the subscript
        
        # Call to isabs(...): (line 327)
        # Processing the call arguments (line 327)
        # Getting the type of 'base' (line 327)
        base_5725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 38), 'base', False)
        # Processing the call keyword arguments (line 327)
        kwargs_5726 = {}
        # Getting the type of 'os' (line 327)
        os_5722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 327)
        path_5723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 24), os_5722, 'path')
        # Obtaining the member 'isabs' of a type (line 327)
        isabs_5724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 24), path_5723, 'isabs')
        # Calling isabs(args, kwargs) (line 327)
        isabs_call_result_5727 = invoke(stypy.reporting.localization.Localization(__file__, 327, 24), isabs_5724, *[base_5725], **kwargs_5726)
        
        slice_5728 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 327, 19), isabs_call_result_5727, None, None)
        # Getting the type of 'base' (line 327)
        base_5729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 19), 'base')
        # Obtaining the member '__getitem__' of a type (line 327)
        getitem___5730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 19), base_5729, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 327)
        subscript_call_result_5731 = invoke(stypy.reporting.localization.Localization(__file__, 327, 19), getitem___5730, slice_5728)
        
        # Assigning a type to the variable 'base' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'base', subscript_call_result_5731)
        
        
        # Getting the type of 'ext' (line 328)
        ext_5732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 15), 'ext')
        # Getting the type of 'self' (line 328)
        self_5733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 26), 'self')
        # Obtaining the member 'src_extensions' of a type (line 328)
        src_extensions_5734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 26), self_5733, 'src_extensions')
        # Applying the binary operator 'notin' (line 328)
        result_contains_5735 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 15), 'notin', ext_5732, src_extensions_5734)
        
        # Testing the type of an if condition (line 328)
        if_condition_5736 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 328, 12), result_contains_5735)
        # Assigning a type to the variable 'if_condition_5736' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'if_condition_5736', if_condition_5736)
        # SSA begins for if statement (line 328)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to CompileError(...): (line 332)
        # Processing the call arguments (line 332)
        str_5738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 36), 'str', "Don't know how to compile %s")
        # Getting the type of 'src_name' (line 332)
        src_name_5739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 69), 'src_name', False)
        # Applying the binary operator '%' (line 332)
        result_mod_5740 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 36), '%', str_5738, src_name_5739)
        
        # Processing the call keyword arguments (line 332)
        kwargs_5741 = {}
        # Getting the type of 'CompileError' (line 332)
        CompileError_5737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 22), 'CompileError', False)
        # Calling CompileError(args, kwargs) (line 332)
        CompileError_call_result_5742 = invoke(stypy.reporting.localization.Localization(__file__, 332, 22), CompileError_5737, *[result_mod_5740], **kwargs_5741)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 332, 16), CompileError_call_result_5742, 'raise parameter', BaseException)
        # SSA join for if statement (line 328)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'strip_dir' (line 333)
        strip_dir_5743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 15), 'strip_dir')
        # Testing the type of an if condition (line 333)
        if_condition_5744 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 333, 12), strip_dir_5743)
        # Assigning a type to the variable 'if_condition_5744' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'if_condition_5744', if_condition_5744)
        # SSA begins for if statement (line 333)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 334):
        
        # Assigning a Call to a Name (line 334):
        
        # Call to basename(...): (line 334)
        # Processing the call arguments (line 334)
        # Getting the type of 'base' (line 334)
        base_5748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 41), 'base', False)
        # Processing the call keyword arguments (line 334)
        kwargs_5749 = {}
        # Getting the type of 'os' (line 334)
        os_5745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 334)
        path_5746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 23), os_5745, 'path')
        # Obtaining the member 'basename' of a type (line 334)
        basename_5747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 23), path_5746, 'basename')
        # Calling basename(args, kwargs) (line 334)
        basename_call_result_5750 = invoke(stypy.reporting.localization.Localization(__file__, 334, 23), basename_5747, *[base_5748], **kwargs_5749)
        
        # Assigning a type to the variable 'base' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'base', basename_call_result_5750)
        # SSA join for if statement (line 333)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'ext' (line 335)
        ext_5751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 15), 'ext')
        # Getting the type of 'self' (line 335)
        self_5752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 22), 'self')
        # Obtaining the member '_rc_extensions' of a type (line 335)
        _rc_extensions_5753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 22), self_5752, '_rc_extensions')
        # Applying the binary operator 'in' (line 335)
        result_contains_5754 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 15), 'in', ext_5751, _rc_extensions_5753)
        
        # Testing the type of an if condition (line 335)
        if_condition_5755 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 335, 12), result_contains_5754)
        # Assigning a type to the variable 'if_condition_5755' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'if_condition_5755', if_condition_5755)
        # SSA begins for if statement (line 335)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 336)
        # Processing the call arguments (line 336)
        
        # Call to join(...): (line 336)
        # Processing the call arguments (line 336)
        # Getting the type of 'output_dir' (line 336)
        output_dir_5761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 48), 'output_dir', False)
        # Getting the type of 'base' (line 337)
        base_5762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 48), 'base', False)
        # Getting the type of 'self' (line 337)
        self_5763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 55), 'self', False)
        # Obtaining the member 'res_extension' of a type (line 337)
        res_extension_5764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 55), self_5763, 'res_extension')
        # Applying the binary operator '+' (line 337)
        result_add_5765 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 48), '+', base_5762, res_extension_5764)
        
        # Processing the call keyword arguments (line 336)
        kwargs_5766 = {}
        # Getting the type of 'os' (line 336)
        os_5758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 34), 'os', False)
        # Obtaining the member 'path' of a type (line 336)
        path_5759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 34), os_5758, 'path')
        # Obtaining the member 'join' of a type (line 336)
        join_5760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 34), path_5759, 'join')
        # Calling join(args, kwargs) (line 336)
        join_call_result_5767 = invoke(stypy.reporting.localization.Localization(__file__, 336, 34), join_5760, *[output_dir_5761, result_add_5765], **kwargs_5766)
        
        # Processing the call keyword arguments (line 336)
        kwargs_5768 = {}
        # Getting the type of 'obj_names' (line 336)
        obj_names_5756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 16), 'obj_names', False)
        # Obtaining the member 'append' of a type (line 336)
        append_5757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 16), obj_names_5756, 'append')
        # Calling append(args, kwargs) (line 336)
        append_call_result_5769 = invoke(stypy.reporting.localization.Localization(__file__, 336, 16), append_5757, *[join_call_result_5767], **kwargs_5768)
        
        # SSA branch for the else part of an if statement (line 335)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'ext' (line 338)
        ext_5770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 17), 'ext')
        # Getting the type of 'self' (line 338)
        self_5771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 24), 'self')
        # Obtaining the member '_mc_extensions' of a type (line 338)
        _mc_extensions_5772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 24), self_5771, '_mc_extensions')
        # Applying the binary operator 'in' (line 338)
        result_contains_5773 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 17), 'in', ext_5770, _mc_extensions_5772)
        
        # Testing the type of an if condition (line 338)
        if_condition_5774 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 338, 17), result_contains_5773)
        # Assigning a type to the variable 'if_condition_5774' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 17), 'if_condition_5774', if_condition_5774)
        # SSA begins for if statement (line 338)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 339)
        # Processing the call arguments (line 339)
        
        # Call to join(...): (line 339)
        # Processing the call arguments (line 339)
        # Getting the type of 'output_dir' (line 339)
        output_dir_5780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 48), 'output_dir', False)
        # Getting the type of 'base' (line 340)
        base_5781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 48), 'base', False)
        # Getting the type of 'self' (line 340)
        self_5782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 55), 'self', False)
        # Obtaining the member 'res_extension' of a type (line 340)
        res_extension_5783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 55), self_5782, 'res_extension')
        # Applying the binary operator '+' (line 340)
        result_add_5784 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 48), '+', base_5781, res_extension_5783)
        
        # Processing the call keyword arguments (line 339)
        kwargs_5785 = {}
        # Getting the type of 'os' (line 339)
        os_5777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 34), 'os', False)
        # Obtaining the member 'path' of a type (line 339)
        path_5778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 34), os_5777, 'path')
        # Obtaining the member 'join' of a type (line 339)
        join_5779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 34), path_5778, 'join')
        # Calling join(args, kwargs) (line 339)
        join_call_result_5786 = invoke(stypy.reporting.localization.Localization(__file__, 339, 34), join_5779, *[output_dir_5780, result_add_5784], **kwargs_5785)
        
        # Processing the call keyword arguments (line 339)
        kwargs_5787 = {}
        # Getting the type of 'obj_names' (line 339)
        obj_names_5775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 16), 'obj_names', False)
        # Obtaining the member 'append' of a type (line 339)
        append_5776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 16), obj_names_5775, 'append')
        # Calling append(args, kwargs) (line 339)
        append_call_result_5788 = invoke(stypy.reporting.localization.Localization(__file__, 339, 16), append_5776, *[join_call_result_5786], **kwargs_5787)
        
        # SSA branch for the else part of an if statement (line 338)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 342)
        # Processing the call arguments (line 342)
        
        # Call to join(...): (line 342)
        # Processing the call arguments (line 342)
        # Getting the type of 'output_dir' (line 342)
        output_dir_5794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 48), 'output_dir', False)
        # Getting the type of 'base' (line 343)
        base_5795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 48), 'base', False)
        # Getting the type of 'self' (line 343)
        self_5796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 55), 'self', False)
        # Obtaining the member 'obj_extension' of a type (line 343)
        obj_extension_5797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 55), self_5796, 'obj_extension')
        # Applying the binary operator '+' (line 343)
        result_add_5798 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 48), '+', base_5795, obj_extension_5797)
        
        # Processing the call keyword arguments (line 342)
        kwargs_5799 = {}
        # Getting the type of 'os' (line 342)
        os_5791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 34), 'os', False)
        # Obtaining the member 'path' of a type (line 342)
        path_5792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 34), os_5791, 'path')
        # Obtaining the member 'join' of a type (line 342)
        join_5793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 34), path_5792, 'join')
        # Calling join(args, kwargs) (line 342)
        join_call_result_5800 = invoke(stypy.reporting.localization.Localization(__file__, 342, 34), join_5793, *[output_dir_5794, result_add_5798], **kwargs_5799)
        
        # Processing the call keyword arguments (line 342)
        kwargs_5801 = {}
        # Getting the type of 'obj_names' (line 342)
        obj_names_5789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 16), 'obj_names', False)
        # Obtaining the member 'append' of a type (line 342)
        append_5790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 16), obj_names_5789, 'append')
        # Calling append(args, kwargs) (line 342)
        append_call_result_5802 = invoke(stypy.reporting.localization.Localization(__file__, 342, 16), append_5790, *[join_call_result_5800], **kwargs_5801)
        
        # SSA join for if statement (line 338)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 335)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'obj_names' (line 344)
        obj_names_5803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 15), 'obj_names')
        # Assigning a type to the variable 'stypy_return_type' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'stypy_return_type', obj_names_5803)
        
        # ################# End of 'object_filenames(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'object_filenames' in the type store
        # Getting the type of 'stypy_return_type' (line 316)
        stypy_return_type_5804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5804)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'object_filenames'
        return stypy_return_type_5804


    @norecursion
    def compile(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 350)
        None_5805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 27), 'None')
        # Getting the type of 'None' (line 350)
        None_5806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 40), 'None')
        # Getting the type of 'None' (line 350)
        None_5807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 59), 'None')
        int_5808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 71), 'int')
        # Getting the type of 'None' (line 351)
        None_5809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 30), 'None')
        # Getting the type of 'None' (line 351)
        None_5810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 51), 'None')
        # Getting the type of 'None' (line 351)
        None_5811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 65), 'None')
        defaults = [None_5805, None_5806, None_5807, int_5808, None_5809, None_5810, None_5811]
        # Create a new context for function 'compile'
        module_type_store = module_type_store.open_function_context('compile', 349, 4, False)
        # Assigning a type to the variable 'self' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MSVCCompiler.compile.__dict__.__setitem__('stypy_localization', localization)
        MSVCCompiler.compile.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MSVCCompiler.compile.__dict__.__setitem__('stypy_type_store', module_type_store)
        MSVCCompiler.compile.__dict__.__setitem__('stypy_function_name', 'MSVCCompiler.compile')
        MSVCCompiler.compile.__dict__.__setitem__('stypy_param_names_list', ['sources', 'output_dir', 'macros', 'include_dirs', 'debug', 'extra_preargs', 'extra_postargs', 'depends'])
        MSVCCompiler.compile.__dict__.__setitem__('stypy_varargs_param_name', None)
        MSVCCompiler.compile.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MSVCCompiler.compile.__dict__.__setitem__('stypy_call_defaults', defaults)
        MSVCCompiler.compile.__dict__.__setitem__('stypy_call_varargs', varargs)
        MSVCCompiler.compile.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MSVCCompiler.compile.__dict__.__setitem__('stypy_declared_arg_number', 9)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MSVCCompiler.compile', ['sources', 'output_dir', 'macros', 'include_dirs', 'debug', 'extra_preargs', 'extra_postargs', 'depends'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'compile', localization, ['sources', 'output_dir', 'macros', 'include_dirs', 'debug', 'extra_preargs', 'extra_postargs', 'depends'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'compile(...)' code ##################

        
        
        # Getting the type of 'self' (line 353)
        self_5812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 15), 'self')
        # Obtaining the member 'initialized' of a type (line 353)
        initialized_5813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 15), self_5812, 'initialized')
        # Applying the 'not' unary operator (line 353)
        result_not__5814 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 11), 'not', initialized_5813)
        
        # Testing the type of an if condition (line 353)
        if_condition_5815 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 353, 8), result_not__5814)
        # Assigning a type to the variable 'if_condition_5815' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'if_condition_5815', if_condition_5815)
        # SSA begins for if statement (line 353)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to initialize(...): (line 353)
        # Processing the call keyword arguments (line 353)
        kwargs_5818 = {}
        # Getting the type of 'self' (line 353)
        self_5816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 33), 'self', False)
        # Obtaining the member 'initialize' of a type (line 353)
        initialize_5817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 33), self_5816, 'initialize')
        # Calling initialize(args, kwargs) (line 353)
        initialize_call_result_5819 = invoke(stypy.reporting.localization.Localization(__file__, 353, 33), initialize_5817, *[], **kwargs_5818)
        
        # SSA join for if statement (line 353)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 354):
        
        # Assigning a Subscript to a Name (line 354):
        
        # Obtaining the type of the subscript
        int_5820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 8), 'int')
        
        # Call to _setup_compile(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'output_dir' (line 355)
        output_dir_5823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 36), 'output_dir', False)
        # Getting the type of 'macros' (line 355)
        macros_5824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 48), 'macros', False)
        # Getting the type of 'include_dirs' (line 355)
        include_dirs_5825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 56), 'include_dirs', False)
        # Getting the type of 'sources' (line 355)
        sources_5826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 70), 'sources', False)
        # Getting the type of 'depends' (line 356)
        depends_5827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 36), 'depends', False)
        # Getting the type of 'extra_postargs' (line 356)
        extra_postargs_5828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 45), 'extra_postargs', False)
        # Processing the call keyword arguments (line 355)
        kwargs_5829 = {}
        # Getting the type of 'self' (line 355)
        self_5821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 16), 'self', False)
        # Obtaining the member '_setup_compile' of a type (line 355)
        _setup_compile_5822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 16), self_5821, '_setup_compile')
        # Calling _setup_compile(args, kwargs) (line 355)
        _setup_compile_call_result_5830 = invoke(stypy.reporting.localization.Localization(__file__, 355, 16), _setup_compile_5822, *[output_dir_5823, macros_5824, include_dirs_5825, sources_5826, depends_5827, extra_postargs_5828], **kwargs_5829)
        
        # Obtaining the member '__getitem__' of a type (line 354)
        getitem___5831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 8), _setup_compile_call_result_5830, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 354)
        subscript_call_result_5832 = invoke(stypy.reporting.localization.Localization(__file__, 354, 8), getitem___5831, int_5820)
        
        # Assigning a type to the variable 'tuple_var_assignment_5021' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'tuple_var_assignment_5021', subscript_call_result_5832)
        
        # Assigning a Subscript to a Name (line 354):
        
        # Obtaining the type of the subscript
        int_5833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 8), 'int')
        
        # Call to _setup_compile(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'output_dir' (line 355)
        output_dir_5836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 36), 'output_dir', False)
        # Getting the type of 'macros' (line 355)
        macros_5837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 48), 'macros', False)
        # Getting the type of 'include_dirs' (line 355)
        include_dirs_5838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 56), 'include_dirs', False)
        # Getting the type of 'sources' (line 355)
        sources_5839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 70), 'sources', False)
        # Getting the type of 'depends' (line 356)
        depends_5840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 36), 'depends', False)
        # Getting the type of 'extra_postargs' (line 356)
        extra_postargs_5841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 45), 'extra_postargs', False)
        # Processing the call keyword arguments (line 355)
        kwargs_5842 = {}
        # Getting the type of 'self' (line 355)
        self_5834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 16), 'self', False)
        # Obtaining the member '_setup_compile' of a type (line 355)
        _setup_compile_5835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 16), self_5834, '_setup_compile')
        # Calling _setup_compile(args, kwargs) (line 355)
        _setup_compile_call_result_5843 = invoke(stypy.reporting.localization.Localization(__file__, 355, 16), _setup_compile_5835, *[output_dir_5836, macros_5837, include_dirs_5838, sources_5839, depends_5840, extra_postargs_5841], **kwargs_5842)
        
        # Obtaining the member '__getitem__' of a type (line 354)
        getitem___5844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 8), _setup_compile_call_result_5843, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 354)
        subscript_call_result_5845 = invoke(stypy.reporting.localization.Localization(__file__, 354, 8), getitem___5844, int_5833)
        
        # Assigning a type to the variable 'tuple_var_assignment_5022' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'tuple_var_assignment_5022', subscript_call_result_5845)
        
        # Assigning a Subscript to a Name (line 354):
        
        # Obtaining the type of the subscript
        int_5846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 8), 'int')
        
        # Call to _setup_compile(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'output_dir' (line 355)
        output_dir_5849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 36), 'output_dir', False)
        # Getting the type of 'macros' (line 355)
        macros_5850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 48), 'macros', False)
        # Getting the type of 'include_dirs' (line 355)
        include_dirs_5851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 56), 'include_dirs', False)
        # Getting the type of 'sources' (line 355)
        sources_5852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 70), 'sources', False)
        # Getting the type of 'depends' (line 356)
        depends_5853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 36), 'depends', False)
        # Getting the type of 'extra_postargs' (line 356)
        extra_postargs_5854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 45), 'extra_postargs', False)
        # Processing the call keyword arguments (line 355)
        kwargs_5855 = {}
        # Getting the type of 'self' (line 355)
        self_5847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 16), 'self', False)
        # Obtaining the member '_setup_compile' of a type (line 355)
        _setup_compile_5848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 16), self_5847, '_setup_compile')
        # Calling _setup_compile(args, kwargs) (line 355)
        _setup_compile_call_result_5856 = invoke(stypy.reporting.localization.Localization(__file__, 355, 16), _setup_compile_5848, *[output_dir_5849, macros_5850, include_dirs_5851, sources_5852, depends_5853, extra_postargs_5854], **kwargs_5855)
        
        # Obtaining the member '__getitem__' of a type (line 354)
        getitem___5857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 8), _setup_compile_call_result_5856, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 354)
        subscript_call_result_5858 = invoke(stypy.reporting.localization.Localization(__file__, 354, 8), getitem___5857, int_5846)
        
        # Assigning a type to the variable 'tuple_var_assignment_5023' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'tuple_var_assignment_5023', subscript_call_result_5858)
        
        # Assigning a Subscript to a Name (line 354):
        
        # Obtaining the type of the subscript
        int_5859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 8), 'int')
        
        # Call to _setup_compile(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'output_dir' (line 355)
        output_dir_5862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 36), 'output_dir', False)
        # Getting the type of 'macros' (line 355)
        macros_5863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 48), 'macros', False)
        # Getting the type of 'include_dirs' (line 355)
        include_dirs_5864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 56), 'include_dirs', False)
        # Getting the type of 'sources' (line 355)
        sources_5865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 70), 'sources', False)
        # Getting the type of 'depends' (line 356)
        depends_5866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 36), 'depends', False)
        # Getting the type of 'extra_postargs' (line 356)
        extra_postargs_5867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 45), 'extra_postargs', False)
        # Processing the call keyword arguments (line 355)
        kwargs_5868 = {}
        # Getting the type of 'self' (line 355)
        self_5860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 16), 'self', False)
        # Obtaining the member '_setup_compile' of a type (line 355)
        _setup_compile_5861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 16), self_5860, '_setup_compile')
        # Calling _setup_compile(args, kwargs) (line 355)
        _setup_compile_call_result_5869 = invoke(stypy.reporting.localization.Localization(__file__, 355, 16), _setup_compile_5861, *[output_dir_5862, macros_5863, include_dirs_5864, sources_5865, depends_5866, extra_postargs_5867], **kwargs_5868)
        
        # Obtaining the member '__getitem__' of a type (line 354)
        getitem___5870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 8), _setup_compile_call_result_5869, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 354)
        subscript_call_result_5871 = invoke(stypy.reporting.localization.Localization(__file__, 354, 8), getitem___5870, int_5859)
        
        # Assigning a type to the variable 'tuple_var_assignment_5024' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'tuple_var_assignment_5024', subscript_call_result_5871)
        
        # Assigning a Subscript to a Name (line 354):
        
        # Obtaining the type of the subscript
        int_5872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 8), 'int')
        
        # Call to _setup_compile(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'output_dir' (line 355)
        output_dir_5875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 36), 'output_dir', False)
        # Getting the type of 'macros' (line 355)
        macros_5876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 48), 'macros', False)
        # Getting the type of 'include_dirs' (line 355)
        include_dirs_5877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 56), 'include_dirs', False)
        # Getting the type of 'sources' (line 355)
        sources_5878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 70), 'sources', False)
        # Getting the type of 'depends' (line 356)
        depends_5879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 36), 'depends', False)
        # Getting the type of 'extra_postargs' (line 356)
        extra_postargs_5880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 45), 'extra_postargs', False)
        # Processing the call keyword arguments (line 355)
        kwargs_5881 = {}
        # Getting the type of 'self' (line 355)
        self_5873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 16), 'self', False)
        # Obtaining the member '_setup_compile' of a type (line 355)
        _setup_compile_5874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 16), self_5873, '_setup_compile')
        # Calling _setup_compile(args, kwargs) (line 355)
        _setup_compile_call_result_5882 = invoke(stypy.reporting.localization.Localization(__file__, 355, 16), _setup_compile_5874, *[output_dir_5875, macros_5876, include_dirs_5877, sources_5878, depends_5879, extra_postargs_5880], **kwargs_5881)
        
        # Obtaining the member '__getitem__' of a type (line 354)
        getitem___5883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 8), _setup_compile_call_result_5882, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 354)
        subscript_call_result_5884 = invoke(stypy.reporting.localization.Localization(__file__, 354, 8), getitem___5883, int_5872)
        
        # Assigning a type to the variable 'tuple_var_assignment_5025' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'tuple_var_assignment_5025', subscript_call_result_5884)
        
        # Assigning a Name to a Name (line 354):
        # Getting the type of 'tuple_var_assignment_5021' (line 354)
        tuple_var_assignment_5021_5885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'tuple_var_assignment_5021')
        # Assigning a type to the variable 'macros' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'macros', tuple_var_assignment_5021_5885)
        
        # Assigning a Name to a Name (line 354):
        # Getting the type of 'tuple_var_assignment_5022' (line 354)
        tuple_var_assignment_5022_5886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'tuple_var_assignment_5022')
        # Assigning a type to the variable 'objects' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 16), 'objects', tuple_var_assignment_5022_5886)
        
        # Assigning a Name to a Name (line 354):
        # Getting the type of 'tuple_var_assignment_5023' (line 354)
        tuple_var_assignment_5023_5887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'tuple_var_assignment_5023')
        # Assigning a type to the variable 'extra_postargs' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 25), 'extra_postargs', tuple_var_assignment_5023_5887)
        
        # Assigning a Name to a Name (line 354):
        # Getting the type of 'tuple_var_assignment_5024' (line 354)
        tuple_var_assignment_5024_5888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'tuple_var_assignment_5024')
        # Assigning a type to the variable 'pp_opts' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 41), 'pp_opts', tuple_var_assignment_5024_5888)
        
        # Assigning a Name to a Name (line 354):
        # Getting the type of 'tuple_var_assignment_5025' (line 354)
        tuple_var_assignment_5025_5889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'tuple_var_assignment_5025')
        # Assigning a type to the variable 'build' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 50), 'build', tuple_var_assignment_5025_5889)
        
        # Assigning a BoolOp to a Name (line 358):
        
        # Assigning a BoolOp to a Name (line 358):
        
        # Evaluating a boolean operation
        # Getting the type of 'extra_preargs' (line 358)
        extra_preargs_5890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 23), 'extra_preargs')
        
        # Obtaining an instance of the builtin type 'list' (line 358)
        list_5891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 358)
        
        # Applying the binary operator 'or' (line 358)
        result_or_keyword_5892 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 23), 'or', extra_preargs_5890, list_5891)
        
        # Assigning a type to the variable 'compile_opts' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'compile_opts', result_or_keyword_5892)
        
        # Call to append(...): (line 359)
        # Processing the call arguments (line 359)
        str_5895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 29), 'str', '/c')
        # Processing the call keyword arguments (line 359)
        kwargs_5896 = {}
        # Getting the type of 'compile_opts' (line 359)
        compile_opts_5893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'compile_opts', False)
        # Obtaining the member 'append' of a type (line 359)
        append_5894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 8), compile_opts_5893, 'append')
        # Calling append(args, kwargs) (line 359)
        append_call_result_5897 = invoke(stypy.reporting.localization.Localization(__file__, 359, 8), append_5894, *[str_5895], **kwargs_5896)
        
        
        # Getting the type of 'debug' (line 360)
        debug_5898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 11), 'debug')
        # Testing the type of an if condition (line 360)
        if_condition_5899 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 360, 8), debug_5898)
        # Assigning a type to the variable 'if_condition_5899' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'if_condition_5899', if_condition_5899)
        # SSA begins for if statement (line 360)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 361)
        # Processing the call arguments (line 361)
        # Getting the type of 'self' (line 361)
        self_5902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 32), 'self', False)
        # Obtaining the member 'compile_options_debug' of a type (line 361)
        compile_options_debug_5903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 32), self_5902, 'compile_options_debug')
        # Processing the call keyword arguments (line 361)
        kwargs_5904 = {}
        # Getting the type of 'compile_opts' (line 361)
        compile_opts_5900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'compile_opts', False)
        # Obtaining the member 'extend' of a type (line 361)
        extend_5901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 12), compile_opts_5900, 'extend')
        # Calling extend(args, kwargs) (line 361)
        extend_call_result_5905 = invoke(stypy.reporting.localization.Localization(__file__, 361, 12), extend_5901, *[compile_options_debug_5903], **kwargs_5904)
        
        # SSA branch for the else part of an if statement (line 360)
        module_type_store.open_ssa_branch('else')
        
        # Call to extend(...): (line 363)
        # Processing the call arguments (line 363)
        # Getting the type of 'self' (line 363)
        self_5908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 32), 'self', False)
        # Obtaining the member 'compile_options' of a type (line 363)
        compile_options_5909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 32), self_5908, 'compile_options')
        # Processing the call keyword arguments (line 363)
        kwargs_5910 = {}
        # Getting the type of 'compile_opts' (line 363)
        compile_opts_5906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'compile_opts', False)
        # Obtaining the member 'extend' of a type (line 363)
        extend_5907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 12), compile_opts_5906, 'extend')
        # Calling extend(args, kwargs) (line 363)
        extend_call_result_5911 = invoke(stypy.reporting.localization.Localization(__file__, 363, 12), extend_5907, *[compile_options_5909], **kwargs_5910)
        
        # SSA join for if statement (line 360)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'objects' (line 365)
        objects_5912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 19), 'objects')
        # Testing the type of a for loop iterable (line 365)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 365, 8), objects_5912)
        # Getting the type of the for loop variable (line 365)
        for_loop_var_5913 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 365, 8), objects_5912)
        # Assigning a type to the variable 'obj' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'obj', for_loop_var_5913)
        # SSA begins for a for statement (line 365)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # SSA begins for try-except statement (line 366)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Tuple (line 367):
        
        # Assigning a Subscript to a Name (line 367):
        
        # Obtaining the type of the subscript
        int_5914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 16), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'obj' (line 367)
        obj_5915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 33), 'obj')
        # Getting the type of 'build' (line 367)
        build_5916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 27), 'build')
        # Obtaining the member '__getitem__' of a type (line 367)
        getitem___5917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 27), build_5916, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 367)
        subscript_call_result_5918 = invoke(stypy.reporting.localization.Localization(__file__, 367, 27), getitem___5917, obj_5915)
        
        # Obtaining the member '__getitem__' of a type (line 367)
        getitem___5919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 16), subscript_call_result_5918, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 367)
        subscript_call_result_5920 = invoke(stypy.reporting.localization.Localization(__file__, 367, 16), getitem___5919, int_5914)
        
        # Assigning a type to the variable 'tuple_var_assignment_5026' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 16), 'tuple_var_assignment_5026', subscript_call_result_5920)
        
        # Assigning a Subscript to a Name (line 367):
        
        # Obtaining the type of the subscript
        int_5921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 16), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'obj' (line 367)
        obj_5922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 33), 'obj')
        # Getting the type of 'build' (line 367)
        build_5923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 27), 'build')
        # Obtaining the member '__getitem__' of a type (line 367)
        getitem___5924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 27), build_5923, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 367)
        subscript_call_result_5925 = invoke(stypy.reporting.localization.Localization(__file__, 367, 27), getitem___5924, obj_5922)
        
        # Obtaining the member '__getitem__' of a type (line 367)
        getitem___5926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 16), subscript_call_result_5925, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 367)
        subscript_call_result_5927 = invoke(stypy.reporting.localization.Localization(__file__, 367, 16), getitem___5926, int_5921)
        
        # Assigning a type to the variable 'tuple_var_assignment_5027' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 16), 'tuple_var_assignment_5027', subscript_call_result_5927)
        
        # Assigning a Name to a Name (line 367):
        # Getting the type of 'tuple_var_assignment_5026' (line 367)
        tuple_var_assignment_5026_5928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 16), 'tuple_var_assignment_5026')
        # Assigning a type to the variable 'src' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 16), 'src', tuple_var_assignment_5026_5928)
        
        # Assigning a Name to a Name (line 367):
        # Getting the type of 'tuple_var_assignment_5027' (line 367)
        tuple_var_assignment_5027_5929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 16), 'tuple_var_assignment_5027')
        # Assigning a type to the variable 'ext' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 21), 'ext', tuple_var_assignment_5027_5929)
        # SSA branch for the except part of a try statement (line 366)
        # SSA branch for the except 'KeyError' branch of a try statement (line 366)
        module_type_store.open_ssa_branch('except')
        # SSA join for try-except statement (line 366)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'debug' (line 370)
        debug_5930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 15), 'debug')
        # Testing the type of an if condition (line 370)
        if_condition_5931 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 370, 12), debug_5930)
        # Assigning a type to the variable 'if_condition_5931' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'if_condition_5931', if_condition_5931)
        # SSA begins for if statement (line 370)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 374):
        
        # Assigning a Call to a Name (line 374):
        
        # Call to abspath(...): (line 374)
        # Processing the call arguments (line 374)
        # Getting the type of 'src' (line 374)
        src_5935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 38), 'src', False)
        # Processing the call keyword arguments (line 374)
        kwargs_5936 = {}
        # Getting the type of 'os' (line 374)
        os_5932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 22), 'os', False)
        # Obtaining the member 'path' of a type (line 374)
        path_5933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 22), os_5932, 'path')
        # Obtaining the member 'abspath' of a type (line 374)
        abspath_5934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 22), path_5933, 'abspath')
        # Calling abspath(args, kwargs) (line 374)
        abspath_call_result_5937 = invoke(stypy.reporting.localization.Localization(__file__, 374, 22), abspath_5934, *[src_5935], **kwargs_5936)
        
        # Assigning a type to the variable 'src' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 16), 'src', abspath_call_result_5937)
        # SSA join for if statement (line 370)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'ext' (line 376)
        ext_5938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 15), 'ext')
        # Getting the type of 'self' (line 376)
        self_5939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 22), 'self')
        # Obtaining the member '_c_extensions' of a type (line 376)
        _c_extensions_5940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 22), self_5939, '_c_extensions')
        # Applying the binary operator 'in' (line 376)
        result_contains_5941 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 15), 'in', ext_5938, _c_extensions_5940)
        
        # Testing the type of an if condition (line 376)
        if_condition_5942 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 376, 12), result_contains_5941)
        # Assigning a type to the variable 'if_condition_5942' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'if_condition_5942', if_condition_5942)
        # SSA begins for if statement (line 376)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 377):
        
        # Assigning a BinOp to a Name (line 377):
        str_5943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 28), 'str', '/Tc')
        # Getting the type of 'src' (line 377)
        src_5944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 36), 'src')
        # Applying the binary operator '+' (line 377)
        result_add_5945 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 28), '+', str_5943, src_5944)
        
        # Assigning a type to the variable 'input_opt' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 16), 'input_opt', result_add_5945)
        # SSA branch for the else part of an if statement (line 376)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'ext' (line 378)
        ext_5946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 17), 'ext')
        # Getting the type of 'self' (line 378)
        self_5947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 24), 'self')
        # Obtaining the member '_cpp_extensions' of a type (line 378)
        _cpp_extensions_5948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 24), self_5947, '_cpp_extensions')
        # Applying the binary operator 'in' (line 378)
        result_contains_5949 = python_operator(stypy.reporting.localization.Localization(__file__, 378, 17), 'in', ext_5946, _cpp_extensions_5948)
        
        # Testing the type of an if condition (line 378)
        if_condition_5950 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 378, 17), result_contains_5949)
        # Assigning a type to the variable 'if_condition_5950' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 17), 'if_condition_5950', if_condition_5950)
        # SSA begins for if statement (line 378)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 379):
        
        # Assigning a BinOp to a Name (line 379):
        str_5951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 28), 'str', '/Tp')
        # Getting the type of 'src' (line 379)
        src_5952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 36), 'src')
        # Applying the binary operator '+' (line 379)
        result_add_5953 = python_operator(stypy.reporting.localization.Localization(__file__, 379, 28), '+', str_5951, src_5952)
        
        # Assigning a type to the variable 'input_opt' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 16), 'input_opt', result_add_5953)
        # SSA branch for the else part of an if statement (line 378)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'ext' (line 380)
        ext_5954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 17), 'ext')
        # Getting the type of 'self' (line 380)
        self_5955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 24), 'self')
        # Obtaining the member '_rc_extensions' of a type (line 380)
        _rc_extensions_5956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 24), self_5955, '_rc_extensions')
        # Applying the binary operator 'in' (line 380)
        result_contains_5957 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 17), 'in', ext_5954, _rc_extensions_5956)
        
        # Testing the type of an if condition (line 380)
        if_condition_5958 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 380, 17), result_contains_5957)
        # Assigning a type to the variable 'if_condition_5958' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 17), 'if_condition_5958', if_condition_5958)
        # SSA begins for if statement (line 380)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 382):
        
        # Assigning a Name to a Name (line 382):
        # Getting the type of 'src' (line 382)
        src_5959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 28), 'src')
        # Assigning a type to the variable 'input_opt' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 16), 'input_opt', src_5959)
        
        # Assigning a BinOp to a Name (line 383):
        
        # Assigning a BinOp to a Name (line 383):
        str_5960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 29), 'str', '/fo')
        # Getting the type of 'obj' (line 383)
        obj_5961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 37), 'obj')
        # Applying the binary operator '+' (line 383)
        result_add_5962 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 29), '+', str_5960, obj_5961)
        
        # Assigning a type to the variable 'output_opt' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 16), 'output_opt', result_add_5962)
        
        
        # SSA begins for try-except statement (line 384)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to spawn(...): (line 385)
        # Processing the call arguments (line 385)
        
        # Obtaining an instance of the builtin type 'list' (line 385)
        list_5965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 385)
        # Adding element type (line 385)
        # Getting the type of 'self' (line 385)
        self_5966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 33), 'self', False)
        # Obtaining the member 'rc' of a type (line 385)
        rc_5967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 33), self_5966, 'rc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 32), list_5965, rc_5967)
        
        # Getting the type of 'pp_opts' (line 385)
        pp_opts_5968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 44), 'pp_opts', False)
        # Applying the binary operator '+' (line 385)
        result_add_5969 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 32), '+', list_5965, pp_opts_5968)
        
        
        # Obtaining an instance of the builtin type 'list' (line 386)
        list_5970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 386)
        # Adding element type (line 386)
        # Getting the type of 'output_opt' (line 386)
        output_opt_5971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 33), 'output_opt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 32), list_5970, output_opt_5971)
        
        # Applying the binary operator '+' (line 385)
        result_add_5972 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 52), '+', result_add_5969, list_5970)
        
        
        # Obtaining an instance of the builtin type 'list' (line 386)
        list_5973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 386)
        # Adding element type (line 386)
        # Getting the type of 'input_opt' (line 386)
        input_opt_5974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 48), 'input_opt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 47), list_5973, input_opt_5974)
        
        # Applying the binary operator '+' (line 386)
        result_add_5975 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 45), '+', result_add_5972, list_5973)
        
        # Processing the call keyword arguments (line 385)
        kwargs_5976 = {}
        # Getting the type of 'self' (line 385)
        self_5963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 20), 'self', False)
        # Obtaining the member 'spawn' of a type (line 385)
        spawn_5964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 20), self_5963, 'spawn')
        # Calling spawn(args, kwargs) (line 385)
        spawn_call_result_5977 = invoke(stypy.reporting.localization.Localization(__file__, 385, 20), spawn_5964, *[result_add_5975], **kwargs_5976)
        
        # SSA branch for the except part of a try statement (line 384)
        # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 384)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsExecError' (line 387)
        DistutilsExecError_5978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 23), 'DistutilsExecError')
        # Assigning a type to the variable 'msg' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 16), 'msg', DistutilsExecError_5978)
        # Getting the type of 'CompileError' (line 388)
        CompileError_5979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 26), 'CompileError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 388, 20), CompileError_5979, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 384)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 380)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'ext' (line 390)
        ext_5980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 17), 'ext')
        # Getting the type of 'self' (line 390)
        self_5981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 24), 'self')
        # Obtaining the member '_mc_extensions' of a type (line 390)
        _mc_extensions_5982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 24), self_5981, '_mc_extensions')
        # Applying the binary operator 'in' (line 390)
        result_contains_5983 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 17), 'in', ext_5980, _mc_extensions_5982)
        
        # Testing the type of an if condition (line 390)
        if_condition_5984 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 390, 17), result_contains_5983)
        # Assigning a type to the variable 'if_condition_5984' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 17), 'if_condition_5984', if_condition_5984)
        # SSA begins for if statement (line 390)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 404):
        
        # Assigning a Call to a Name (line 404):
        
        # Call to dirname(...): (line 404)
        # Processing the call arguments (line 404)
        # Getting the type of 'src' (line 404)
        src_5988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 41), 'src', False)
        # Processing the call keyword arguments (line 404)
        kwargs_5989 = {}
        # Getting the type of 'os' (line 404)
        os_5985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 404)
        path_5986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 24), os_5985, 'path')
        # Obtaining the member 'dirname' of a type (line 404)
        dirname_5987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 24), path_5986, 'dirname')
        # Calling dirname(args, kwargs) (line 404)
        dirname_call_result_5990 = invoke(stypy.reporting.localization.Localization(__file__, 404, 24), dirname_5987, *[src_5988], **kwargs_5989)
        
        # Assigning a type to the variable 'h_dir' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 16), 'h_dir', dirname_call_result_5990)
        
        # Assigning a Call to a Name (line 405):
        
        # Assigning a Call to a Name (line 405):
        
        # Call to dirname(...): (line 405)
        # Processing the call arguments (line 405)
        # Getting the type of 'obj' (line 405)
        obj_5994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 42), 'obj', False)
        # Processing the call keyword arguments (line 405)
        kwargs_5995 = {}
        # Getting the type of 'os' (line 405)
        os_5991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 405)
        path_5992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 25), os_5991, 'path')
        # Obtaining the member 'dirname' of a type (line 405)
        dirname_5993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 25), path_5992, 'dirname')
        # Calling dirname(args, kwargs) (line 405)
        dirname_call_result_5996 = invoke(stypy.reporting.localization.Localization(__file__, 405, 25), dirname_5993, *[obj_5994], **kwargs_5995)
        
        # Assigning a type to the variable 'rc_dir' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 16), 'rc_dir', dirname_call_result_5996)
        
        
        # SSA begins for try-except statement (line 406)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to spawn(...): (line 408)
        # Processing the call arguments (line 408)
        
        # Obtaining an instance of the builtin type 'list' (line 408)
        list_5999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 408)
        # Adding element type (line 408)
        # Getting the type of 'self' (line 408)
        self_6000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 33), 'self', False)
        # Obtaining the member 'mc' of a type (line 408)
        mc_6001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 33), self_6000, 'mc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 32), list_5999, mc_6001)
        
        
        # Obtaining an instance of the builtin type 'list' (line 409)
        list_6002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 409)
        # Adding element type (line 409)
        str_6003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 33), 'str', '-h')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 32), list_6002, str_6003)
        # Adding element type (line 409)
        # Getting the type of 'h_dir' (line 409)
        h_dir_6004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 39), 'h_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 32), list_6002, h_dir_6004)
        # Adding element type (line 409)
        str_6005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 46), 'str', '-r')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 32), list_6002, str_6005)
        # Adding element type (line 409)
        # Getting the type of 'rc_dir' (line 409)
        rc_dir_6006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 52), 'rc_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 32), list_6002, rc_dir_6006)
        
        # Applying the binary operator '+' (line 408)
        result_add_6007 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 32), '+', list_5999, list_6002)
        
        
        # Obtaining an instance of the builtin type 'list' (line 409)
        list_6008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 62), 'list')
        # Adding type elements to the builtin type 'list' instance (line 409)
        # Adding element type (line 409)
        # Getting the type of 'src' (line 409)
        src_6009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 63), 'src', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 62), list_6008, src_6009)
        
        # Applying the binary operator '+' (line 409)
        result_add_6010 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 60), '+', result_add_6007, list_6008)
        
        # Processing the call keyword arguments (line 408)
        kwargs_6011 = {}
        # Getting the type of 'self' (line 408)
        self_5997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 20), 'self', False)
        # Obtaining the member 'spawn' of a type (line 408)
        spawn_5998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 20), self_5997, 'spawn')
        # Calling spawn(args, kwargs) (line 408)
        spawn_call_result_6012 = invoke(stypy.reporting.localization.Localization(__file__, 408, 20), spawn_5998, *[result_add_6010], **kwargs_6011)
        
        
        # Assigning a Call to a Tuple (line 410):
        
        # Assigning a Subscript to a Name (line 410):
        
        # Obtaining the type of the subscript
        int_6013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 20), 'int')
        
        # Call to splitext(...): (line 410)
        # Processing the call arguments (line 410)
        
        # Call to basename(...): (line 410)
        # Processing the call arguments (line 410)
        # Getting the type of 'src' (line 410)
        src_6020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 66), 'src', False)
        # Processing the call keyword arguments (line 410)
        kwargs_6021 = {}
        # Getting the type of 'os' (line 410)
        os_6017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 48), 'os', False)
        # Obtaining the member 'path' of a type (line 410)
        path_6018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 48), os_6017, 'path')
        # Obtaining the member 'basename' of a type (line 410)
        basename_6019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 48), path_6018, 'basename')
        # Calling basename(args, kwargs) (line 410)
        basename_call_result_6022 = invoke(stypy.reporting.localization.Localization(__file__, 410, 48), basename_6019, *[src_6020], **kwargs_6021)
        
        # Processing the call keyword arguments (line 410)
        kwargs_6023 = {}
        # Getting the type of 'os' (line 410)
        os_6014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 30), 'os', False)
        # Obtaining the member 'path' of a type (line 410)
        path_6015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 30), os_6014, 'path')
        # Obtaining the member 'splitext' of a type (line 410)
        splitext_6016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 30), path_6015, 'splitext')
        # Calling splitext(args, kwargs) (line 410)
        splitext_call_result_6024 = invoke(stypy.reporting.localization.Localization(__file__, 410, 30), splitext_6016, *[basename_call_result_6022], **kwargs_6023)
        
        # Obtaining the member '__getitem__' of a type (line 410)
        getitem___6025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 20), splitext_call_result_6024, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 410)
        subscript_call_result_6026 = invoke(stypy.reporting.localization.Localization(__file__, 410, 20), getitem___6025, int_6013)
        
        # Assigning a type to the variable 'tuple_var_assignment_5028' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 20), 'tuple_var_assignment_5028', subscript_call_result_6026)
        
        # Assigning a Subscript to a Name (line 410):
        
        # Obtaining the type of the subscript
        int_6027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 20), 'int')
        
        # Call to splitext(...): (line 410)
        # Processing the call arguments (line 410)
        
        # Call to basename(...): (line 410)
        # Processing the call arguments (line 410)
        # Getting the type of 'src' (line 410)
        src_6034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 66), 'src', False)
        # Processing the call keyword arguments (line 410)
        kwargs_6035 = {}
        # Getting the type of 'os' (line 410)
        os_6031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 48), 'os', False)
        # Obtaining the member 'path' of a type (line 410)
        path_6032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 48), os_6031, 'path')
        # Obtaining the member 'basename' of a type (line 410)
        basename_6033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 48), path_6032, 'basename')
        # Calling basename(args, kwargs) (line 410)
        basename_call_result_6036 = invoke(stypy.reporting.localization.Localization(__file__, 410, 48), basename_6033, *[src_6034], **kwargs_6035)
        
        # Processing the call keyword arguments (line 410)
        kwargs_6037 = {}
        # Getting the type of 'os' (line 410)
        os_6028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 30), 'os', False)
        # Obtaining the member 'path' of a type (line 410)
        path_6029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 30), os_6028, 'path')
        # Obtaining the member 'splitext' of a type (line 410)
        splitext_6030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 30), path_6029, 'splitext')
        # Calling splitext(args, kwargs) (line 410)
        splitext_call_result_6038 = invoke(stypy.reporting.localization.Localization(__file__, 410, 30), splitext_6030, *[basename_call_result_6036], **kwargs_6037)
        
        # Obtaining the member '__getitem__' of a type (line 410)
        getitem___6039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 20), splitext_call_result_6038, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 410)
        subscript_call_result_6040 = invoke(stypy.reporting.localization.Localization(__file__, 410, 20), getitem___6039, int_6027)
        
        # Assigning a type to the variable 'tuple_var_assignment_5029' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 20), 'tuple_var_assignment_5029', subscript_call_result_6040)
        
        # Assigning a Name to a Name (line 410):
        # Getting the type of 'tuple_var_assignment_5028' (line 410)
        tuple_var_assignment_5028_6041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 20), 'tuple_var_assignment_5028')
        # Assigning a type to the variable 'base' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 20), 'base', tuple_var_assignment_5028_6041)
        
        # Assigning a Name to a Name (line 410):
        # Getting the type of 'tuple_var_assignment_5029' (line 410)
        tuple_var_assignment_5029_6042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 20), 'tuple_var_assignment_5029')
        # Assigning a type to the variable '_' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 26), '_', tuple_var_assignment_5029_6042)
        
        # Assigning a Call to a Name (line 411):
        
        # Assigning a Call to a Name (line 411):
        
        # Call to join(...): (line 411)
        # Processing the call arguments (line 411)
        # Getting the type of 'rc_dir' (line 411)
        rc_dir_6046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 44), 'rc_dir', False)
        # Getting the type of 'base' (line 411)
        base_6047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 52), 'base', False)
        str_6048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 59), 'str', '.rc')
        # Applying the binary operator '+' (line 411)
        result_add_6049 = python_operator(stypy.reporting.localization.Localization(__file__, 411, 52), '+', base_6047, str_6048)
        
        # Processing the call keyword arguments (line 411)
        kwargs_6050 = {}
        # Getting the type of 'os' (line 411)
        os_6043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 30), 'os', False)
        # Obtaining the member 'path' of a type (line 411)
        path_6044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 30), os_6043, 'path')
        # Obtaining the member 'join' of a type (line 411)
        join_6045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 30), path_6044, 'join')
        # Calling join(args, kwargs) (line 411)
        join_call_result_6051 = invoke(stypy.reporting.localization.Localization(__file__, 411, 30), join_6045, *[rc_dir_6046, result_add_6049], **kwargs_6050)
        
        # Assigning a type to the variable 'rc_file' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 20), 'rc_file', join_call_result_6051)
        
        # Call to spawn(...): (line 413)
        # Processing the call arguments (line 413)
        
        # Obtaining an instance of the builtin type 'list' (line 413)
        list_6054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 413)
        # Adding element type (line 413)
        # Getting the type of 'self' (line 413)
        self_6055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 33), 'self', False)
        # Obtaining the member 'rc' of a type (line 413)
        rc_6056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 33), self_6055, 'rc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 32), list_6054, rc_6056)
        
        
        # Obtaining an instance of the builtin type 'list' (line 414)
        list_6057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 414)
        # Adding element type (line 414)
        str_6058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 33), 'str', '/fo')
        # Getting the type of 'obj' (line 414)
        obj_6059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 41), 'obj', False)
        # Applying the binary operator '+' (line 414)
        result_add_6060 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 33), '+', str_6058, obj_6059)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 32), list_6057, result_add_6060)
        
        # Applying the binary operator '+' (line 413)
        result_add_6061 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 32), '+', list_6054, list_6057)
        
        
        # Obtaining an instance of the builtin type 'list' (line 414)
        list_6062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 414)
        # Adding element type (line 414)
        # Getting the type of 'rc_file' (line 414)
        rc_file_6063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 49), 'rc_file', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 48), list_6062, rc_file_6063)
        
        # Applying the binary operator '+' (line 414)
        result_add_6064 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 46), '+', result_add_6061, list_6062)
        
        # Processing the call keyword arguments (line 413)
        kwargs_6065 = {}
        # Getting the type of 'self' (line 413)
        self_6052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 20), 'self', False)
        # Obtaining the member 'spawn' of a type (line 413)
        spawn_6053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 20), self_6052, 'spawn')
        # Calling spawn(args, kwargs) (line 413)
        spawn_call_result_6066 = invoke(stypy.reporting.localization.Localization(__file__, 413, 20), spawn_6053, *[result_add_6064], **kwargs_6065)
        
        # SSA branch for the except part of a try statement (line 406)
        # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 406)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsExecError' (line 416)
        DistutilsExecError_6067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 23), 'DistutilsExecError')
        # Assigning a type to the variable 'msg' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 16), 'msg', DistutilsExecError_6067)
        # Getting the type of 'CompileError' (line 417)
        CompileError_6068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 26), 'CompileError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 417, 20), CompileError_6068, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 406)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 390)
        module_type_store.open_ssa_branch('else')
        
        # Call to CompileError(...): (line 421)
        # Processing the call arguments (line 421)
        str_6070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 20), 'str', "Don't know how to compile %s to %s")
        
        # Obtaining an instance of the builtin type 'tuple' (line 423)
        tuple_6071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 423)
        # Adding element type (line 423)
        # Getting the type of 'src' (line 423)
        src_6072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 21), 'src', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 21), tuple_6071, src_6072)
        # Adding element type (line 423)
        # Getting the type of 'obj' (line 423)
        obj_6073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 26), 'obj', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 21), tuple_6071, obj_6073)
        
        # Applying the binary operator '%' (line 422)
        result_mod_6074 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 20), '%', str_6070, tuple_6071)
        
        # Processing the call keyword arguments (line 421)
        kwargs_6075 = {}
        # Getting the type of 'CompileError' (line 421)
        CompileError_6069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 22), 'CompileError', False)
        # Calling CompileError(args, kwargs) (line 421)
        CompileError_call_result_6076 = invoke(stypy.reporting.localization.Localization(__file__, 421, 22), CompileError_6069, *[result_mod_6074], **kwargs_6075)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 421, 16), CompileError_call_result_6076, 'raise parameter', BaseException)
        # SSA join for if statement (line 390)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 380)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 378)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 376)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 425):
        
        # Assigning a BinOp to a Name (line 425):
        str_6077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 25), 'str', '/Fo')
        # Getting the type of 'obj' (line 425)
        obj_6078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 33), 'obj')
        # Applying the binary operator '+' (line 425)
        result_add_6079 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 25), '+', str_6077, obj_6078)
        
        # Assigning a type to the variable 'output_opt' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'output_opt', result_add_6079)
        
        
        # SSA begins for try-except statement (line 426)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to spawn(...): (line 427)
        # Processing the call arguments (line 427)
        
        # Obtaining an instance of the builtin type 'list' (line 427)
        list_6082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 427)
        # Adding element type (line 427)
        # Getting the type of 'self' (line 427)
        self_6083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 29), 'self', False)
        # Obtaining the member 'cc' of a type (line 427)
        cc_6084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 29), self_6083, 'cc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 28), list_6082, cc_6084)
        
        # Getting the type of 'compile_opts' (line 427)
        compile_opts_6085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 40), 'compile_opts', False)
        # Applying the binary operator '+' (line 427)
        result_add_6086 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 28), '+', list_6082, compile_opts_6085)
        
        # Getting the type of 'pp_opts' (line 427)
        pp_opts_6087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 55), 'pp_opts', False)
        # Applying the binary operator '+' (line 427)
        result_add_6088 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 53), '+', result_add_6086, pp_opts_6087)
        
        
        # Obtaining an instance of the builtin type 'list' (line 428)
        list_6089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 428)
        # Adding element type (line 428)
        # Getting the type of 'input_opt' (line 428)
        input_opt_6090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 29), 'input_opt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 28), list_6089, input_opt_6090)
        # Adding element type (line 428)
        # Getting the type of 'output_opt' (line 428)
        output_opt_6091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 40), 'output_opt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 28), list_6089, output_opt_6091)
        
        # Applying the binary operator '+' (line 427)
        result_add_6092 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 63), '+', result_add_6088, list_6089)
        
        # Getting the type of 'extra_postargs' (line 429)
        extra_postargs_6093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 28), 'extra_postargs', False)
        # Applying the binary operator '+' (line 428)
        result_add_6094 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 52), '+', result_add_6092, extra_postargs_6093)
        
        # Processing the call keyword arguments (line 427)
        kwargs_6095 = {}
        # Getting the type of 'self' (line 427)
        self_6080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 16), 'self', False)
        # Obtaining the member 'spawn' of a type (line 427)
        spawn_6081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 16), self_6080, 'spawn')
        # Calling spawn(args, kwargs) (line 427)
        spawn_call_result_6096 = invoke(stypy.reporting.localization.Localization(__file__, 427, 16), spawn_6081, *[result_add_6094], **kwargs_6095)
        
        # SSA branch for the except part of a try statement (line 426)
        # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 426)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsExecError' (line 430)
        DistutilsExecError_6097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 19), 'DistutilsExecError')
        # Assigning a type to the variable 'msg' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 12), 'msg', DistutilsExecError_6097)
        # Getting the type of 'CompileError' (line 431)
        CompileError_6098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 22), 'CompileError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 431, 16), CompileError_6098, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 426)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'objects' (line 433)
        objects_6099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 15), 'objects')
        # Assigning a type to the variable 'stypy_return_type' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'stypy_return_type', objects_6099)
        
        # ################# End of 'compile(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'compile' in the type store
        # Getting the type of 'stypy_return_type' (line 349)
        stypy_return_type_6100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6100)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'compile'
        return stypy_return_type_6100


    @norecursion
    def create_static_lib(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 441)
        None_6101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 38), 'None')
        int_6102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 33), 'int')
        # Getting the type of 'None' (line 443)
        None_6103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 39), 'None')
        defaults = [None_6101, int_6102, None_6103]
        # Create a new context for function 'create_static_lib'
        module_type_store = module_type_store.open_function_context('create_static_lib', 438, 4, False)
        # Assigning a type to the variable 'self' (line 439)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MSVCCompiler.create_static_lib.__dict__.__setitem__('stypy_localization', localization)
        MSVCCompiler.create_static_lib.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MSVCCompiler.create_static_lib.__dict__.__setitem__('stypy_type_store', module_type_store)
        MSVCCompiler.create_static_lib.__dict__.__setitem__('stypy_function_name', 'MSVCCompiler.create_static_lib')
        MSVCCompiler.create_static_lib.__dict__.__setitem__('stypy_param_names_list', ['objects', 'output_libname', 'output_dir', 'debug', 'target_lang'])
        MSVCCompiler.create_static_lib.__dict__.__setitem__('stypy_varargs_param_name', None)
        MSVCCompiler.create_static_lib.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MSVCCompiler.create_static_lib.__dict__.__setitem__('stypy_call_defaults', defaults)
        MSVCCompiler.create_static_lib.__dict__.__setitem__('stypy_call_varargs', varargs)
        MSVCCompiler.create_static_lib.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MSVCCompiler.create_static_lib.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MSVCCompiler.create_static_lib', ['objects', 'output_libname', 'output_dir', 'debug', 'target_lang'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_static_lib', localization, ['objects', 'output_libname', 'output_dir', 'debug', 'target_lang'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_static_lib(...)' code ##################

        
        
        # Getting the type of 'self' (line 445)
        self_6104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 15), 'self')
        # Obtaining the member 'initialized' of a type (line 445)
        initialized_6105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 15), self_6104, 'initialized')
        # Applying the 'not' unary operator (line 445)
        result_not__6106 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 11), 'not', initialized_6105)
        
        # Testing the type of an if condition (line 445)
        if_condition_6107 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 445, 8), result_not__6106)
        # Assigning a type to the variable 'if_condition_6107' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'if_condition_6107', if_condition_6107)
        # SSA begins for if statement (line 445)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to initialize(...): (line 445)
        # Processing the call keyword arguments (line 445)
        kwargs_6110 = {}
        # Getting the type of 'self' (line 445)
        self_6108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 33), 'self', False)
        # Obtaining the member 'initialize' of a type (line 445)
        initialize_6109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 33), self_6108, 'initialize')
        # Calling initialize(args, kwargs) (line 445)
        initialize_call_result_6111 = invoke(stypy.reporting.localization.Localization(__file__, 445, 33), initialize_6109, *[], **kwargs_6110)
        
        # SSA join for if statement (line 445)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 446):
        
        # Assigning a Subscript to a Name (line 446):
        
        # Obtaining the type of the subscript
        int_6112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 8), 'int')
        
        # Call to _fix_object_args(...): (line 446)
        # Processing the call arguments (line 446)
        # Getting the type of 'objects' (line 446)
        objects_6115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 55), 'objects', False)
        # Getting the type of 'output_dir' (line 446)
        output_dir_6116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 64), 'output_dir', False)
        # Processing the call keyword arguments (line 446)
        kwargs_6117 = {}
        # Getting the type of 'self' (line 446)
        self_6113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 32), 'self', False)
        # Obtaining the member '_fix_object_args' of a type (line 446)
        _fix_object_args_6114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 32), self_6113, '_fix_object_args')
        # Calling _fix_object_args(args, kwargs) (line 446)
        _fix_object_args_call_result_6118 = invoke(stypy.reporting.localization.Localization(__file__, 446, 32), _fix_object_args_6114, *[objects_6115, output_dir_6116], **kwargs_6117)
        
        # Obtaining the member '__getitem__' of a type (line 446)
        getitem___6119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 8), _fix_object_args_call_result_6118, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 446)
        subscript_call_result_6120 = invoke(stypy.reporting.localization.Localization(__file__, 446, 8), getitem___6119, int_6112)
        
        # Assigning a type to the variable 'tuple_var_assignment_5030' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'tuple_var_assignment_5030', subscript_call_result_6120)
        
        # Assigning a Subscript to a Name (line 446):
        
        # Obtaining the type of the subscript
        int_6121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 8), 'int')
        
        # Call to _fix_object_args(...): (line 446)
        # Processing the call arguments (line 446)
        # Getting the type of 'objects' (line 446)
        objects_6124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 55), 'objects', False)
        # Getting the type of 'output_dir' (line 446)
        output_dir_6125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 64), 'output_dir', False)
        # Processing the call keyword arguments (line 446)
        kwargs_6126 = {}
        # Getting the type of 'self' (line 446)
        self_6122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 32), 'self', False)
        # Obtaining the member '_fix_object_args' of a type (line 446)
        _fix_object_args_6123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 32), self_6122, '_fix_object_args')
        # Calling _fix_object_args(args, kwargs) (line 446)
        _fix_object_args_call_result_6127 = invoke(stypy.reporting.localization.Localization(__file__, 446, 32), _fix_object_args_6123, *[objects_6124, output_dir_6125], **kwargs_6126)
        
        # Obtaining the member '__getitem__' of a type (line 446)
        getitem___6128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 8), _fix_object_args_call_result_6127, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 446)
        subscript_call_result_6129 = invoke(stypy.reporting.localization.Localization(__file__, 446, 8), getitem___6128, int_6121)
        
        # Assigning a type to the variable 'tuple_var_assignment_5031' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'tuple_var_assignment_5031', subscript_call_result_6129)
        
        # Assigning a Name to a Name (line 446):
        # Getting the type of 'tuple_var_assignment_5030' (line 446)
        tuple_var_assignment_5030_6130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'tuple_var_assignment_5030')
        # Assigning a type to the variable 'objects' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 9), 'objects', tuple_var_assignment_5030_6130)
        
        # Assigning a Name to a Name (line 446):
        # Getting the type of 'tuple_var_assignment_5031' (line 446)
        tuple_var_assignment_5031_6131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'tuple_var_assignment_5031')
        # Assigning a type to the variable 'output_dir' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 18), 'output_dir', tuple_var_assignment_5031_6131)
        
        # Assigning a Call to a Name (line 447):
        
        # Assigning a Call to a Name (line 447):
        
        # Call to library_filename(...): (line 448)
        # Processing the call arguments (line 448)
        # Getting the type of 'output_libname' (line 448)
        output_libname_6134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 35), 'output_libname', False)
        # Processing the call keyword arguments (line 448)
        # Getting the type of 'output_dir' (line 448)
        output_dir_6135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 62), 'output_dir', False)
        keyword_6136 = output_dir_6135
        kwargs_6137 = {'output_dir': keyword_6136}
        # Getting the type of 'self' (line 448)
        self_6132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 12), 'self', False)
        # Obtaining the member 'library_filename' of a type (line 448)
        library_filename_6133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 12), self_6132, 'library_filename')
        # Calling library_filename(args, kwargs) (line 448)
        library_filename_call_result_6138 = invoke(stypy.reporting.localization.Localization(__file__, 448, 12), library_filename_6133, *[output_libname_6134], **kwargs_6137)
        
        # Assigning a type to the variable 'output_filename' (line 447)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'output_filename', library_filename_call_result_6138)
        
        
        # Call to _need_link(...): (line 450)
        # Processing the call arguments (line 450)
        # Getting the type of 'objects' (line 450)
        objects_6141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 28), 'objects', False)
        # Getting the type of 'output_filename' (line 450)
        output_filename_6142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 37), 'output_filename', False)
        # Processing the call keyword arguments (line 450)
        kwargs_6143 = {}
        # Getting the type of 'self' (line 450)
        self_6139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 11), 'self', False)
        # Obtaining the member '_need_link' of a type (line 450)
        _need_link_6140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 11), self_6139, '_need_link')
        # Calling _need_link(args, kwargs) (line 450)
        _need_link_call_result_6144 = invoke(stypy.reporting.localization.Localization(__file__, 450, 11), _need_link_6140, *[objects_6141, output_filename_6142], **kwargs_6143)
        
        # Testing the type of an if condition (line 450)
        if_condition_6145 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 450, 8), _need_link_call_result_6144)
        # Assigning a type to the variable 'if_condition_6145' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'if_condition_6145', if_condition_6145)
        # SSA begins for if statement (line 450)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 451):
        
        # Assigning a BinOp to a Name (line 451):
        # Getting the type of 'objects' (line 451)
        objects_6146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 23), 'objects')
        
        # Obtaining an instance of the builtin type 'list' (line 451)
        list_6147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 451)
        # Adding element type (line 451)
        str_6148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 34), 'str', '/OUT:')
        # Getting the type of 'output_filename' (line 451)
        output_filename_6149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 44), 'output_filename')
        # Applying the binary operator '+' (line 451)
        result_add_6150 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 34), '+', str_6148, output_filename_6149)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 33), list_6147, result_add_6150)
        
        # Applying the binary operator '+' (line 451)
        result_add_6151 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 23), '+', objects_6146, list_6147)
        
        # Assigning a type to the variable 'lib_args' (line 451)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'lib_args', result_add_6151)
        
        # Getting the type of 'debug' (line 452)
        debug_6152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 15), 'debug')
        # Testing the type of an if condition (line 452)
        if_condition_6153 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 452, 12), debug_6152)
        # Assigning a type to the variable 'if_condition_6153' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 12), 'if_condition_6153', if_condition_6153)
        # SSA begins for if statement (line 452)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA join for if statement (line 452)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 454)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to spawn(...): (line 455)
        # Processing the call arguments (line 455)
        
        # Obtaining an instance of the builtin type 'list' (line 455)
        list_6156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 455)
        # Adding element type (line 455)
        # Getting the type of 'self' (line 455)
        self_6157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 29), 'self', False)
        # Obtaining the member 'lib' of a type (line 455)
        lib_6158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 29), self_6157, 'lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 28), list_6156, lib_6158)
        
        # Getting the type of 'lib_args' (line 455)
        lib_args_6159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 41), 'lib_args', False)
        # Applying the binary operator '+' (line 455)
        result_add_6160 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 28), '+', list_6156, lib_args_6159)
        
        # Processing the call keyword arguments (line 455)
        kwargs_6161 = {}
        # Getting the type of 'self' (line 455)
        self_6154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 16), 'self', False)
        # Obtaining the member 'spawn' of a type (line 455)
        spawn_6155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 16), self_6154, 'spawn')
        # Calling spawn(args, kwargs) (line 455)
        spawn_call_result_6162 = invoke(stypy.reporting.localization.Localization(__file__, 455, 16), spawn_6155, *[result_add_6160], **kwargs_6161)
        
        # SSA branch for the except part of a try statement (line 454)
        # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 454)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsExecError' (line 456)
        DistutilsExecError_6163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 19), 'DistutilsExecError')
        # Assigning a type to the variable 'msg' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 12), 'msg', DistutilsExecError_6163)
        # Getting the type of 'LibError' (line 457)
        LibError_6164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 22), 'LibError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 457, 16), LibError_6164, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 454)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 450)
        module_type_store.open_ssa_branch('else')
        
        # Call to debug(...): (line 460)
        # Processing the call arguments (line 460)
        str_6167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 22), 'str', 'skipping %s (up-to-date)')
        # Getting the type of 'output_filename' (line 460)
        output_filename_6168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 50), 'output_filename', False)
        # Processing the call keyword arguments (line 460)
        kwargs_6169 = {}
        # Getting the type of 'log' (line 460)
        log_6165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 12), 'log', False)
        # Obtaining the member 'debug' of a type (line 460)
        debug_6166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 12), log_6165, 'debug')
        # Calling debug(args, kwargs) (line 460)
        debug_call_result_6170 = invoke(stypy.reporting.localization.Localization(__file__, 460, 12), debug_6166, *[str_6167, output_filename_6168], **kwargs_6169)
        
        # SSA join for if statement (line 450)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'create_static_lib(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_static_lib' in the type store
        # Getting the type of 'stypy_return_type' (line 438)
        stypy_return_type_6171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6171)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_static_lib'
        return stypy_return_type_6171


    @norecursion
    def link(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 468)
        None_6172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 25), 'None')
        # Getting the type of 'None' (line 469)
        None_6173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 24), 'None')
        # Getting the type of 'None' (line 470)
        None_6174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 27), 'None')
        # Getting the type of 'None' (line 471)
        None_6175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 35), 'None')
        # Getting the type of 'None' (line 472)
        None_6176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 29), 'None')
        int_6177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 20), 'int')
        # Getting the type of 'None' (line 474)
        None_6178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 28), 'None')
        # Getting the type of 'None' (line 475)
        None_6179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 29), 'None')
        # Getting the type of 'None' (line 476)
        None_6180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 25), 'None')
        # Getting the type of 'None' (line 477)
        None_6181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 26), 'None')
        defaults = [None_6172, None_6173, None_6174, None_6175, None_6176, int_6177, None_6178, None_6179, None_6180, None_6181]
        # Create a new context for function 'link'
        module_type_store = module_type_store.open_function_context('link', 464, 4, False)
        # Assigning a type to the variable 'self' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MSVCCompiler.link.__dict__.__setitem__('stypy_localization', localization)
        MSVCCompiler.link.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MSVCCompiler.link.__dict__.__setitem__('stypy_type_store', module_type_store)
        MSVCCompiler.link.__dict__.__setitem__('stypy_function_name', 'MSVCCompiler.link')
        MSVCCompiler.link.__dict__.__setitem__('stypy_param_names_list', ['target_desc', 'objects', 'output_filename', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'])
        MSVCCompiler.link.__dict__.__setitem__('stypy_varargs_param_name', None)
        MSVCCompiler.link.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MSVCCompiler.link.__dict__.__setitem__('stypy_call_defaults', defaults)
        MSVCCompiler.link.__dict__.__setitem__('stypy_call_varargs', varargs)
        MSVCCompiler.link.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MSVCCompiler.link.__dict__.__setitem__('stypy_declared_arg_number', 14)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MSVCCompiler.link', ['target_desc', 'objects', 'output_filename', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'link', localization, ['target_desc', 'objects', 'output_filename', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'link(...)' code ##################

        
        
        # Getting the type of 'self' (line 479)
        self_6182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 15), 'self')
        # Obtaining the member 'initialized' of a type (line 479)
        initialized_6183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 15), self_6182, 'initialized')
        # Applying the 'not' unary operator (line 479)
        result_not__6184 = python_operator(stypy.reporting.localization.Localization(__file__, 479, 11), 'not', initialized_6183)
        
        # Testing the type of an if condition (line 479)
        if_condition_6185 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 479, 8), result_not__6184)
        # Assigning a type to the variable 'if_condition_6185' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'if_condition_6185', if_condition_6185)
        # SSA begins for if statement (line 479)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to initialize(...): (line 479)
        # Processing the call keyword arguments (line 479)
        kwargs_6188 = {}
        # Getting the type of 'self' (line 479)
        self_6186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 33), 'self', False)
        # Obtaining the member 'initialize' of a type (line 479)
        initialize_6187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 33), self_6186, 'initialize')
        # Calling initialize(args, kwargs) (line 479)
        initialize_call_result_6189 = invoke(stypy.reporting.localization.Localization(__file__, 479, 33), initialize_6187, *[], **kwargs_6188)
        
        # SSA join for if statement (line 479)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 480):
        
        # Assigning a Subscript to a Name (line 480):
        
        # Obtaining the type of the subscript
        int_6190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 8), 'int')
        
        # Call to _fix_object_args(...): (line 480)
        # Processing the call arguments (line 480)
        # Getting the type of 'objects' (line 480)
        objects_6193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 55), 'objects', False)
        # Getting the type of 'output_dir' (line 480)
        output_dir_6194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 64), 'output_dir', False)
        # Processing the call keyword arguments (line 480)
        kwargs_6195 = {}
        # Getting the type of 'self' (line 480)
        self_6191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 32), 'self', False)
        # Obtaining the member '_fix_object_args' of a type (line 480)
        _fix_object_args_6192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 32), self_6191, '_fix_object_args')
        # Calling _fix_object_args(args, kwargs) (line 480)
        _fix_object_args_call_result_6196 = invoke(stypy.reporting.localization.Localization(__file__, 480, 32), _fix_object_args_6192, *[objects_6193, output_dir_6194], **kwargs_6195)
        
        # Obtaining the member '__getitem__' of a type (line 480)
        getitem___6197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 8), _fix_object_args_call_result_6196, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 480)
        subscript_call_result_6198 = invoke(stypy.reporting.localization.Localization(__file__, 480, 8), getitem___6197, int_6190)
        
        # Assigning a type to the variable 'tuple_var_assignment_5032' (line 480)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'tuple_var_assignment_5032', subscript_call_result_6198)
        
        # Assigning a Subscript to a Name (line 480):
        
        # Obtaining the type of the subscript
        int_6199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 8), 'int')
        
        # Call to _fix_object_args(...): (line 480)
        # Processing the call arguments (line 480)
        # Getting the type of 'objects' (line 480)
        objects_6202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 55), 'objects', False)
        # Getting the type of 'output_dir' (line 480)
        output_dir_6203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 64), 'output_dir', False)
        # Processing the call keyword arguments (line 480)
        kwargs_6204 = {}
        # Getting the type of 'self' (line 480)
        self_6200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 32), 'self', False)
        # Obtaining the member '_fix_object_args' of a type (line 480)
        _fix_object_args_6201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 32), self_6200, '_fix_object_args')
        # Calling _fix_object_args(args, kwargs) (line 480)
        _fix_object_args_call_result_6205 = invoke(stypy.reporting.localization.Localization(__file__, 480, 32), _fix_object_args_6201, *[objects_6202, output_dir_6203], **kwargs_6204)
        
        # Obtaining the member '__getitem__' of a type (line 480)
        getitem___6206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 8), _fix_object_args_call_result_6205, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 480)
        subscript_call_result_6207 = invoke(stypy.reporting.localization.Localization(__file__, 480, 8), getitem___6206, int_6199)
        
        # Assigning a type to the variable 'tuple_var_assignment_5033' (line 480)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'tuple_var_assignment_5033', subscript_call_result_6207)
        
        # Assigning a Name to a Name (line 480):
        # Getting the type of 'tuple_var_assignment_5032' (line 480)
        tuple_var_assignment_5032_6208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'tuple_var_assignment_5032')
        # Assigning a type to the variable 'objects' (line 480)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 9), 'objects', tuple_var_assignment_5032_6208)
        
        # Assigning a Name to a Name (line 480):
        # Getting the type of 'tuple_var_assignment_5033' (line 480)
        tuple_var_assignment_5033_6209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'tuple_var_assignment_5033')
        # Assigning a type to the variable 'output_dir' (line 480)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 18), 'output_dir', tuple_var_assignment_5033_6209)
        
        # Assigning a Call to a Tuple (line 481):
        
        # Assigning a Subscript to a Name (line 481):
        
        # Obtaining the type of the subscript
        int_6210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 8), 'int')
        
        # Call to _fix_lib_args(...): (line 482)
        # Processing the call arguments (line 482)
        # Getting the type of 'libraries' (line 482)
        libraries_6213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 32), 'libraries', False)
        # Getting the type of 'library_dirs' (line 482)
        library_dirs_6214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 43), 'library_dirs', False)
        # Getting the type of 'runtime_library_dirs' (line 482)
        runtime_library_dirs_6215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 57), 'runtime_library_dirs', False)
        # Processing the call keyword arguments (line 482)
        kwargs_6216 = {}
        # Getting the type of 'self' (line 482)
        self_6211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 12), 'self', False)
        # Obtaining the member '_fix_lib_args' of a type (line 482)
        _fix_lib_args_6212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 12), self_6211, '_fix_lib_args')
        # Calling _fix_lib_args(args, kwargs) (line 482)
        _fix_lib_args_call_result_6217 = invoke(stypy.reporting.localization.Localization(__file__, 482, 12), _fix_lib_args_6212, *[libraries_6213, library_dirs_6214, runtime_library_dirs_6215], **kwargs_6216)
        
        # Obtaining the member '__getitem__' of a type (line 481)
        getitem___6218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 8), _fix_lib_args_call_result_6217, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 481)
        subscript_call_result_6219 = invoke(stypy.reporting.localization.Localization(__file__, 481, 8), getitem___6218, int_6210)
        
        # Assigning a type to the variable 'tuple_var_assignment_5034' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'tuple_var_assignment_5034', subscript_call_result_6219)
        
        # Assigning a Subscript to a Name (line 481):
        
        # Obtaining the type of the subscript
        int_6220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 8), 'int')
        
        # Call to _fix_lib_args(...): (line 482)
        # Processing the call arguments (line 482)
        # Getting the type of 'libraries' (line 482)
        libraries_6223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 32), 'libraries', False)
        # Getting the type of 'library_dirs' (line 482)
        library_dirs_6224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 43), 'library_dirs', False)
        # Getting the type of 'runtime_library_dirs' (line 482)
        runtime_library_dirs_6225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 57), 'runtime_library_dirs', False)
        # Processing the call keyword arguments (line 482)
        kwargs_6226 = {}
        # Getting the type of 'self' (line 482)
        self_6221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 12), 'self', False)
        # Obtaining the member '_fix_lib_args' of a type (line 482)
        _fix_lib_args_6222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 12), self_6221, '_fix_lib_args')
        # Calling _fix_lib_args(args, kwargs) (line 482)
        _fix_lib_args_call_result_6227 = invoke(stypy.reporting.localization.Localization(__file__, 482, 12), _fix_lib_args_6222, *[libraries_6223, library_dirs_6224, runtime_library_dirs_6225], **kwargs_6226)
        
        # Obtaining the member '__getitem__' of a type (line 481)
        getitem___6228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 8), _fix_lib_args_call_result_6227, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 481)
        subscript_call_result_6229 = invoke(stypy.reporting.localization.Localization(__file__, 481, 8), getitem___6228, int_6220)
        
        # Assigning a type to the variable 'tuple_var_assignment_5035' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'tuple_var_assignment_5035', subscript_call_result_6229)
        
        # Assigning a Subscript to a Name (line 481):
        
        # Obtaining the type of the subscript
        int_6230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 8), 'int')
        
        # Call to _fix_lib_args(...): (line 482)
        # Processing the call arguments (line 482)
        # Getting the type of 'libraries' (line 482)
        libraries_6233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 32), 'libraries', False)
        # Getting the type of 'library_dirs' (line 482)
        library_dirs_6234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 43), 'library_dirs', False)
        # Getting the type of 'runtime_library_dirs' (line 482)
        runtime_library_dirs_6235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 57), 'runtime_library_dirs', False)
        # Processing the call keyword arguments (line 482)
        kwargs_6236 = {}
        # Getting the type of 'self' (line 482)
        self_6231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 12), 'self', False)
        # Obtaining the member '_fix_lib_args' of a type (line 482)
        _fix_lib_args_6232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 12), self_6231, '_fix_lib_args')
        # Calling _fix_lib_args(args, kwargs) (line 482)
        _fix_lib_args_call_result_6237 = invoke(stypy.reporting.localization.Localization(__file__, 482, 12), _fix_lib_args_6232, *[libraries_6233, library_dirs_6234, runtime_library_dirs_6235], **kwargs_6236)
        
        # Obtaining the member '__getitem__' of a type (line 481)
        getitem___6238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 8), _fix_lib_args_call_result_6237, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 481)
        subscript_call_result_6239 = invoke(stypy.reporting.localization.Localization(__file__, 481, 8), getitem___6238, int_6230)
        
        # Assigning a type to the variable 'tuple_var_assignment_5036' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'tuple_var_assignment_5036', subscript_call_result_6239)
        
        # Assigning a Name to a Name (line 481):
        # Getting the type of 'tuple_var_assignment_5034' (line 481)
        tuple_var_assignment_5034_6240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'tuple_var_assignment_5034')
        # Assigning a type to the variable 'libraries' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 9), 'libraries', tuple_var_assignment_5034_6240)
        
        # Assigning a Name to a Name (line 481):
        # Getting the type of 'tuple_var_assignment_5035' (line 481)
        tuple_var_assignment_5035_6241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'tuple_var_assignment_5035')
        # Assigning a type to the variable 'library_dirs' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 20), 'library_dirs', tuple_var_assignment_5035_6241)
        
        # Assigning a Name to a Name (line 481):
        # Getting the type of 'tuple_var_assignment_5036' (line 481)
        tuple_var_assignment_5036_6242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'tuple_var_assignment_5036')
        # Assigning a type to the variable 'runtime_library_dirs' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 34), 'runtime_library_dirs', tuple_var_assignment_5036_6242)
        
        # Getting the type of 'runtime_library_dirs' (line 484)
        runtime_library_dirs_6243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 11), 'runtime_library_dirs')
        # Testing the type of an if condition (line 484)
        if_condition_6244 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 484, 8), runtime_library_dirs_6243)
        # Assigning a type to the variable 'if_condition_6244' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'if_condition_6244', if_condition_6244)
        # SSA begins for if statement (line 484)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 485)
        # Processing the call arguments (line 485)
        str_6247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 23), 'str', "I don't know what to do with 'runtime_library_dirs': ")
        
        # Call to str(...): (line 486)
        # Processing the call arguments (line 486)
        # Getting the type of 'runtime_library_dirs' (line 486)
        runtime_library_dirs_6249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 30), 'runtime_library_dirs', False)
        # Processing the call keyword arguments (line 486)
        kwargs_6250 = {}
        # Getting the type of 'str' (line 486)
        str_6248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 25), 'str', False)
        # Calling str(args, kwargs) (line 486)
        str_call_result_6251 = invoke(stypy.reporting.localization.Localization(__file__, 486, 25), str_6248, *[runtime_library_dirs_6249], **kwargs_6250)
        
        # Applying the binary operator '+' (line 485)
        result_add_6252 = python_operator(stypy.reporting.localization.Localization(__file__, 485, 23), '+', str_6247, str_call_result_6251)
        
        # Processing the call keyword arguments (line 485)
        kwargs_6253 = {}
        # Getting the type of 'self' (line 485)
        self_6245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 12), 'self', False)
        # Obtaining the member 'warn' of a type (line 485)
        warn_6246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 12), self_6245, 'warn')
        # Calling warn(args, kwargs) (line 485)
        warn_call_result_6254 = invoke(stypy.reporting.localization.Localization(__file__, 485, 12), warn_6246, *[result_add_6252], **kwargs_6253)
        
        # SSA join for if statement (line 484)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 488):
        
        # Assigning a Call to a Name (line 488):
        
        # Call to gen_lib_options(...): (line 488)
        # Processing the call arguments (line 488)
        # Getting the type of 'self' (line 488)
        self_6256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 36), 'self', False)
        # Getting the type of 'library_dirs' (line 489)
        library_dirs_6257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 36), 'library_dirs', False)
        # Getting the type of 'runtime_library_dirs' (line 489)
        runtime_library_dirs_6258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 50), 'runtime_library_dirs', False)
        # Getting the type of 'libraries' (line 490)
        libraries_6259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 36), 'libraries', False)
        # Processing the call keyword arguments (line 488)
        kwargs_6260 = {}
        # Getting the type of 'gen_lib_options' (line 488)
        gen_lib_options_6255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 19), 'gen_lib_options', False)
        # Calling gen_lib_options(args, kwargs) (line 488)
        gen_lib_options_call_result_6261 = invoke(stypy.reporting.localization.Localization(__file__, 488, 19), gen_lib_options_6255, *[self_6256, library_dirs_6257, runtime_library_dirs_6258, libraries_6259], **kwargs_6260)
        
        # Assigning a type to the variable 'lib_opts' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'lib_opts', gen_lib_options_call_result_6261)
        
        # Type idiom detected: calculating its left and rigth part (line 491)
        # Getting the type of 'output_dir' (line 491)
        output_dir_6262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'output_dir')
        # Getting the type of 'None' (line 491)
        None_6263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 29), 'None')
        
        (may_be_6264, more_types_in_union_6265) = may_not_be_none(output_dir_6262, None_6263)

        if may_be_6264:

            if more_types_in_union_6265:
                # Runtime conditional SSA (line 491)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 492):
            
            # Assigning a Call to a Name (line 492):
            
            # Call to join(...): (line 492)
            # Processing the call arguments (line 492)
            # Getting the type of 'output_dir' (line 492)
            output_dir_6269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 44), 'output_dir', False)
            # Getting the type of 'output_filename' (line 492)
            output_filename_6270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 56), 'output_filename', False)
            # Processing the call keyword arguments (line 492)
            kwargs_6271 = {}
            # Getting the type of 'os' (line 492)
            os_6266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 30), 'os', False)
            # Obtaining the member 'path' of a type (line 492)
            path_6267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 30), os_6266, 'path')
            # Obtaining the member 'join' of a type (line 492)
            join_6268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 30), path_6267, 'join')
            # Calling join(args, kwargs) (line 492)
            join_call_result_6272 = invoke(stypy.reporting.localization.Localization(__file__, 492, 30), join_6268, *[output_dir_6269, output_filename_6270], **kwargs_6271)
            
            # Assigning a type to the variable 'output_filename' (line 492)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 12), 'output_filename', join_call_result_6272)

            if more_types_in_union_6265:
                # SSA join for if statement (line 491)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Call to _need_link(...): (line 494)
        # Processing the call arguments (line 494)
        # Getting the type of 'objects' (line 494)
        objects_6275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 28), 'objects', False)
        # Getting the type of 'output_filename' (line 494)
        output_filename_6276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 37), 'output_filename', False)
        # Processing the call keyword arguments (line 494)
        kwargs_6277 = {}
        # Getting the type of 'self' (line 494)
        self_6273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 11), 'self', False)
        # Obtaining the member '_need_link' of a type (line 494)
        _need_link_6274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 11), self_6273, '_need_link')
        # Calling _need_link(args, kwargs) (line 494)
        _need_link_call_result_6278 = invoke(stypy.reporting.localization.Localization(__file__, 494, 11), _need_link_6274, *[objects_6275, output_filename_6276], **kwargs_6277)
        
        # Testing the type of an if condition (line 494)
        if_condition_6279 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 494, 8), _need_link_call_result_6278)
        # Assigning a type to the variable 'if_condition_6279' (line 494)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'if_condition_6279', if_condition_6279)
        # SSA begins for if statement (line 494)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'target_desc' (line 496)
        target_desc_6280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 15), 'target_desc')
        # Getting the type of 'CCompiler' (line 496)
        CCompiler_6281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 30), 'CCompiler')
        # Obtaining the member 'EXECUTABLE' of a type (line 496)
        EXECUTABLE_6282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 30), CCompiler_6281, 'EXECUTABLE')
        # Applying the binary operator '==' (line 496)
        result_eq_6283 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 15), '==', target_desc_6280, EXECUTABLE_6282)
        
        # Testing the type of an if condition (line 496)
        if_condition_6284 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 496, 12), result_eq_6283)
        # Assigning a type to the variable 'if_condition_6284' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'if_condition_6284', if_condition_6284)
        # SSA begins for if statement (line 496)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'debug' (line 497)
        debug_6285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 19), 'debug')
        # Testing the type of an if condition (line 497)
        if_condition_6286 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 497, 16), debug_6285)
        # Assigning a type to the variable 'if_condition_6286' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 16), 'if_condition_6286', if_condition_6286)
        # SSA begins for if statement (line 497)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 498):
        
        # Assigning a Subscript to a Name (line 498):
        
        # Obtaining the type of the subscript
        int_6287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 56), 'int')
        slice_6288 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 498, 30), int_6287, None, None)
        # Getting the type of 'self' (line 498)
        self_6289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 30), 'self')
        # Obtaining the member 'ldflags_shared_debug' of a type (line 498)
        ldflags_shared_debug_6290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 30), self_6289, 'ldflags_shared_debug')
        # Obtaining the member '__getitem__' of a type (line 498)
        getitem___6291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 30), ldflags_shared_debug_6290, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 498)
        subscript_call_result_6292 = invoke(stypy.reporting.localization.Localization(__file__, 498, 30), getitem___6291, slice_6288)
        
        # Assigning a type to the variable 'ldflags' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 20), 'ldflags', subscript_call_result_6292)
        # SSA branch for the else part of an if statement (line 497)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Name (line 500):
        
        # Assigning a Subscript to a Name (line 500):
        
        # Obtaining the type of the subscript
        int_6293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 50), 'int')
        slice_6294 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 500, 30), int_6293, None, None)
        # Getting the type of 'self' (line 500)
        self_6295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 30), 'self')
        # Obtaining the member 'ldflags_shared' of a type (line 500)
        ldflags_shared_6296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 30), self_6295, 'ldflags_shared')
        # Obtaining the member '__getitem__' of a type (line 500)
        getitem___6297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 30), ldflags_shared_6296, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 500)
        subscript_call_result_6298 = invoke(stypy.reporting.localization.Localization(__file__, 500, 30), getitem___6297, slice_6294)
        
        # Assigning a type to the variable 'ldflags' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 20), 'ldflags', subscript_call_result_6298)
        # SSA join for if statement (line 497)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 496)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'debug' (line 502)
        debug_6299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 19), 'debug')
        # Testing the type of an if condition (line 502)
        if_condition_6300 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 502, 16), debug_6299)
        # Assigning a type to the variable 'if_condition_6300' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 16), 'if_condition_6300', if_condition_6300)
        # SSA begins for if statement (line 502)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 503):
        
        # Assigning a Attribute to a Name (line 503):
        # Getting the type of 'self' (line 503)
        self_6301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 30), 'self')
        # Obtaining the member 'ldflags_shared_debug' of a type (line 503)
        ldflags_shared_debug_6302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 30), self_6301, 'ldflags_shared_debug')
        # Assigning a type to the variable 'ldflags' (line 503)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 20), 'ldflags', ldflags_shared_debug_6302)
        # SSA branch for the else part of an if statement (line 502)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 505):
        
        # Assigning a Attribute to a Name (line 505):
        # Getting the type of 'self' (line 505)
        self_6303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 30), 'self')
        # Obtaining the member 'ldflags_shared' of a type (line 505)
        ldflags_shared_6304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 30), self_6303, 'ldflags_shared')
        # Assigning a type to the variable 'ldflags' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 20), 'ldflags', ldflags_shared_6304)
        # SSA join for if statement (line 502)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 496)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 507):
        
        # Assigning a List to a Name (line 507):
        
        # Obtaining an instance of the builtin type 'list' (line 507)
        list_6305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 507)
        
        # Assigning a type to the variable 'export_opts' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 12), 'export_opts', list_6305)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'export_symbols' (line 508)
        export_symbols_6306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 24), 'export_symbols')
        
        # Obtaining an instance of the builtin type 'list' (line 508)
        list_6307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 508)
        
        # Applying the binary operator 'or' (line 508)
        result_or_keyword_6308 = python_operator(stypy.reporting.localization.Localization(__file__, 508, 24), 'or', export_symbols_6306, list_6307)
        
        # Testing the type of a for loop iterable (line 508)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 508, 12), result_or_keyword_6308)
        # Getting the type of the for loop variable (line 508)
        for_loop_var_6309 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 508, 12), result_or_keyword_6308)
        # Assigning a type to the variable 'sym' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'sym', for_loop_var_6309)
        # SSA begins for a for statement (line 508)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 509)
        # Processing the call arguments (line 509)
        str_6312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 35), 'str', '/EXPORT:')
        # Getting the type of 'sym' (line 509)
        sym_6313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 48), 'sym', False)
        # Applying the binary operator '+' (line 509)
        result_add_6314 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 35), '+', str_6312, sym_6313)
        
        # Processing the call keyword arguments (line 509)
        kwargs_6315 = {}
        # Getting the type of 'export_opts' (line 509)
        export_opts_6310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 16), 'export_opts', False)
        # Obtaining the member 'append' of a type (line 509)
        append_6311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 16), export_opts_6310, 'append')
        # Calling append(args, kwargs) (line 509)
        append_call_result_6316 = invoke(stypy.reporting.localization.Localization(__file__, 509, 16), append_6311, *[result_add_6314], **kwargs_6315)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 511):
        
        # Assigning a BinOp to a Name (line 511):
        # Getting the type of 'ldflags' (line 511)
        ldflags_6317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 23), 'ldflags')
        # Getting the type of 'lib_opts' (line 511)
        lib_opts_6318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 33), 'lib_opts')
        # Applying the binary operator '+' (line 511)
        result_add_6319 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 23), '+', ldflags_6317, lib_opts_6318)
        
        # Getting the type of 'export_opts' (line 511)
        export_opts_6320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 44), 'export_opts')
        # Applying the binary operator '+' (line 511)
        result_add_6321 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 42), '+', result_add_6319, export_opts_6320)
        
        # Getting the type of 'objects' (line 512)
        objects_6322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 23), 'objects')
        # Applying the binary operator '+' (line 511)
        result_add_6323 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 56), '+', result_add_6321, objects_6322)
        
        
        # Obtaining an instance of the builtin type 'list' (line 512)
        list_6324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 512)
        # Adding element type (line 512)
        str_6325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 34), 'str', '/OUT:')
        # Getting the type of 'output_filename' (line 512)
        output_filename_6326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 44), 'output_filename')
        # Applying the binary operator '+' (line 512)
        result_add_6327 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 34), '+', str_6325, output_filename_6326)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 33), list_6324, result_add_6327)
        
        # Applying the binary operator '+' (line 512)
        result_add_6328 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 31), '+', result_add_6323, list_6324)
        
        # Assigning a type to the variable 'ld_args' (line 511)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 12), 'ld_args', result_add_6328)
        
        # Type idiom detected: calculating its left and rigth part (line 519)
        # Getting the type of 'export_symbols' (line 519)
        export_symbols_6329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 12), 'export_symbols')
        # Getting the type of 'None' (line 519)
        None_6330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 37), 'None')
        
        (may_be_6331, more_types_in_union_6332) = may_not_be_none(export_symbols_6329, None_6330)

        if may_be_6331:

            if more_types_in_union_6332:
                # Runtime conditional SSA (line 519)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Tuple (line 520):
            
            # Assigning a Subscript to a Name (line 520):
            
            # Obtaining the type of the subscript
            int_6333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 16), 'int')
            
            # Call to splitext(...): (line 520)
            # Processing the call arguments (line 520)
            
            # Call to basename(...): (line 521)
            # Processing the call arguments (line 521)
            # Getting the type of 'output_filename' (line 521)
            output_filename_6340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 37), 'output_filename', False)
            # Processing the call keyword arguments (line 521)
            kwargs_6341 = {}
            # Getting the type of 'os' (line 521)
            os_6337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 20), 'os', False)
            # Obtaining the member 'path' of a type (line 521)
            path_6338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 20), os_6337, 'path')
            # Obtaining the member 'basename' of a type (line 521)
            basename_6339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 20), path_6338, 'basename')
            # Calling basename(args, kwargs) (line 521)
            basename_call_result_6342 = invoke(stypy.reporting.localization.Localization(__file__, 521, 20), basename_6339, *[output_filename_6340], **kwargs_6341)
            
            # Processing the call keyword arguments (line 520)
            kwargs_6343 = {}
            # Getting the type of 'os' (line 520)
            os_6334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 38), 'os', False)
            # Obtaining the member 'path' of a type (line 520)
            path_6335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 38), os_6334, 'path')
            # Obtaining the member 'splitext' of a type (line 520)
            splitext_6336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 38), path_6335, 'splitext')
            # Calling splitext(args, kwargs) (line 520)
            splitext_call_result_6344 = invoke(stypy.reporting.localization.Localization(__file__, 520, 38), splitext_6336, *[basename_call_result_6342], **kwargs_6343)
            
            # Obtaining the member '__getitem__' of a type (line 520)
            getitem___6345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 16), splitext_call_result_6344, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 520)
            subscript_call_result_6346 = invoke(stypy.reporting.localization.Localization(__file__, 520, 16), getitem___6345, int_6333)
            
            # Assigning a type to the variable 'tuple_var_assignment_5037' (line 520)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 16), 'tuple_var_assignment_5037', subscript_call_result_6346)
            
            # Assigning a Subscript to a Name (line 520):
            
            # Obtaining the type of the subscript
            int_6347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 16), 'int')
            
            # Call to splitext(...): (line 520)
            # Processing the call arguments (line 520)
            
            # Call to basename(...): (line 521)
            # Processing the call arguments (line 521)
            # Getting the type of 'output_filename' (line 521)
            output_filename_6354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 37), 'output_filename', False)
            # Processing the call keyword arguments (line 521)
            kwargs_6355 = {}
            # Getting the type of 'os' (line 521)
            os_6351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 20), 'os', False)
            # Obtaining the member 'path' of a type (line 521)
            path_6352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 20), os_6351, 'path')
            # Obtaining the member 'basename' of a type (line 521)
            basename_6353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 20), path_6352, 'basename')
            # Calling basename(args, kwargs) (line 521)
            basename_call_result_6356 = invoke(stypy.reporting.localization.Localization(__file__, 521, 20), basename_6353, *[output_filename_6354], **kwargs_6355)
            
            # Processing the call keyword arguments (line 520)
            kwargs_6357 = {}
            # Getting the type of 'os' (line 520)
            os_6348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 38), 'os', False)
            # Obtaining the member 'path' of a type (line 520)
            path_6349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 38), os_6348, 'path')
            # Obtaining the member 'splitext' of a type (line 520)
            splitext_6350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 38), path_6349, 'splitext')
            # Calling splitext(args, kwargs) (line 520)
            splitext_call_result_6358 = invoke(stypy.reporting.localization.Localization(__file__, 520, 38), splitext_6350, *[basename_call_result_6356], **kwargs_6357)
            
            # Obtaining the member '__getitem__' of a type (line 520)
            getitem___6359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 16), splitext_call_result_6358, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 520)
            subscript_call_result_6360 = invoke(stypy.reporting.localization.Localization(__file__, 520, 16), getitem___6359, int_6347)
            
            # Assigning a type to the variable 'tuple_var_assignment_5038' (line 520)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 16), 'tuple_var_assignment_5038', subscript_call_result_6360)
            
            # Assigning a Name to a Name (line 520):
            # Getting the type of 'tuple_var_assignment_5037' (line 520)
            tuple_var_assignment_5037_6361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 16), 'tuple_var_assignment_5037')
            # Assigning a type to the variable 'dll_name' (line 520)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 17), 'dll_name', tuple_var_assignment_5037_6361)
            
            # Assigning a Name to a Name (line 520):
            # Getting the type of 'tuple_var_assignment_5038' (line 520)
            tuple_var_assignment_5038_6362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 16), 'tuple_var_assignment_5038')
            # Assigning a type to the variable 'dll_ext' (line 520)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 27), 'dll_ext', tuple_var_assignment_5038_6362)
            
            # Assigning a Call to a Name (line 522):
            
            # Assigning a Call to a Name (line 522):
            
            # Call to join(...): (line 522)
            # Processing the call arguments (line 522)
            
            # Call to dirname(...): (line 523)
            # Processing the call arguments (line 523)
            
            # Obtaining the type of the subscript
            int_6369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 44), 'int')
            # Getting the type of 'objects' (line 523)
            objects_6370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 36), 'objects', False)
            # Obtaining the member '__getitem__' of a type (line 523)
            getitem___6371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 36), objects_6370, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 523)
            subscript_call_result_6372 = invoke(stypy.reporting.localization.Localization(__file__, 523, 36), getitem___6371, int_6369)
            
            # Processing the call keyword arguments (line 523)
            kwargs_6373 = {}
            # Getting the type of 'os' (line 523)
            os_6366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 20), 'os', False)
            # Obtaining the member 'path' of a type (line 523)
            path_6367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 20), os_6366, 'path')
            # Obtaining the member 'dirname' of a type (line 523)
            dirname_6368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 20), path_6367, 'dirname')
            # Calling dirname(args, kwargs) (line 523)
            dirname_call_result_6374 = invoke(stypy.reporting.localization.Localization(__file__, 523, 20), dirname_6368, *[subscript_call_result_6372], **kwargs_6373)
            
            
            # Call to library_filename(...): (line 524)
            # Processing the call arguments (line 524)
            # Getting the type of 'dll_name' (line 524)
            dll_name_6377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 42), 'dll_name', False)
            # Processing the call keyword arguments (line 524)
            kwargs_6378 = {}
            # Getting the type of 'self' (line 524)
            self_6375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 20), 'self', False)
            # Obtaining the member 'library_filename' of a type (line 524)
            library_filename_6376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 20), self_6375, 'library_filename')
            # Calling library_filename(args, kwargs) (line 524)
            library_filename_call_result_6379 = invoke(stypy.reporting.localization.Localization(__file__, 524, 20), library_filename_6376, *[dll_name_6377], **kwargs_6378)
            
            # Processing the call keyword arguments (line 522)
            kwargs_6380 = {}
            # Getting the type of 'os' (line 522)
            os_6363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 30), 'os', False)
            # Obtaining the member 'path' of a type (line 522)
            path_6364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 30), os_6363, 'path')
            # Obtaining the member 'join' of a type (line 522)
            join_6365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 30), path_6364, 'join')
            # Calling join(args, kwargs) (line 522)
            join_call_result_6381 = invoke(stypy.reporting.localization.Localization(__file__, 522, 30), join_6365, *[dirname_call_result_6374, library_filename_call_result_6379], **kwargs_6380)
            
            # Assigning a type to the variable 'implib_file' (line 522)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 16), 'implib_file', join_call_result_6381)
            
            # Call to append(...): (line 525)
            # Processing the call arguments (line 525)
            str_6384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 32), 'str', '/IMPLIB:')
            # Getting the type of 'implib_file' (line 525)
            implib_file_6385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 45), 'implib_file', False)
            # Applying the binary operator '+' (line 525)
            result_add_6386 = python_operator(stypy.reporting.localization.Localization(__file__, 525, 32), '+', str_6384, implib_file_6385)
            
            # Processing the call keyword arguments (line 525)
            kwargs_6387 = {}
            # Getting the type of 'ld_args' (line 525)
            ld_args_6382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 16), 'ld_args', False)
            # Obtaining the member 'append' of a type (line 525)
            append_6383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 16), ld_args_6382, 'append')
            # Calling append(args, kwargs) (line 525)
            append_call_result_6388 = invoke(stypy.reporting.localization.Localization(__file__, 525, 16), append_6383, *[result_add_6386], **kwargs_6387)
            

            if more_types_in_union_6332:
                # SSA join for if statement (line 519)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'extra_preargs' (line 527)
        extra_preargs_6389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 15), 'extra_preargs')
        # Testing the type of an if condition (line 527)
        if_condition_6390 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 527, 12), extra_preargs_6389)
        # Assigning a type to the variable 'if_condition_6390' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 12), 'if_condition_6390', if_condition_6390)
        # SSA begins for if statement (line 527)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 528):
        
        # Assigning a Name to a Subscript (line 528):
        # Getting the type of 'extra_preargs' (line 528)
        extra_preargs_6391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 30), 'extra_preargs')
        # Getting the type of 'ld_args' (line 528)
        ld_args_6392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 16), 'ld_args')
        int_6393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 25), 'int')
        slice_6394 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 528, 16), None, int_6393, None)
        # Storing an element on a container (line 528)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 528, 16), ld_args_6392, (slice_6394, extra_preargs_6391))
        # SSA join for if statement (line 527)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'extra_postargs' (line 529)
        extra_postargs_6395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 15), 'extra_postargs')
        # Testing the type of an if condition (line 529)
        if_condition_6396 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 529, 12), extra_postargs_6395)
        # Assigning a type to the variable 'if_condition_6396' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'if_condition_6396', if_condition_6396)
        # SSA begins for if statement (line 529)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 530)
        # Processing the call arguments (line 530)
        # Getting the type of 'extra_postargs' (line 530)
        extra_postargs_6399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 31), 'extra_postargs', False)
        # Processing the call keyword arguments (line 530)
        kwargs_6400 = {}
        # Getting the type of 'ld_args' (line 530)
        ld_args_6397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 16), 'ld_args', False)
        # Obtaining the member 'extend' of a type (line 530)
        extend_6398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 16), ld_args_6397, 'extend')
        # Calling extend(args, kwargs) (line 530)
        extend_call_result_6401 = invoke(stypy.reporting.localization.Localization(__file__, 530, 16), extend_6398, *[extra_postargs_6399], **kwargs_6400)
        
        # SSA join for if statement (line 529)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to mkpath(...): (line 532)
        # Processing the call arguments (line 532)
        
        # Call to dirname(...): (line 532)
        # Processing the call arguments (line 532)
        # Getting the type of 'output_filename' (line 532)
        output_filename_6407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 42), 'output_filename', False)
        # Processing the call keyword arguments (line 532)
        kwargs_6408 = {}
        # Getting the type of 'os' (line 532)
        os_6404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 532)
        path_6405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 25), os_6404, 'path')
        # Obtaining the member 'dirname' of a type (line 532)
        dirname_6406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 25), path_6405, 'dirname')
        # Calling dirname(args, kwargs) (line 532)
        dirname_call_result_6409 = invoke(stypy.reporting.localization.Localization(__file__, 532, 25), dirname_6406, *[output_filename_6407], **kwargs_6408)
        
        # Processing the call keyword arguments (line 532)
        kwargs_6410 = {}
        # Getting the type of 'self' (line 532)
        self_6402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 12), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 532)
        mkpath_6403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 12), self_6402, 'mkpath')
        # Calling mkpath(args, kwargs) (line 532)
        mkpath_call_result_6411 = invoke(stypy.reporting.localization.Localization(__file__, 532, 12), mkpath_6403, *[dirname_call_result_6409], **kwargs_6410)
        
        
        
        # SSA begins for try-except statement (line 533)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to spawn(...): (line 534)
        # Processing the call arguments (line 534)
        
        # Obtaining an instance of the builtin type 'list' (line 534)
        list_6414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 534)
        # Adding element type (line 534)
        # Getting the type of 'self' (line 534)
        self_6415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 29), 'self', False)
        # Obtaining the member 'linker' of a type (line 534)
        linker_6416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 29), self_6415, 'linker')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 534, 28), list_6414, linker_6416)
        
        # Getting the type of 'ld_args' (line 534)
        ld_args_6417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 44), 'ld_args', False)
        # Applying the binary operator '+' (line 534)
        result_add_6418 = python_operator(stypy.reporting.localization.Localization(__file__, 534, 28), '+', list_6414, ld_args_6417)
        
        # Processing the call keyword arguments (line 534)
        kwargs_6419 = {}
        # Getting the type of 'self' (line 534)
        self_6412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 16), 'self', False)
        # Obtaining the member 'spawn' of a type (line 534)
        spawn_6413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 16), self_6412, 'spawn')
        # Calling spawn(args, kwargs) (line 534)
        spawn_call_result_6420 = invoke(stypy.reporting.localization.Localization(__file__, 534, 16), spawn_6413, *[result_add_6418], **kwargs_6419)
        
        # SSA branch for the except part of a try statement (line 533)
        # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 533)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsExecError' (line 535)
        DistutilsExecError_6421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 19), 'DistutilsExecError')
        # Assigning a type to the variable 'msg' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 12), 'msg', DistutilsExecError_6421)
        # Getting the type of 'LinkError' (line 536)
        LinkError_6422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 22), 'LinkError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 536, 16), LinkError_6422, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 533)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 494)
        module_type_store.open_ssa_branch('else')
        
        # Call to debug(...): (line 539)
        # Processing the call arguments (line 539)
        str_6425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 22), 'str', 'skipping %s (up-to-date)')
        # Getting the type of 'output_filename' (line 539)
        output_filename_6426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 50), 'output_filename', False)
        # Processing the call keyword arguments (line 539)
        kwargs_6427 = {}
        # Getting the type of 'log' (line 539)
        log_6423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 12), 'log', False)
        # Obtaining the member 'debug' of a type (line 539)
        debug_6424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 12), log_6423, 'debug')
        # Calling debug(args, kwargs) (line 539)
        debug_call_result_6428 = invoke(stypy.reporting.localization.Localization(__file__, 539, 12), debug_6424, *[str_6425, output_filename_6426], **kwargs_6427)
        
        # SSA join for if statement (line 494)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'link(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'link' in the type store
        # Getting the type of 'stypy_return_type' (line 464)
        stypy_return_type_6429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6429)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'link'
        return stypy_return_type_6429


    @norecursion
    def library_dir_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'library_dir_option'
        module_type_store = module_type_store.open_function_context('library_dir_option', 548, 4, False)
        # Assigning a type to the variable 'self' (line 549)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MSVCCompiler.library_dir_option.__dict__.__setitem__('stypy_localization', localization)
        MSVCCompiler.library_dir_option.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MSVCCompiler.library_dir_option.__dict__.__setitem__('stypy_type_store', module_type_store)
        MSVCCompiler.library_dir_option.__dict__.__setitem__('stypy_function_name', 'MSVCCompiler.library_dir_option')
        MSVCCompiler.library_dir_option.__dict__.__setitem__('stypy_param_names_list', ['dir'])
        MSVCCompiler.library_dir_option.__dict__.__setitem__('stypy_varargs_param_name', None)
        MSVCCompiler.library_dir_option.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MSVCCompiler.library_dir_option.__dict__.__setitem__('stypy_call_defaults', defaults)
        MSVCCompiler.library_dir_option.__dict__.__setitem__('stypy_call_varargs', varargs)
        MSVCCompiler.library_dir_option.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MSVCCompiler.library_dir_option.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MSVCCompiler.library_dir_option', ['dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'library_dir_option', localization, ['dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'library_dir_option(...)' code ##################

        str_6430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 15), 'str', '/LIBPATH:')
        # Getting the type of 'dir' (line 549)
        dir_6431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 29), 'dir')
        # Applying the binary operator '+' (line 549)
        result_add_6432 = python_operator(stypy.reporting.localization.Localization(__file__, 549, 15), '+', str_6430, dir_6431)
        
        # Assigning a type to the variable 'stypy_return_type' (line 549)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 8), 'stypy_return_type', result_add_6432)
        
        # ################# End of 'library_dir_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'library_dir_option' in the type store
        # Getting the type of 'stypy_return_type' (line 548)
        stypy_return_type_6433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6433)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'library_dir_option'
        return stypy_return_type_6433


    @norecursion
    def runtime_library_dir_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'runtime_library_dir_option'
        module_type_store = module_type_store.open_function_context('runtime_library_dir_option', 551, 4, False)
        # Assigning a type to the variable 'self' (line 552)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MSVCCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_localization', localization)
        MSVCCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MSVCCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_type_store', module_type_store)
        MSVCCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_function_name', 'MSVCCompiler.runtime_library_dir_option')
        MSVCCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_param_names_list', ['dir'])
        MSVCCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_varargs_param_name', None)
        MSVCCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MSVCCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_call_defaults', defaults)
        MSVCCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_call_varargs', varargs)
        MSVCCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MSVCCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MSVCCompiler.runtime_library_dir_option', ['dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'runtime_library_dir_option', localization, ['dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'runtime_library_dir_option(...)' code ##################

        # Getting the type of 'DistutilsPlatformError' (line 552)
        DistutilsPlatformError_6434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 14), 'DistutilsPlatformError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 552, 8), DistutilsPlatformError_6434, 'raise parameter', BaseException)
        
        # ################# End of 'runtime_library_dir_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'runtime_library_dir_option' in the type store
        # Getting the type of 'stypy_return_type' (line 551)
        stypy_return_type_6435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6435)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'runtime_library_dir_option'
        return stypy_return_type_6435


    @norecursion
    def library_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'library_option'
        module_type_store = module_type_store.open_function_context('library_option', 555, 4, False)
        # Assigning a type to the variable 'self' (line 556)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MSVCCompiler.library_option.__dict__.__setitem__('stypy_localization', localization)
        MSVCCompiler.library_option.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MSVCCompiler.library_option.__dict__.__setitem__('stypy_type_store', module_type_store)
        MSVCCompiler.library_option.__dict__.__setitem__('stypy_function_name', 'MSVCCompiler.library_option')
        MSVCCompiler.library_option.__dict__.__setitem__('stypy_param_names_list', ['lib'])
        MSVCCompiler.library_option.__dict__.__setitem__('stypy_varargs_param_name', None)
        MSVCCompiler.library_option.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MSVCCompiler.library_option.__dict__.__setitem__('stypy_call_defaults', defaults)
        MSVCCompiler.library_option.__dict__.__setitem__('stypy_call_varargs', varargs)
        MSVCCompiler.library_option.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MSVCCompiler.library_option.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MSVCCompiler.library_option', ['lib'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'library_option', localization, ['lib'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'library_option(...)' code ##################

        
        # Call to library_filename(...): (line 556)
        # Processing the call arguments (line 556)
        # Getting the type of 'lib' (line 556)
        lib_6438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 38), 'lib', False)
        # Processing the call keyword arguments (line 556)
        kwargs_6439 = {}
        # Getting the type of 'self' (line 556)
        self_6436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 15), 'self', False)
        # Obtaining the member 'library_filename' of a type (line 556)
        library_filename_6437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 15), self_6436, 'library_filename')
        # Calling library_filename(args, kwargs) (line 556)
        library_filename_call_result_6440 = invoke(stypy.reporting.localization.Localization(__file__, 556, 15), library_filename_6437, *[lib_6438], **kwargs_6439)
        
        # Assigning a type to the variable 'stypy_return_type' (line 556)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'stypy_return_type', library_filename_call_result_6440)
        
        # ################# End of 'library_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'library_option' in the type store
        # Getting the type of 'stypy_return_type' (line 555)
        stypy_return_type_6441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6441)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'library_option'
        return stypy_return_type_6441


    @norecursion
    def find_library_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_6442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 50), 'int')
        defaults = [int_6442]
        # Create a new context for function 'find_library_file'
        module_type_store = module_type_store.open_function_context('find_library_file', 559, 4, False)
        # Assigning a type to the variable 'self' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MSVCCompiler.find_library_file.__dict__.__setitem__('stypy_localization', localization)
        MSVCCompiler.find_library_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MSVCCompiler.find_library_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        MSVCCompiler.find_library_file.__dict__.__setitem__('stypy_function_name', 'MSVCCompiler.find_library_file')
        MSVCCompiler.find_library_file.__dict__.__setitem__('stypy_param_names_list', ['dirs', 'lib', 'debug'])
        MSVCCompiler.find_library_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        MSVCCompiler.find_library_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MSVCCompiler.find_library_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        MSVCCompiler.find_library_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        MSVCCompiler.find_library_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MSVCCompiler.find_library_file.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MSVCCompiler.find_library_file', ['dirs', 'lib', 'debug'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'find_library_file', localization, ['dirs', 'lib', 'debug'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'find_library_file(...)' code ##################

        
        # Getting the type of 'debug' (line 562)
        debug_6443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 11), 'debug')
        # Testing the type of an if condition (line 562)
        if_condition_6444 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 562, 8), debug_6443)
        # Assigning a type to the variable 'if_condition_6444' (line 562)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'if_condition_6444', if_condition_6444)
        # SSA begins for if statement (line 562)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 563):
        
        # Assigning a List to a Name (line 563):
        
        # Obtaining an instance of the builtin type 'list' (line 563)
        list_6445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 563)
        # Adding element type (line 563)
        # Getting the type of 'lib' (line 563)
        lib_6446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 25), 'lib')
        str_6447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 31), 'str', '_d')
        # Applying the binary operator '+' (line 563)
        result_add_6448 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 25), '+', lib_6446, str_6447)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 24), list_6445, result_add_6448)
        # Adding element type (line 563)
        # Getting the type of 'lib' (line 563)
        lib_6449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 37), 'lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 24), list_6445, lib_6449)
        
        # Assigning a type to the variable 'try_names' (line 563)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 12), 'try_names', list_6445)
        # SSA branch for the else part of an if statement (line 562)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Name (line 565):
        
        # Assigning a List to a Name (line 565):
        
        # Obtaining an instance of the builtin type 'list' (line 565)
        list_6450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 565)
        # Adding element type (line 565)
        # Getting the type of 'lib' (line 565)
        lib_6451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 25), 'lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 565, 24), list_6450, lib_6451)
        
        # Assigning a type to the variable 'try_names' (line 565)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 12), 'try_names', list_6450)
        # SSA join for if statement (line 562)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'dirs' (line 566)
        dirs_6452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 19), 'dirs')
        # Testing the type of a for loop iterable (line 566)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 566, 8), dirs_6452)
        # Getting the type of the for loop variable (line 566)
        for_loop_var_6453 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 566, 8), dirs_6452)
        # Assigning a type to the variable 'dir' (line 566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'dir', for_loop_var_6453)
        # SSA begins for a for statement (line 566)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'try_names' (line 567)
        try_names_6454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 24), 'try_names')
        # Testing the type of a for loop iterable (line 567)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 567, 12), try_names_6454)
        # Getting the type of the for loop variable (line 567)
        for_loop_var_6455 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 567, 12), try_names_6454)
        # Assigning a type to the variable 'name' (line 567)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 12), 'name', for_loop_var_6455)
        # SSA begins for a for statement (line 567)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 568):
        
        # Assigning a Call to a Name (line 568):
        
        # Call to join(...): (line 568)
        # Processing the call arguments (line 568)
        # Getting the type of 'dir' (line 568)
        dir_6459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 39), 'dir', False)
        
        # Call to library_filename(...): (line 568)
        # Processing the call arguments (line 568)
        # Getting the type of 'name' (line 568)
        name_6462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 67), 'name', False)
        # Processing the call keyword arguments (line 568)
        kwargs_6463 = {}
        # Getting the type of 'self' (line 568)
        self_6460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 44), 'self', False)
        # Obtaining the member 'library_filename' of a type (line 568)
        library_filename_6461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 44), self_6460, 'library_filename')
        # Calling library_filename(args, kwargs) (line 568)
        library_filename_call_result_6464 = invoke(stypy.reporting.localization.Localization(__file__, 568, 44), library_filename_6461, *[name_6462], **kwargs_6463)
        
        # Processing the call keyword arguments (line 568)
        kwargs_6465 = {}
        # Getting the type of 'os' (line 568)
        os_6456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 568)
        path_6457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 26), os_6456, 'path')
        # Obtaining the member 'join' of a type (line 568)
        join_6458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 26), path_6457, 'join')
        # Calling join(args, kwargs) (line 568)
        join_call_result_6466 = invoke(stypy.reporting.localization.Localization(__file__, 568, 26), join_6458, *[dir_6459, library_filename_call_result_6464], **kwargs_6465)
        
        # Assigning a type to the variable 'libfile' (line 568)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 16), 'libfile', join_call_result_6466)
        
        
        # Call to exists(...): (line 569)
        # Processing the call arguments (line 569)
        # Getting the type of 'libfile' (line 569)
        libfile_6470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 34), 'libfile', False)
        # Processing the call keyword arguments (line 569)
        kwargs_6471 = {}
        # Getting the type of 'os' (line 569)
        os_6467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 569)
        path_6468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 19), os_6467, 'path')
        # Obtaining the member 'exists' of a type (line 569)
        exists_6469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 19), path_6468, 'exists')
        # Calling exists(args, kwargs) (line 569)
        exists_call_result_6472 = invoke(stypy.reporting.localization.Localization(__file__, 569, 19), exists_6469, *[libfile_6470], **kwargs_6471)
        
        # Testing the type of an if condition (line 569)
        if_condition_6473 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 569, 16), exists_call_result_6472)
        # Assigning a type to the variable 'if_condition_6473' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 16), 'if_condition_6473', if_condition_6473)
        # SSA begins for if statement (line 569)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'libfile' (line 570)
        libfile_6474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 27), 'libfile')
        # Assigning a type to the variable 'stypy_return_type' (line 570)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 20), 'stypy_return_type', libfile_6474)
        # SSA join for if statement (line 569)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of a for statement (line 566)
        module_type_store.open_ssa_branch('for loop else')
        # Getting the type of 'None' (line 573)
        None_6475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 573)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 12), 'stypy_return_type', None_6475)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'find_library_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find_library_file' in the type store
        # Getting the type of 'stypy_return_type' (line 559)
        stypy_return_type_6476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6476)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find_library_file'
        return stypy_return_type_6476


    @norecursion
    def find_exe(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'find_exe'
        module_type_store = module_type_store.open_function_context('find_exe', 579, 4, False)
        # Assigning a type to the variable 'self' (line 580)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MSVCCompiler.find_exe.__dict__.__setitem__('stypy_localization', localization)
        MSVCCompiler.find_exe.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MSVCCompiler.find_exe.__dict__.__setitem__('stypy_type_store', module_type_store)
        MSVCCompiler.find_exe.__dict__.__setitem__('stypy_function_name', 'MSVCCompiler.find_exe')
        MSVCCompiler.find_exe.__dict__.__setitem__('stypy_param_names_list', ['exe'])
        MSVCCompiler.find_exe.__dict__.__setitem__('stypy_varargs_param_name', None)
        MSVCCompiler.find_exe.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MSVCCompiler.find_exe.__dict__.__setitem__('stypy_call_defaults', defaults)
        MSVCCompiler.find_exe.__dict__.__setitem__('stypy_call_varargs', varargs)
        MSVCCompiler.find_exe.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MSVCCompiler.find_exe.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MSVCCompiler.find_exe', ['exe'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'find_exe', localization, ['exe'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'find_exe(...)' code ##################

        str_6477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, (-1)), 'str', "Return path to an MSVC executable program.\n\n        Tries to find the program in several places: first, one of the\n        MSVC program search paths from the registry; next, the directories\n        in the PATH environment variable.  If any of those work, return an\n        absolute path that is known to exist.  If none of them work, just\n        return the original program name, 'exe'.\n        ")
        
        # Getting the type of 'self' (line 589)
        self_6478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 17), 'self')
        # Obtaining the member '__paths' of a type (line 589)
        paths_6479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 17), self_6478, '__paths')
        # Testing the type of a for loop iterable (line 589)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 589, 8), paths_6479)
        # Getting the type of the for loop variable (line 589)
        for_loop_var_6480 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 589, 8), paths_6479)
        # Assigning a type to the variable 'p' (line 589)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 8), 'p', for_loop_var_6480)
        # SSA begins for a for statement (line 589)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 590):
        
        # Assigning a Call to a Name (line 590):
        
        # Call to join(...): (line 590)
        # Processing the call arguments (line 590)
        
        # Call to abspath(...): (line 590)
        # Processing the call arguments (line 590)
        # Getting the type of 'p' (line 590)
        p_6487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 46), 'p', False)
        # Processing the call keyword arguments (line 590)
        kwargs_6488 = {}
        # Getting the type of 'os' (line 590)
        os_6484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 30), 'os', False)
        # Obtaining the member 'path' of a type (line 590)
        path_6485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 30), os_6484, 'path')
        # Obtaining the member 'abspath' of a type (line 590)
        abspath_6486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 30), path_6485, 'abspath')
        # Calling abspath(args, kwargs) (line 590)
        abspath_call_result_6489 = invoke(stypy.reporting.localization.Localization(__file__, 590, 30), abspath_6486, *[p_6487], **kwargs_6488)
        
        # Getting the type of 'exe' (line 590)
        exe_6490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 50), 'exe', False)
        # Processing the call keyword arguments (line 590)
        kwargs_6491 = {}
        # Getting the type of 'os' (line 590)
        os_6481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 590)
        path_6482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 17), os_6481, 'path')
        # Obtaining the member 'join' of a type (line 590)
        join_6483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 17), path_6482, 'join')
        # Calling join(args, kwargs) (line 590)
        join_call_result_6492 = invoke(stypy.reporting.localization.Localization(__file__, 590, 17), join_6483, *[abspath_call_result_6489, exe_6490], **kwargs_6491)
        
        # Assigning a type to the variable 'fn' (line 590)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 12), 'fn', join_call_result_6492)
        
        
        # Call to isfile(...): (line 591)
        # Processing the call arguments (line 591)
        # Getting the type of 'fn' (line 591)
        fn_6496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 30), 'fn', False)
        # Processing the call keyword arguments (line 591)
        kwargs_6497 = {}
        # Getting the type of 'os' (line 591)
        os_6493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 591)
        path_6494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 15), os_6493, 'path')
        # Obtaining the member 'isfile' of a type (line 591)
        isfile_6495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 15), path_6494, 'isfile')
        # Calling isfile(args, kwargs) (line 591)
        isfile_call_result_6498 = invoke(stypy.reporting.localization.Localization(__file__, 591, 15), isfile_6495, *[fn_6496], **kwargs_6497)
        
        # Testing the type of an if condition (line 591)
        if_condition_6499 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 591, 12), isfile_call_result_6498)
        # Assigning a type to the variable 'if_condition_6499' (line 591)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 12), 'if_condition_6499', if_condition_6499)
        # SSA begins for if statement (line 591)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'fn' (line 592)
        fn_6500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 23), 'fn')
        # Assigning a type to the variable 'stypy_return_type' (line 592)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 16), 'stypy_return_type', fn_6500)
        # SSA join for if statement (line 591)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to split(...): (line 595)
        # Processing the call arguments (line 595)
        
        # Obtaining the type of the subscript
        str_6503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 41), 'str', 'Path')
        # Getting the type of 'os' (line 595)
        os_6504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 30), 'os', False)
        # Obtaining the member 'environ' of a type (line 595)
        environ_6505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 30), os_6504, 'environ')
        # Obtaining the member '__getitem__' of a type (line 595)
        getitem___6506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 30), environ_6505, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 595)
        subscript_call_result_6507 = invoke(stypy.reporting.localization.Localization(__file__, 595, 30), getitem___6506, str_6503)
        
        str_6508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 49), 'str', ';')
        # Processing the call keyword arguments (line 595)
        kwargs_6509 = {}
        # Getting the type of 'string' (line 595)
        string_6501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 17), 'string', False)
        # Obtaining the member 'split' of a type (line 595)
        split_6502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 17), string_6501, 'split')
        # Calling split(args, kwargs) (line 595)
        split_call_result_6510 = invoke(stypy.reporting.localization.Localization(__file__, 595, 17), split_6502, *[subscript_call_result_6507, str_6508], **kwargs_6509)
        
        # Testing the type of a for loop iterable (line 595)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 595, 8), split_call_result_6510)
        # Getting the type of the for loop variable (line 595)
        for_loop_var_6511 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 595, 8), split_call_result_6510)
        # Assigning a type to the variable 'p' (line 595)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 8), 'p', for_loop_var_6511)
        # SSA begins for a for statement (line 595)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 596):
        
        # Assigning a Call to a Name (line 596):
        
        # Call to join(...): (line 596)
        # Processing the call arguments (line 596)
        
        # Call to abspath(...): (line 596)
        # Processing the call arguments (line 596)
        # Getting the type of 'p' (line 596)
        p_6518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 46), 'p', False)
        # Processing the call keyword arguments (line 596)
        kwargs_6519 = {}
        # Getting the type of 'os' (line 596)
        os_6515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 30), 'os', False)
        # Obtaining the member 'path' of a type (line 596)
        path_6516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 30), os_6515, 'path')
        # Obtaining the member 'abspath' of a type (line 596)
        abspath_6517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 30), path_6516, 'abspath')
        # Calling abspath(args, kwargs) (line 596)
        abspath_call_result_6520 = invoke(stypy.reporting.localization.Localization(__file__, 596, 30), abspath_6517, *[p_6518], **kwargs_6519)
        
        # Getting the type of 'exe' (line 596)
        exe_6521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 49), 'exe', False)
        # Processing the call keyword arguments (line 596)
        kwargs_6522 = {}
        # Getting the type of 'os' (line 596)
        os_6512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 596)
        path_6513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 17), os_6512, 'path')
        # Obtaining the member 'join' of a type (line 596)
        join_6514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 17), path_6513, 'join')
        # Calling join(args, kwargs) (line 596)
        join_call_result_6523 = invoke(stypy.reporting.localization.Localization(__file__, 596, 17), join_6514, *[abspath_call_result_6520, exe_6521], **kwargs_6522)
        
        # Assigning a type to the variable 'fn' (line 596)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 12), 'fn', join_call_result_6523)
        
        
        # Call to isfile(...): (line 597)
        # Processing the call arguments (line 597)
        # Getting the type of 'fn' (line 597)
        fn_6527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 30), 'fn', False)
        # Processing the call keyword arguments (line 597)
        kwargs_6528 = {}
        # Getting the type of 'os' (line 597)
        os_6524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 597)
        path_6525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 15), os_6524, 'path')
        # Obtaining the member 'isfile' of a type (line 597)
        isfile_6526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 15), path_6525, 'isfile')
        # Calling isfile(args, kwargs) (line 597)
        isfile_call_result_6529 = invoke(stypy.reporting.localization.Localization(__file__, 597, 15), isfile_6526, *[fn_6527], **kwargs_6528)
        
        # Testing the type of an if condition (line 597)
        if_condition_6530 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 597, 12), isfile_call_result_6529)
        # Assigning a type to the variable 'if_condition_6530' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 12), 'if_condition_6530', if_condition_6530)
        # SSA begins for if statement (line 597)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'fn' (line 598)
        fn_6531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 23), 'fn')
        # Assigning a type to the variable 'stypy_return_type' (line 598)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 16), 'stypy_return_type', fn_6531)
        # SSA join for if statement (line 597)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'exe' (line 600)
        exe_6532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 15), 'exe')
        # Assigning a type to the variable 'stypy_return_type' (line 600)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 8), 'stypy_return_type', exe_6532)
        
        # ################# End of 'find_exe(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find_exe' in the type store
        # Getting the type of 'stypy_return_type' (line 579)
        stypy_return_type_6533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6533)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find_exe'
        return stypy_return_type_6533


    @norecursion
    def get_msvc_paths(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_6534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 44), 'str', 'x86')
        defaults = [str_6534]
        # Create a new context for function 'get_msvc_paths'
        module_type_store = module_type_store.open_function_context('get_msvc_paths', 602, 4, False)
        # Assigning a type to the variable 'self' (line 603)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MSVCCompiler.get_msvc_paths.__dict__.__setitem__('stypy_localization', localization)
        MSVCCompiler.get_msvc_paths.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MSVCCompiler.get_msvc_paths.__dict__.__setitem__('stypy_type_store', module_type_store)
        MSVCCompiler.get_msvc_paths.__dict__.__setitem__('stypy_function_name', 'MSVCCompiler.get_msvc_paths')
        MSVCCompiler.get_msvc_paths.__dict__.__setitem__('stypy_param_names_list', ['path', 'platform'])
        MSVCCompiler.get_msvc_paths.__dict__.__setitem__('stypy_varargs_param_name', None)
        MSVCCompiler.get_msvc_paths.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MSVCCompiler.get_msvc_paths.__dict__.__setitem__('stypy_call_defaults', defaults)
        MSVCCompiler.get_msvc_paths.__dict__.__setitem__('stypy_call_varargs', varargs)
        MSVCCompiler.get_msvc_paths.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MSVCCompiler.get_msvc_paths.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MSVCCompiler.get_msvc_paths', ['path', 'platform'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_msvc_paths', localization, ['path', 'platform'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_msvc_paths(...)' code ##################

        str_6535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, (-1)), 'str', 'Get a list of devstudio directories (include, lib or path).\n\n        Return a list of strings.  The list will be empty if unable to\n        access the registry or appropriate registry keys not found.\n        ')
        
        
        # Getting the type of '_can_read_reg' (line 609)
        _can_read_reg_6536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 15), '_can_read_reg')
        # Applying the 'not' unary operator (line 609)
        result_not__6537 = python_operator(stypy.reporting.localization.Localization(__file__, 609, 11), 'not', _can_read_reg_6536)
        
        # Testing the type of an if condition (line 609)
        if_condition_6538 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 609, 8), result_not__6537)
        # Assigning a type to the variable 'if_condition_6538' (line 609)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 8), 'if_condition_6538', if_condition_6538)
        # SSA begins for if statement (line 609)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'list' (line 610)
        list_6539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 610)
        
        # Assigning a type to the variable 'stypy_return_type' (line 610)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 12), 'stypy_return_type', list_6539)
        # SSA join for if statement (line 609)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 612):
        
        # Assigning a BinOp to a Name (line 612):
        # Getting the type of 'path' (line 612)
        path_6540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 15), 'path')
        str_6541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 22), 'str', ' dirs')
        # Applying the binary operator '+' (line 612)
        result_add_6542 = python_operator(stypy.reporting.localization.Localization(__file__, 612, 15), '+', path_6540, str_6541)
        
        # Assigning a type to the variable 'path' (line 612)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 8), 'path', result_add_6542)
        
        
        # Getting the type of 'self' (line 613)
        self_6543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 11), 'self')
        # Obtaining the member '__version' of a type (line 613)
        version_6544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 11), self_6543, '__version')
        int_6545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 29), 'int')
        # Applying the binary operator '>=' (line 613)
        result_ge_6546 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 11), '>=', version_6544, int_6545)
        
        # Testing the type of an if condition (line 613)
        if_condition_6547 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 613, 8), result_ge_6546)
        # Assigning a type to the variable 'if_condition_6547' (line 613)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 8), 'if_condition_6547', if_condition_6547)
        # SSA begins for if statement (line 613)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 614):
        
        # Assigning a BinOp to a Name (line 614):
        str_6548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 19), 'str', '%s\\%0.1f\\VC\\VC_OBJECTS_PLATFORM_INFO\\Win32\\Directories')
        
        # Obtaining an instance of the builtin type 'tuple' (line 615)
        tuple_6549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 615)
        # Adding element type (line 615)
        # Getting the type of 'self' (line 615)
        self_6550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 22), 'self')
        # Obtaining the member '__root' of a type (line 615)
        root_6551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 22), self_6550, '__root')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 615, 22), tuple_6549, root_6551)
        # Adding element type (line 615)
        # Getting the type of 'self' (line 615)
        self_6552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 35), 'self')
        # Obtaining the member '__version' of a type (line 615)
        version_6553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 35), self_6552, '__version')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 615, 22), tuple_6549, version_6553)
        
        # Applying the binary operator '%' (line 614)
        result_mod_6554 = python_operator(stypy.reporting.localization.Localization(__file__, 614, 19), '%', str_6548, tuple_6549)
        
        # Assigning a type to the variable 'key' (line 614)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 12), 'key', result_mod_6554)
        # SSA branch for the else part of an if statement (line 613)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 617):
        
        # Assigning a BinOp to a Name (line 617):
        str_6555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 19), 'str', '%s\\6.0\\Build System\\Components\\Platforms\\Win32 (%s)\\Directories')
        
        # Obtaining an instance of the builtin type 'tuple' (line 618)
        tuple_6556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 618)
        # Adding element type (line 618)
        # Getting the type of 'self' (line 618)
        self_6557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 49), 'self')
        # Obtaining the member '__root' of a type (line 618)
        root_6558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 49), self_6557, '__root')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 618, 49), tuple_6556, root_6558)
        # Adding element type (line 618)
        # Getting the type of 'platform' (line 618)
        platform_6559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 62), 'platform')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 618, 49), tuple_6556, platform_6559)
        
        # Applying the binary operator '%' (line 617)
        result_mod_6560 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 19), '%', str_6555, tuple_6556)
        
        # Assigning a type to the variable 'key' (line 617)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 12), 'key', result_mod_6560)
        # SSA join for if statement (line 613)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'HKEYS' (line 620)
        HKEYS_6561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 20), 'HKEYS')
        # Testing the type of a for loop iterable (line 620)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 620, 8), HKEYS_6561)
        # Getting the type of the for loop variable (line 620)
        for_loop_var_6562 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 620, 8), HKEYS_6561)
        # Assigning a type to the variable 'base' (line 620)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 8), 'base', for_loop_var_6562)
        # SSA begins for a for statement (line 620)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 621):
        
        # Assigning a Call to a Name (line 621):
        
        # Call to read_values(...): (line 621)
        # Processing the call arguments (line 621)
        # Getting the type of 'base' (line 621)
        base_6564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 28), 'base', False)
        # Getting the type of 'key' (line 621)
        key_6565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 34), 'key', False)
        # Processing the call keyword arguments (line 621)
        kwargs_6566 = {}
        # Getting the type of 'read_values' (line 621)
        read_values_6563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 16), 'read_values', False)
        # Calling read_values(args, kwargs) (line 621)
        read_values_call_result_6567 = invoke(stypy.reporting.localization.Localization(__file__, 621, 16), read_values_6563, *[base_6564, key_6565], **kwargs_6566)
        
        # Assigning a type to the variable 'd' (line 621)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 12), 'd', read_values_call_result_6567)
        
        # Getting the type of 'd' (line 622)
        d_6568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 15), 'd')
        # Testing the type of an if condition (line 622)
        if_condition_6569 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 622, 12), d_6568)
        # Assigning a type to the variable 'if_condition_6569' (line 622)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 12), 'if_condition_6569', if_condition_6569)
        # SSA begins for if statement (line 622)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'self' (line 623)
        self_6570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 19), 'self')
        # Obtaining the member '__version' of a type (line 623)
        version_6571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 19), self_6570, '__version')
        int_6572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 37), 'int')
        # Applying the binary operator '>=' (line 623)
        result_ge_6573 = python_operator(stypy.reporting.localization.Localization(__file__, 623, 19), '>=', version_6571, int_6572)
        
        # Testing the type of an if condition (line 623)
        if_condition_6574 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 623, 16), result_ge_6573)
        # Assigning a type to the variable 'if_condition_6574' (line 623)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 16), 'if_condition_6574', if_condition_6574)
        # SSA begins for if statement (line 623)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to split(...): (line 624)
        # Processing the call arguments (line 624)
        
        # Call to sub(...): (line 624)
        # Processing the call arguments (line 624)
        
        # Obtaining the type of the subscript
        # Getting the type of 'path' (line 624)
        path_6580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 60), 'path', False)
        # Getting the type of 'd' (line 624)
        d_6581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 58), 'd', False)
        # Obtaining the member '__getitem__' of a type (line 624)
        getitem___6582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 58), d_6581, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 624)
        subscript_call_result_6583 = invoke(stypy.reporting.localization.Localization(__file__, 624, 58), getitem___6582, path_6580)
        
        # Processing the call keyword arguments (line 624)
        kwargs_6584 = {}
        # Getting the type of 'self' (line 624)
        self_6577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 40), 'self', False)
        # Obtaining the member '__macros' of a type (line 624)
        macros_6578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 40), self_6577, '__macros')
        # Obtaining the member 'sub' of a type (line 624)
        sub_6579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 40), macros_6578, 'sub')
        # Calling sub(args, kwargs) (line 624)
        sub_call_result_6585 = invoke(stypy.reporting.localization.Localization(__file__, 624, 40), sub_6579, *[subscript_call_result_6583], **kwargs_6584)
        
        str_6586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 68), 'str', ';')
        # Processing the call keyword arguments (line 624)
        kwargs_6587 = {}
        # Getting the type of 'string' (line 624)
        string_6575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 27), 'string', False)
        # Obtaining the member 'split' of a type (line 624)
        split_6576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 27), string_6575, 'split')
        # Calling split(args, kwargs) (line 624)
        split_call_result_6588 = invoke(stypy.reporting.localization.Localization(__file__, 624, 27), split_6576, *[sub_call_result_6585, str_6586], **kwargs_6587)
        
        # Assigning a type to the variable 'stypy_return_type' (line 624)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 20), 'stypy_return_type', split_call_result_6588)
        # SSA branch for the else part of an if statement (line 623)
        module_type_store.open_ssa_branch('else')
        
        # Call to split(...): (line 626)
        # Processing the call arguments (line 626)
        
        # Obtaining the type of the subscript
        # Getting the type of 'path' (line 626)
        path_6591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 42), 'path', False)
        # Getting the type of 'd' (line 626)
        d_6592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 40), 'd', False)
        # Obtaining the member '__getitem__' of a type (line 626)
        getitem___6593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 40), d_6592, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 626)
        subscript_call_result_6594 = invoke(stypy.reporting.localization.Localization(__file__, 626, 40), getitem___6593, path_6591)
        
        str_6595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 49), 'str', ';')
        # Processing the call keyword arguments (line 626)
        kwargs_6596 = {}
        # Getting the type of 'string' (line 626)
        string_6589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 27), 'string', False)
        # Obtaining the member 'split' of a type (line 626)
        split_6590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 27), string_6589, 'split')
        # Calling split(args, kwargs) (line 626)
        split_call_result_6597 = invoke(stypy.reporting.localization.Localization(__file__, 626, 27), split_6590, *[subscript_call_result_6594, str_6595], **kwargs_6596)
        
        # Assigning a type to the variable 'stypy_return_type' (line 626)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 20), 'stypy_return_type', split_call_result_6597)
        # SSA join for if statement (line 623)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 622)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 629)
        self_6598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 11), 'self')
        # Obtaining the member '__version' of a type (line 629)
        version_6599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 11), self_6598, '__version')
        int_6600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 29), 'int')
        # Applying the binary operator '==' (line 629)
        result_eq_6601 = python_operator(stypy.reporting.localization.Localization(__file__, 629, 11), '==', version_6599, int_6600)
        
        # Testing the type of an if condition (line 629)
        if_condition_6602 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 629, 8), result_eq_6601)
        # Assigning a type to the variable 'if_condition_6602' (line 629)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 8), 'if_condition_6602', if_condition_6602)
        # SSA begins for if statement (line 629)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'HKEYS' (line 630)
        HKEYS_6603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 24), 'HKEYS')
        # Testing the type of a for loop iterable (line 630)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 630, 12), HKEYS_6603)
        # Getting the type of the for loop variable (line 630)
        for_loop_var_6604 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 630, 12), HKEYS_6603)
        # Assigning a type to the variable 'base' (line 630)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 12), 'base', for_loop_var_6604)
        # SSA begins for a for statement (line 630)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Call to read_values(...): (line 631)
        # Processing the call arguments (line 631)
        # Getting the type of 'base' (line 631)
        base_6606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 31), 'base', False)
        str_6607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 37), 'str', '%s\\6.0')
        # Getting the type of 'self' (line 631)
        self_6608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 49), 'self', False)
        # Obtaining the member '__root' of a type (line 631)
        root_6609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 49), self_6608, '__root')
        # Applying the binary operator '%' (line 631)
        result_mod_6610 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 37), '%', str_6607, root_6609)
        
        # Processing the call keyword arguments (line 631)
        kwargs_6611 = {}
        # Getting the type of 'read_values' (line 631)
        read_values_6605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 19), 'read_values', False)
        # Calling read_values(args, kwargs) (line 631)
        read_values_call_result_6612 = invoke(stypy.reporting.localization.Localization(__file__, 631, 19), read_values_6605, *[base_6606, result_mod_6610], **kwargs_6611)
        
        # Getting the type of 'None' (line 631)
        None_6613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 69), 'None')
        # Applying the binary operator 'isnot' (line 631)
        result_is_not_6614 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 19), 'isnot', read_values_call_result_6612, None_6613)
        
        # Testing the type of an if condition (line 631)
        if_condition_6615 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 631, 16), result_is_not_6614)
        # Assigning a type to the variable 'if_condition_6615' (line 631)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 16), 'if_condition_6615', if_condition_6615)
        # SSA begins for if statement (line 631)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 632)
        # Processing the call arguments (line 632)
        str_6618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 30), 'str', 'It seems you have Visual Studio 6 installed, but the expected registry settings are not present.\nYou must at least run the Visual Studio GUI once so that these entries are created.')
        # Processing the call keyword arguments (line 632)
        kwargs_6619 = {}
        # Getting the type of 'self' (line 632)
        self_6616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 20), 'self', False)
        # Obtaining the member 'warn' of a type (line 632)
        warn_6617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 20), self_6616, 'warn')
        # Calling warn(args, kwargs) (line 632)
        warn_call_result_6620 = invoke(stypy.reporting.localization.Localization(__file__, 632, 20), warn_6617, *[str_6618], **kwargs_6619)
        
        # SSA join for if statement (line 631)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 629)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'list' (line 637)
        list_6621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 637)
        
        # Assigning a type to the variable 'stypy_return_type' (line 637)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 8), 'stypy_return_type', list_6621)
        
        # ################# End of 'get_msvc_paths(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_msvc_paths' in the type store
        # Getting the type of 'stypy_return_type' (line 602)
        stypy_return_type_6622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6622)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_msvc_paths'
        return stypy_return_type_6622


    @norecursion
    def set_path_env_var(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_path_env_var'
        module_type_store = module_type_store.open_function_context('set_path_env_var', 639, 4, False)
        # Assigning a type to the variable 'self' (line 640)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MSVCCompiler.set_path_env_var.__dict__.__setitem__('stypy_localization', localization)
        MSVCCompiler.set_path_env_var.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MSVCCompiler.set_path_env_var.__dict__.__setitem__('stypy_type_store', module_type_store)
        MSVCCompiler.set_path_env_var.__dict__.__setitem__('stypy_function_name', 'MSVCCompiler.set_path_env_var')
        MSVCCompiler.set_path_env_var.__dict__.__setitem__('stypy_param_names_list', ['name'])
        MSVCCompiler.set_path_env_var.__dict__.__setitem__('stypy_varargs_param_name', None)
        MSVCCompiler.set_path_env_var.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MSVCCompiler.set_path_env_var.__dict__.__setitem__('stypy_call_defaults', defaults)
        MSVCCompiler.set_path_env_var.__dict__.__setitem__('stypy_call_varargs', varargs)
        MSVCCompiler.set_path_env_var.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MSVCCompiler.set_path_env_var.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MSVCCompiler.set_path_env_var', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_path_env_var', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_path_env_var(...)' code ##################

        str_6623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, (-1)), 'str', "Set environment variable 'name' to an MSVC path type value.\n\n        This is equivalent to a SET command prior to execution of spawned\n        commands.\n        ")
        
        
        # Getting the type of 'name' (line 646)
        name_6624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 11), 'name')
        str_6625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 19), 'str', 'lib')
        # Applying the binary operator '==' (line 646)
        result_eq_6626 = python_operator(stypy.reporting.localization.Localization(__file__, 646, 11), '==', name_6624, str_6625)
        
        # Testing the type of an if condition (line 646)
        if_condition_6627 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 646, 8), result_eq_6626)
        # Assigning a type to the variable 'if_condition_6627' (line 646)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 8), 'if_condition_6627', if_condition_6627)
        # SSA begins for if statement (line 646)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 647):
        
        # Assigning a Call to a Name (line 647):
        
        # Call to get_msvc_paths(...): (line 647)
        # Processing the call arguments (line 647)
        str_6630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 647, 36), 'str', 'library')
        # Processing the call keyword arguments (line 647)
        kwargs_6631 = {}
        # Getting the type of 'self' (line 647)
        self_6628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 16), 'self', False)
        # Obtaining the member 'get_msvc_paths' of a type (line 647)
        get_msvc_paths_6629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 16), self_6628, 'get_msvc_paths')
        # Calling get_msvc_paths(args, kwargs) (line 647)
        get_msvc_paths_call_result_6632 = invoke(stypy.reporting.localization.Localization(__file__, 647, 16), get_msvc_paths_6629, *[str_6630], **kwargs_6631)
        
        # Assigning a type to the variable 'p' (line 647)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 12), 'p', get_msvc_paths_call_result_6632)
        # SSA branch for the else part of an if statement (line 646)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 649):
        
        # Assigning a Call to a Name (line 649):
        
        # Call to get_msvc_paths(...): (line 649)
        # Processing the call arguments (line 649)
        # Getting the type of 'name' (line 649)
        name_6635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 36), 'name', False)
        # Processing the call keyword arguments (line 649)
        kwargs_6636 = {}
        # Getting the type of 'self' (line 649)
        self_6633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 16), 'self', False)
        # Obtaining the member 'get_msvc_paths' of a type (line 649)
        get_msvc_paths_6634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 16), self_6633, 'get_msvc_paths')
        # Calling get_msvc_paths(args, kwargs) (line 649)
        get_msvc_paths_call_result_6637 = invoke(stypy.reporting.localization.Localization(__file__, 649, 16), get_msvc_paths_6634, *[name_6635], **kwargs_6636)
        
        # Assigning a type to the variable 'p' (line 649)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 12), 'p', get_msvc_paths_call_result_6637)
        # SSA join for if statement (line 646)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'p' (line 650)
        p_6638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 11), 'p')
        # Testing the type of an if condition (line 650)
        if_condition_6639 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 650, 8), p_6638)
        # Assigning a type to the variable 'if_condition_6639' (line 650)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 8), 'if_condition_6639', if_condition_6639)
        # SSA begins for if statement (line 650)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Subscript (line 651):
        
        # Assigning a Call to a Subscript (line 651):
        
        # Call to join(...): (line 651)
        # Processing the call arguments (line 651)
        # Getting the type of 'p' (line 651)
        p_6642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 43), 'p', False)
        str_6643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 46), 'str', ';')
        # Processing the call keyword arguments (line 651)
        kwargs_6644 = {}
        # Getting the type of 'string' (line 651)
        string_6640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 31), 'string', False)
        # Obtaining the member 'join' of a type (line 651)
        join_6641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 31), string_6640, 'join')
        # Calling join(args, kwargs) (line 651)
        join_call_result_6645 = invoke(stypy.reporting.localization.Localization(__file__, 651, 31), join_6641, *[p_6642, str_6643], **kwargs_6644)
        
        # Getting the type of 'os' (line 651)
        os_6646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 12), 'os')
        # Obtaining the member 'environ' of a type (line 651)
        environ_6647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 12), os_6646, 'environ')
        # Getting the type of 'name' (line 651)
        name_6648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 23), 'name')
        # Storing an element on a container (line 651)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 651, 12), environ_6647, (name_6648, join_call_result_6645))
        # SSA join for if statement (line 650)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'set_path_env_var(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_path_env_var' in the type store
        # Getting the type of 'stypy_return_type' (line 639)
        stypy_return_type_6649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6649)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_path_env_var'
        return stypy_return_type_6649


# Assigning a type to the variable 'MSVCCompiler' (line 204)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 0), 'MSVCCompiler', MSVCCompiler)

# Assigning a Str to a Name (line 208):
str_6650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 20), 'str', 'msvc')
# Getting the type of 'MSVCCompiler'
MSVCCompiler_6651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_6651, 'compiler_type', str_6650)

# Assigning a Dict to a Name (line 215):

# Obtaining an instance of the builtin type 'dict' (line 215)
dict_6652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 215)

# Getting the type of 'MSVCCompiler'
MSVCCompiler_6653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member 'executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_6653, 'executables', dict_6652)

# Assigning a List to a Name (line 218):

# Obtaining an instance of the builtin type 'list' (line 218)
list_6654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 218)
# Adding element type (line 218)
str_6655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 21), 'str', '.c')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 20), list_6654, str_6655)

# Getting the type of 'MSVCCompiler'
MSVCCompiler_6656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member '_c_extensions' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_6656, '_c_extensions', list_6654)

# Assigning a List to a Name (line 219):

# Obtaining an instance of the builtin type 'list' (line 219)
list_6657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 219)
# Adding element type (line 219)
str_6658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 23), 'str', '.cc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 22), list_6657, str_6658)
# Adding element type (line 219)
str_6659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 30), 'str', '.cpp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 22), list_6657, str_6659)
# Adding element type (line 219)
str_6660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 38), 'str', '.cxx')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 22), list_6657, str_6660)

# Getting the type of 'MSVCCompiler'
MSVCCompiler_6661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member '_cpp_extensions' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_6661, '_cpp_extensions', list_6657)

# Assigning a List to a Name (line 220):

# Obtaining an instance of the builtin type 'list' (line 220)
list_6662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 220)
# Adding element type (line 220)
str_6663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 22), 'str', '.rc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 21), list_6662, str_6663)

# Getting the type of 'MSVCCompiler'
MSVCCompiler_6664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member '_rc_extensions' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_6664, '_rc_extensions', list_6662)

# Assigning a List to a Name (line 221):

# Obtaining an instance of the builtin type 'list' (line 221)
list_6665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 221)
# Adding element type (line 221)
str_6666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 22), 'str', '.mc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 21), list_6665, str_6666)

# Getting the type of 'MSVCCompiler'
MSVCCompiler_6667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member '_mc_extensions' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_6667, '_mc_extensions', list_6665)

# Assigning a BinOp to a Name (line 225):
# Getting the type of 'MSVCCompiler'
MSVCCompiler_6668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Obtaining the member '_c_extensions' of a type
_c_extensions_6669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_6668, '_c_extensions')
# Getting the type of 'MSVCCompiler'
MSVCCompiler_6670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Obtaining the member '_cpp_extensions' of a type
_cpp_extensions_6671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_6670, '_cpp_extensions')
# Applying the binary operator '+' (line 225)
result_add_6672 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 22), '+', _c_extensions_6669, _cpp_extensions_6671)

# Getting the type of 'MSVCCompiler'
MSVCCompiler_6673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Obtaining the member '_rc_extensions' of a type
_rc_extensions_6674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_6673, '_rc_extensions')
# Applying the binary operator '+' (line 225)
result_add_6675 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 54), '+', result_add_6672, _rc_extensions_6674)

# Getting the type of 'MSVCCompiler'
MSVCCompiler_6676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Obtaining the member '_mc_extensions' of a type
_mc_extensions_6677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_6676, '_mc_extensions')
# Applying the binary operator '+' (line 226)
result_add_6678 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 37), '+', result_add_6675, _mc_extensions_6677)

# Getting the type of 'MSVCCompiler'
MSVCCompiler_6679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member 'src_extensions' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_6679, 'src_extensions', result_add_6678)

# Assigning a Str to a Name (line 227):
str_6680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 20), 'str', '.res')
# Getting the type of 'MSVCCompiler'
MSVCCompiler_6681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member 'res_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_6681, 'res_extension', str_6680)

# Assigning a Str to a Name (line 228):
str_6682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 20), 'str', '.obj')
# Getting the type of 'MSVCCompiler'
MSVCCompiler_6683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member 'obj_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_6683, 'obj_extension', str_6682)

# Assigning a Str to a Name (line 229):
str_6684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 27), 'str', '.lib')
# Getting the type of 'MSVCCompiler'
MSVCCompiler_6685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member 'static_lib_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_6685, 'static_lib_extension', str_6684)

# Assigning a Str to a Name (line 230):
str_6686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 27), 'str', '.dll')
# Getting the type of 'MSVCCompiler'
MSVCCompiler_6687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member 'shared_lib_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_6687, 'shared_lib_extension', str_6686)

# Assigning a Str to a Name (line 231):
str_6688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 44), 'str', '%s%s')
# Getting the type of 'MSVCCompiler'
MSVCCompiler_6689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member 'shared_lib_format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_6689, 'shared_lib_format', str_6688)

# Assigning a Name to a Name (line 231):
# Getting the type of 'MSVCCompiler'
MSVCCompiler_6690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Obtaining the member 'shared_lib_format' of a type
shared_lib_format_6691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_6690, 'shared_lib_format')
# Getting the type of 'MSVCCompiler'
MSVCCompiler_6692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member 'static_lib_format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_6692, 'static_lib_format', shared_lib_format_6691)

# Assigning a Str to a Name (line 232):
str_6693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 20), 'str', '.exe')
# Getting the type of 'MSVCCompiler'
MSVCCompiler_6694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member 'exe_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_6694, 'exe_extension', str_6693)



# Call to get_build_version(...): (line 654)
# Processing the call keyword arguments (line 654)
kwargs_6696 = {}
# Getting the type of 'get_build_version' (line 654)
get_build_version_6695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 3), 'get_build_version', False)
# Calling get_build_version(args, kwargs) (line 654)
get_build_version_call_result_6697 = invoke(stypy.reporting.localization.Localization(__file__, 654, 3), get_build_version_6695, *[], **kwargs_6696)

float_6698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 26), 'float')
# Applying the binary operator '>=' (line 654)
result_ge_6699 = python_operator(stypy.reporting.localization.Localization(__file__, 654, 3), '>=', get_build_version_call_result_6697, float_6698)

# Testing the type of an if condition (line 654)
if_condition_6700 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 654, 0), result_ge_6699)
# Assigning a type to the variable 'if_condition_6700' (line 654)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 0), 'if_condition_6700', if_condition_6700)
# SSA begins for if statement (line 654)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to debug(...): (line 655)
# Processing the call arguments (line 655)
str_6703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 14), 'str', 'Importing new compiler from distutils.msvc9compiler')
# Processing the call keyword arguments (line 655)
kwargs_6704 = {}
# Getting the type of 'log' (line 655)
log_6701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 4), 'log', False)
# Obtaining the member 'debug' of a type (line 655)
debug_6702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 4), log_6701, 'debug')
# Calling debug(args, kwargs) (line 655)
debug_call_result_6705 = invoke(stypy.reporting.localization.Localization(__file__, 655, 4), debug_6702, *[str_6703], **kwargs_6704)


# Assigning a Name to a Name (line 656):

# Assigning a Name to a Name (line 656):
# Getting the type of 'MSVCCompiler' (line 656)
MSVCCompiler_6706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 22), 'MSVCCompiler')
# Assigning a type to the variable 'OldMSVCCompiler' (line 656)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 4), 'OldMSVCCompiler', MSVCCompiler_6706)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 657, 4))

# 'from distutils.msvc9compiler import MSVCCompiler' statement (line 657)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_6707 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 657, 4), 'distutils.msvc9compiler')

if (type(import_6707) is not StypyTypeError):

    if (import_6707 != 'pyd_module'):
        __import__(import_6707)
        sys_modules_6708 = sys.modules[import_6707]
        import_from_module(stypy.reporting.localization.Localization(__file__, 657, 4), 'distutils.msvc9compiler', sys_modules_6708.module_type_store, module_type_store, ['MSVCCompiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 657, 4), __file__, sys_modules_6708, sys_modules_6708.module_type_store, module_type_store)
    else:
        from distutils.msvc9compiler import MSVCCompiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 657, 4), 'distutils.msvc9compiler', None, module_type_store, ['MSVCCompiler'], [MSVCCompiler])

else:
    # Assigning a type to the variable 'distutils.msvc9compiler' (line 657)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 4), 'distutils.msvc9compiler', import_6707)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 659, 4))

# 'from distutils.msvc9compiler import MacroExpander' statement (line 659)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_6709 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 659, 4), 'distutils.msvc9compiler')

if (type(import_6709) is not StypyTypeError):

    if (import_6709 != 'pyd_module'):
        __import__(import_6709)
        sys_modules_6710 = sys.modules[import_6709]
        import_from_module(stypy.reporting.localization.Localization(__file__, 659, 4), 'distutils.msvc9compiler', sys_modules_6710.module_type_store, module_type_store, ['MacroExpander'])
        nest_module(stypy.reporting.localization.Localization(__file__, 659, 4), __file__, sys_modules_6710, sys_modules_6710.module_type_store, module_type_store)
    else:
        from distutils.msvc9compiler import MacroExpander

        import_from_module(stypy.reporting.localization.Localization(__file__, 659, 4), 'distutils.msvc9compiler', None, module_type_store, ['MacroExpander'], [MacroExpander])

else:
    # Assigning a type to the variable 'distutils.msvc9compiler' (line 659)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 4), 'distutils.msvc9compiler', import_6709)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

# SSA join for if statement (line 654)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

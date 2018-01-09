
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.msvc9compiler
2: 
3: Contains MSVCCompiler, an implementation of the abstract CCompiler class
4: for the Microsoft Visual Studio 2008.
5: 
6: The module is compatible with VS 2005 and VS 2008. You can find legacy support
7: for older versions of VS in distutils.msvccompiler.
8: '''
9: 
10: # Written by Perry Stoll
11: # hacked by Robin Becker and Thomas Heller to do a better job of
12: #   finding DevStudio (through the registry)
13: # ported to VS2005 and VS 2008 by Christian Heimes
14: 
15: __revision__ = "$Id$"
16: 
17: import os
18: import subprocess
19: import sys
20: import re
21: 
22: from distutils.errors import (DistutilsExecError, DistutilsPlatformError,
23:                               CompileError, LibError, LinkError)
24: from distutils.ccompiler import CCompiler, gen_lib_options
25: from distutils import log
26: from distutils.util import get_platform
27: 
28: import _winreg
29: 
30: RegOpenKeyEx = _winreg.OpenKeyEx
31: RegEnumKey = _winreg.EnumKey
32: RegEnumValue = _winreg.EnumValue
33: RegError = _winreg.error
34: 
35: HKEYS = (_winreg.HKEY_USERS,
36:          _winreg.HKEY_CURRENT_USER,
37:          _winreg.HKEY_LOCAL_MACHINE,
38:          _winreg.HKEY_CLASSES_ROOT)
39: 
40: NATIVE_WIN64 = (sys.platform == 'win32' and sys.maxsize > 2**32)
41: if NATIVE_WIN64:
42:     # Visual C++ is a 32-bit application, so we need to look in
43:     # the corresponding registry branch, if we're running a
44:     # 64-bit Python on Win64
45:     VS_BASE = r"Software\Wow6432Node\Microsoft\VisualStudio\%0.1f"
46:     VSEXPRESS_BASE = r"Software\Wow6432Node\Microsoft\VCExpress\%0.1f"
47:     WINSDK_BASE = r"Software\Wow6432Node\Microsoft\Microsoft SDKs\Windows"
48:     NET_BASE = r"Software\Wow6432Node\Microsoft\.NETFramework"
49: else:
50:     VS_BASE = r"Software\Microsoft\VisualStudio\%0.1f"
51:     VSEXPRESS_BASE = r"Software\Microsoft\VCExpress\%0.1f"
52:     WINSDK_BASE = r"Software\Microsoft\Microsoft SDKs\Windows"
53:     NET_BASE = r"Software\Microsoft\.NETFramework"
54: 
55: # A map keyed by get_platform() return values to values accepted by
56: # 'vcvarsall.bat'.  Note a cross-compile may combine these (eg, 'x86_amd64' is
57: # the param to cross-compile on x86 targeting amd64.)
58: PLAT_TO_VCVARS = {
59:     'win32' : 'x86',
60:     'win-amd64' : 'amd64',
61:     'win-ia64' : 'ia64',
62: }
63: 
64: class Reg:
65:     '''Helper class to read values from the registry
66:     '''
67: 
68:     def get_value(cls, path, key):
69:         for base in HKEYS:
70:             d = cls.read_values(base, path)
71:             if d and key in d:
72:                 return d[key]
73:         raise KeyError(key)
74:     get_value = classmethod(get_value)
75: 
76:     def read_keys(cls, base, key):
77:         '''Return list of registry keys.'''
78:         try:
79:             handle = RegOpenKeyEx(base, key)
80:         except RegError:
81:             return None
82:         L = []
83:         i = 0
84:         while True:
85:             try:
86:                 k = RegEnumKey(handle, i)
87:             except RegError:
88:                 break
89:             L.append(k)
90:             i += 1
91:         return L
92:     read_keys = classmethod(read_keys)
93: 
94:     def read_values(cls, base, key):
95:         '''Return dict of registry keys and values.
96: 
97:         All names are converted to lowercase.
98:         '''
99:         try:
100:             handle = RegOpenKeyEx(base, key)
101:         except RegError:
102:             return None
103:         d = {}
104:         i = 0
105:         while True:
106:             try:
107:                 name, value, type = RegEnumValue(handle, i)
108:             except RegError:
109:                 break
110:             name = name.lower()
111:             d[cls.convert_mbcs(name)] = cls.convert_mbcs(value)
112:             i += 1
113:         return d
114:     read_values = classmethod(read_values)
115: 
116:     def convert_mbcs(s):
117:         dec = getattr(s, "decode", None)
118:         if dec is not None:
119:             try:
120:                 s = dec("mbcs")
121:             except UnicodeError:
122:                 pass
123:         return s
124:     convert_mbcs = staticmethod(convert_mbcs)
125: 
126: class MacroExpander:
127: 
128:     def __init__(self, version):
129:         self.macros = {}
130:         self.vsbase = VS_BASE % version
131:         self.load_macros(version)
132: 
133:     def set_macro(self, macro, path, key):
134:         self.macros["$(%s)" % macro] = Reg.get_value(path, key)
135: 
136:     def load_macros(self, version):
137:         self.set_macro("VCInstallDir", self.vsbase + r"\Setup\VC", "productdir")
138:         self.set_macro("VSInstallDir", self.vsbase + r"\Setup\VS", "productdir")
139:         self.set_macro("FrameworkDir", NET_BASE, "installroot")
140:         try:
141:             if version >= 8.0:
142:                 self.set_macro("FrameworkSDKDir", NET_BASE,
143:                                "sdkinstallrootv2.0")
144:             else:
145:                 raise KeyError("sdkinstallrootv2.0")
146:         except KeyError:
147:             raise DistutilsPlatformError(
148:             '''Python was built with Visual Studio 2008;
149: extensions must be built with a compiler than can generate compatible binaries.
150: Visual Studio 2008 was not found on this system. If you have Cygwin installed,
151: you can try compiling with MingW32, by passing "-c mingw32" to setup.py.''')
152: 
153:         if version >= 9.0:
154:             self.set_macro("FrameworkVersion", self.vsbase, "clr version")
155:             self.set_macro("WindowsSdkDir", WINSDK_BASE, "currentinstallfolder")
156:         else:
157:             p = r"Software\Microsoft\NET Framework Setup\Product"
158:             for base in HKEYS:
159:                 try:
160:                     h = RegOpenKeyEx(base, p)
161:                 except RegError:
162:                     continue
163:                 key = RegEnumKey(h, 0)
164:                 d = Reg.get_value(base, r"%s\%s" % (p, key))
165:                 self.macros["$(FrameworkVersion)"] = d["version"]
166: 
167:     def sub(self, s):
168:         for k, v in self.macros.items():
169:             s = s.replace(k, v)
170:         return s
171: 
172: def get_build_version():
173:     '''Return the version of MSVC that was used to build Python.
174: 
175:     For Python 2.3 and up, the version number is included in
176:     sys.version.  For earlier versions, assume the compiler is MSVC 6.
177:     '''
178:     prefix = "MSC v."
179:     i = sys.version.find(prefix)
180:     if i == -1:
181:         return 6
182:     i = i + len(prefix)
183:     s, rest = sys.version[i:].split(" ", 1)
184:     majorVersion = int(s[:-2]) - 6
185:     minorVersion = int(s[2:3]) / 10.0
186:     # I don't think paths are affected by minor version in version 6
187:     if majorVersion == 6:
188:         minorVersion = 0
189:     if majorVersion >= 6:
190:         return majorVersion + minorVersion
191:     # else we don't know what version of the compiler this is
192:     return None
193: 
194: def normalize_and_reduce_paths(paths):
195:     '''Return a list of normalized paths with duplicates removed.
196: 
197:     The current order of paths is maintained.
198:     '''
199:     # Paths are normalized so things like:  /a and /a/ aren't both preserved.
200:     reduced_paths = []
201:     for p in paths:
202:         np = os.path.normpath(p)
203:         # XXX(nnorwitz): O(n**2), if reduced_paths gets long perhaps use a set.
204:         if np not in reduced_paths:
205:             reduced_paths.append(np)
206:     return reduced_paths
207: 
208: def removeDuplicates(variable):
209:     '''Remove duplicate values of an environment variable.
210:     '''
211:     oldList = variable.split(os.pathsep)
212:     newList = []
213:     for i in oldList:
214:         if i not in newList:
215:             newList.append(i)
216:     newVariable = os.pathsep.join(newList)
217:     return newVariable
218: 
219: def find_vcvarsall(version):
220:     '''Find the vcvarsall.bat file
221: 
222:     At first it tries to find the productdir of VS 2008 in the registry. If
223:     that fails it falls back to the VS90COMNTOOLS env var.
224:     '''
225:     vsbase = VS_BASE % version
226:     try:
227:         productdir = Reg.get_value(r"%s\Setup\VC" % vsbase,
228:                                    "productdir")
229:     except KeyError:
230:         productdir = None
231: 
232:     # trying Express edition
233:     if productdir is None:
234:         vsbase = VSEXPRESS_BASE % version
235:         try:
236:             productdir = Reg.get_value(r"%s\Setup\VC" % vsbase,
237:                                        "productdir")
238:         except KeyError:
239:             productdir = None
240:             log.debug("Unable to find productdir in registry")
241: 
242:     if not productdir or not os.path.isdir(productdir):
243:         toolskey = "VS%0.f0COMNTOOLS" % version
244:         toolsdir = os.environ.get(toolskey, None)
245: 
246:         if toolsdir and os.path.isdir(toolsdir):
247:             productdir = os.path.join(toolsdir, os.pardir, os.pardir, "VC")
248:             productdir = os.path.abspath(productdir)
249:             if not os.path.isdir(productdir):
250:                 log.debug("%s is not a valid directory" % productdir)
251:                 return None
252:         else:
253:             log.debug("Env var %s is not set or invalid" % toolskey)
254:     if not productdir:
255:         log.debug("No productdir found")
256:         return None
257:     vcvarsall = os.path.join(productdir, "vcvarsall.bat")
258:     if os.path.isfile(vcvarsall):
259:         return vcvarsall
260:     log.debug("Unable to find vcvarsall.bat")
261:     return None
262: 
263: def query_vcvarsall(version, arch="x86"):
264:     '''Launch vcvarsall.bat and read the settings from its environment
265:     '''
266:     vcvarsall = find_vcvarsall(version)
267:     interesting = set(("include", "lib", "libpath", "path"))
268:     result = {}
269: 
270:     if vcvarsall is None:
271:         raise DistutilsPlatformError("Unable to find vcvarsall.bat")
272:     log.debug("Calling 'vcvarsall.bat %s' (version=%s)", arch, version)
273:     popen = subprocess.Popen('"%s" %s & set' % (vcvarsall, arch),
274:                              stdout=subprocess.PIPE,
275:                              stderr=subprocess.PIPE)
276:     try:
277:         stdout, stderr = popen.communicate()
278:         if popen.wait() != 0:
279:             raise DistutilsPlatformError(stderr.decode("mbcs"))
280: 
281:         stdout = stdout.decode("mbcs")
282:         for line in stdout.split("\n"):
283:             line = Reg.convert_mbcs(line)
284:             if '=' not in line:
285:                 continue
286:             line = line.strip()
287:             key, value = line.split('=', 1)
288:             key = key.lower()
289:             if key in interesting:
290:                 if value.endswith(os.pathsep):
291:                     value = value[:-1]
292:                 result[key] = removeDuplicates(value)
293: 
294:     finally:
295:         popen.stdout.close()
296:         popen.stderr.close()
297: 
298:     if len(result) != len(interesting):
299:         raise ValueError(str(list(result.keys())))
300: 
301:     return result
302: 
303: # More globals
304: VERSION = get_build_version()
305: if VERSION < 8.0:
306:     raise DistutilsPlatformError("VC %0.1f is not supported by this module" % VERSION)
307: # MACROS = MacroExpander(VERSION)
308: 
309: class MSVCCompiler(CCompiler) :
310:     '''Concrete class that implements an interface to Microsoft Visual C++,
311:        as defined by the CCompiler abstract class.'''
312: 
313:     compiler_type = 'msvc'
314: 
315:     # Just set this so CCompiler's constructor doesn't barf.  We currently
316:     # don't use the 'set_executables()' bureaucracy provided by CCompiler,
317:     # as it really isn't necessary for this sort of single-compiler class.
318:     # Would be nice to have a consistent interface with UnixCCompiler,
319:     # though, so it's worth thinking about.
320:     executables = {}
321: 
322:     # Private class data (need to distinguish C from C++ source for compiler)
323:     _c_extensions = ['.c']
324:     _cpp_extensions = ['.cc', '.cpp', '.cxx']
325:     _rc_extensions = ['.rc']
326:     _mc_extensions = ['.mc']
327: 
328:     # Needed for the filename generation methods provided by the
329:     # base class, CCompiler.
330:     src_extensions = (_c_extensions + _cpp_extensions +
331:                       _rc_extensions + _mc_extensions)
332:     res_extension = '.res'
333:     obj_extension = '.obj'
334:     static_lib_extension = '.lib'
335:     shared_lib_extension = '.dll'
336:     static_lib_format = shared_lib_format = '%s%s'
337:     exe_extension = '.exe'
338: 
339:     def __init__(self, verbose=0, dry_run=0, force=0):
340:         CCompiler.__init__ (self, verbose, dry_run, force)
341:         self.__version = VERSION
342:         self.__root = r"Software\Microsoft\VisualStudio"
343:         # self.__macros = MACROS
344:         self.__paths = []
345:         # target platform (.plat_name is consistent with 'bdist')
346:         self.plat_name = None
347:         self.__arch = None # deprecated name
348:         self.initialized = False
349: 
350:     def initialize(self, plat_name=None):
351:         # multi-init means we would need to check platform same each time...
352:         assert not self.initialized, "don't init multiple times"
353:         if plat_name is None:
354:             plat_name = get_platform()
355:         # sanity check for platforms to prevent obscure errors later.
356:         ok_plats = 'win32', 'win-amd64', 'win-ia64'
357:         if plat_name not in ok_plats:
358:             raise DistutilsPlatformError("--plat-name must be one of %s" %
359:                                          (ok_plats,))
360: 
361:         if "DISTUTILS_USE_SDK" in os.environ and "MSSdk" in os.environ and self.find_exe("cl.exe"):
362:             # Assume that the SDK set up everything alright; don't try to be
363:             # smarter
364:             self.cc = "cl.exe"
365:             self.linker = "link.exe"
366:             self.lib = "lib.exe"
367:             self.rc = "rc.exe"
368:             self.mc = "mc.exe"
369:         else:
370:             # On x86, 'vcvars32.bat amd64' creates an env that doesn't work;
371:             # to cross compile, you use 'x86_amd64'.
372:             # On AMD64, 'vcvars32.bat amd64' is a native build env; to cross
373:             # compile use 'x86' (ie, it runs the x86 compiler directly)
374:             # No idea how itanium handles this, if at all.
375:             if plat_name == get_platform() or plat_name == 'win32':
376:                 # native build or cross-compile to win32
377:                 plat_spec = PLAT_TO_VCVARS[plat_name]
378:             else:
379:                 # cross compile from win32 -> some 64bit
380:                 plat_spec = PLAT_TO_VCVARS[get_platform()] + '_' + \
381:                             PLAT_TO_VCVARS[plat_name]
382: 
383:             vc_env = query_vcvarsall(VERSION, plat_spec)
384: 
385:             # take care to only use strings in the environment.
386:             self.__paths = vc_env['path'].encode('mbcs').split(os.pathsep)
387:             os.environ['lib'] = vc_env['lib'].encode('mbcs')
388:             os.environ['include'] = vc_env['include'].encode('mbcs')
389: 
390:             if len(self.__paths) == 0:
391:                 raise DistutilsPlatformError("Python was built with %s, "
392:                        "and extensions need to be built with the same "
393:                        "version of the compiler, but it isn't installed."
394:                        % self.__product)
395: 
396:             self.cc = self.find_exe("cl.exe")
397:             self.linker = self.find_exe("link.exe")
398:             self.lib = self.find_exe("lib.exe")
399:             self.rc = self.find_exe("rc.exe")   # resource compiler
400:             self.mc = self.find_exe("mc.exe")   # message compiler
401:             #self.set_path_env_var('lib')
402:             #self.set_path_env_var('include')
403: 
404:         # extend the MSVC path with the current path
405:         try:
406:             for p in os.environ['path'].split(';'):
407:                 self.__paths.append(p)
408:         except KeyError:
409:             pass
410:         self.__paths = normalize_and_reduce_paths(self.__paths)
411:         os.environ['path'] = ";".join(self.__paths)
412: 
413:         self.preprocess_options = None
414:         if self.__arch == "x86":
415:             self.compile_options = [ '/nologo', '/Ox', '/MD', '/W3',
416:                                      '/DNDEBUG']
417:             self.compile_options_debug = ['/nologo', '/Od', '/MDd', '/W3',
418:                                           '/Z7', '/D_DEBUG']
419:         else:
420:             # Win64
421:             self.compile_options = [ '/nologo', '/Ox', '/MD', '/W3', '/GS-' ,
422:                                      '/DNDEBUG']
423:             self.compile_options_debug = ['/nologo', '/Od', '/MDd', '/W3', '/GS-',
424:                                           '/Z7', '/D_DEBUG']
425: 
426:         self.ldflags_shared = ['/DLL', '/nologo', '/INCREMENTAL:NO']
427:         if self.__version >= 7:
428:             self.ldflags_shared_debug = [
429:                 '/DLL', '/nologo', '/INCREMENTAL:no', '/DEBUG'
430:                 ]
431:         self.ldflags_static = [ '/nologo']
432: 
433:         self.initialized = True
434: 
435:     # -- Worker methods ------------------------------------------------
436: 
437:     def object_filenames(self,
438:                          source_filenames,
439:                          strip_dir=0,
440:                          output_dir=''):
441:         # Copied from ccompiler.py, extended to return .res as 'object'-file
442:         # for .rc input file
443:         if output_dir is None: output_dir = ''
444:         obj_names = []
445:         for src_name in source_filenames:
446:             (base, ext) = os.path.splitext (src_name)
447:             base = os.path.splitdrive(base)[1] # Chop off the drive
448:             base = base[os.path.isabs(base):]  # If abs, chop off leading /
449:             if ext not in self.src_extensions:
450:                 # Better to raise an exception instead of silently continuing
451:                 # and later complain about sources and targets having
452:                 # different lengths
453:                 raise CompileError ("Don't know how to compile %s" % src_name)
454:             if strip_dir:
455:                 base = os.path.basename (base)
456:             if ext in self._rc_extensions:
457:                 obj_names.append (os.path.join (output_dir,
458:                                                 base + self.res_extension))
459:             elif ext in self._mc_extensions:
460:                 obj_names.append (os.path.join (output_dir,
461:                                                 base + self.res_extension))
462:             else:
463:                 obj_names.append (os.path.join (output_dir,
464:                                                 base + self.obj_extension))
465:         return obj_names
466: 
467: 
468:     def compile(self, sources,
469:                 output_dir=None, macros=None, include_dirs=None, debug=0,
470:                 extra_preargs=None, extra_postargs=None, depends=None):
471: 
472:         if not self.initialized:
473:             self.initialize()
474:         compile_info = self._setup_compile(output_dir, macros, include_dirs,
475:                                            sources, depends, extra_postargs)
476:         macros, objects, extra_postargs, pp_opts, build = compile_info
477: 
478:         compile_opts = extra_preargs or []
479:         compile_opts.append ('/c')
480:         if debug:
481:             compile_opts.extend(self.compile_options_debug)
482:         else:
483:             compile_opts.extend(self.compile_options)
484: 
485:         for obj in objects:
486:             try:
487:                 src, ext = build[obj]
488:             except KeyError:
489:                 continue
490:             if debug:
491:                 # pass the full pathname to MSVC in debug mode,
492:                 # this allows the debugger to find the source file
493:                 # without asking the user to browse for it
494:                 src = os.path.abspath(src)
495: 
496:             if ext in self._c_extensions:
497:                 input_opt = "/Tc" + src
498:             elif ext in self._cpp_extensions:
499:                 input_opt = "/Tp" + src
500:             elif ext in self._rc_extensions:
501:                 # compile .RC to .RES file
502:                 input_opt = src
503:                 output_opt = "/fo" + obj
504:                 try:
505:                     self.spawn([self.rc] + pp_opts +
506:                                [output_opt] + [input_opt])
507:                 except DistutilsExecError, msg:
508:                     raise CompileError(msg)
509:                 continue
510:             elif ext in self._mc_extensions:
511:                 # Compile .MC to .RC file to .RES file.
512:                 #   * '-h dir' specifies the directory for the
513:                 #     generated include file
514:                 #   * '-r dir' specifies the target directory of the
515:                 #     generated RC file and the binary message resource
516:                 #     it includes
517:                 #
518:                 # For now (since there are no options to change this),
519:                 # we use the source-directory for the include file and
520:                 # the build directory for the RC file and message
521:                 # resources. This works at least for win32all.
522:                 h_dir = os.path.dirname(src)
523:                 rc_dir = os.path.dirname(obj)
524:                 try:
525:                     # first compile .MC to .RC and .H file
526:                     self.spawn([self.mc] +
527:                                ['-h', h_dir, '-r', rc_dir] + [src])
528:                     base, _ = os.path.splitext (os.path.basename (src))
529:                     rc_file = os.path.join (rc_dir, base + '.rc')
530:                     # then compile .RC to .RES file
531:                     self.spawn([self.rc] +
532:                                ["/fo" + obj] + [rc_file])
533: 
534:                 except DistutilsExecError, msg:
535:                     raise CompileError(msg)
536:                 continue
537:             else:
538:                 # how to handle this file?
539:                 raise CompileError("Don't know how to compile %s to %s"
540:                                    % (src, obj))
541: 
542:             output_opt = "/Fo" + obj
543:             try:
544:                 self.spawn([self.cc] + compile_opts + pp_opts +
545:                            [input_opt, output_opt] +
546:                            extra_postargs)
547:             except DistutilsExecError, msg:
548:                 raise CompileError(msg)
549: 
550:         return objects
551: 
552: 
553:     def create_static_lib(self,
554:                           objects,
555:                           output_libname,
556:                           output_dir=None,
557:                           debug=0,
558:                           target_lang=None):
559: 
560:         if not self.initialized:
561:             self.initialize()
562:         (objects, output_dir) = self._fix_object_args(objects, output_dir)
563:         output_filename = self.library_filename(output_libname,
564:                                                 output_dir=output_dir)
565: 
566:         if self._need_link(objects, output_filename):
567:             lib_args = objects + ['/OUT:' + output_filename]
568:             if debug:
569:                 pass # XXX what goes here?
570:             try:
571:                 self.spawn([self.lib] + lib_args)
572:             except DistutilsExecError, msg:
573:                 raise LibError(msg)
574:         else:
575:             log.debug("skipping %s (up-to-date)", output_filename)
576: 
577: 
578:     def link(self,
579:              target_desc,
580:              objects,
581:              output_filename,
582:              output_dir=None,
583:              libraries=None,
584:              library_dirs=None,
585:              runtime_library_dirs=None,
586:              export_symbols=None,
587:              debug=0,
588:              extra_preargs=None,
589:              extra_postargs=None,
590:              build_temp=None,
591:              target_lang=None):
592: 
593:         if not self.initialized:
594:             self.initialize()
595:         (objects, output_dir) = self._fix_object_args(objects, output_dir)
596:         fixed_args = self._fix_lib_args(libraries, library_dirs,
597:                                         runtime_library_dirs)
598:         (libraries, library_dirs, runtime_library_dirs) = fixed_args
599: 
600:         if runtime_library_dirs:
601:             self.warn ("I don't know what to do with 'runtime_library_dirs': "
602:                        + str (runtime_library_dirs))
603: 
604:         lib_opts = gen_lib_options(self,
605:                                    library_dirs, runtime_library_dirs,
606:                                    libraries)
607:         if output_dir is not None:
608:             output_filename = os.path.join(output_dir, output_filename)
609: 
610:         if self._need_link(objects, output_filename):
611:             if target_desc == CCompiler.EXECUTABLE:
612:                 if debug:
613:                     ldflags = self.ldflags_shared_debug[1:]
614:                 else:
615:                     ldflags = self.ldflags_shared[1:]
616:             else:
617:                 if debug:
618:                     ldflags = self.ldflags_shared_debug
619:                 else:
620:                     ldflags = self.ldflags_shared
621: 
622:             export_opts = []
623:             for sym in (export_symbols or []):
624:                 export_opts.append("/EXPORT:" + sym)
625: 
626:             ld_args = (ldflags + lib_opts + export_opts +
627:                        objects + ['/OUT:' + output_filename])
628: 
629:             # The MSVC linker generates .lib and .exp files, which cannot be
630:             # suppressed by any linker switches. The .lib files may even be
631:             # needed! Make sure they are generated in the temporary build
632:             # directory. Since they have different names for debug and release
633:             # builds, they can go into the same directory.
634:             build_temp = os.path.dirname(objects[0])
635:             if export_symbols is not None:
636:                 (dll_name, dll_ext) = os.path.splitext(
637:                     os.path.basename(output_filename))
638:                 implib_file = os.path.join(
639:                     build_temp,
640:                     self.library_filename(dll_name))
641:                 ld_args.append ('/IMPLIB:' + implib_file)
642: 
643:             self.manifest_setup_ldargs(output_filename, build_temp, ld_args)
644: 
645:             if extra_preargs:
646:                 ld_args[:0] = extra_preargs
647:             if extra_postargs:
648:                 ld_args.extend(extra_postargs)
649: 
650:             self.mkpath(os.path.dirname(output_filename))
651:             try:
652:                 self.spawn([self.linker] + ld_args)
653:             except DistutilsExecError, msg:
654:                 raise LinkError(msg)
655: 
656:             # embed the manifest
657:             # XXX - this is somewhat fragile - if mt.exe fails, distutils
658:             # will still consider the DLL up-to-date, but it will not have a
659:             # manifest.  Maybe we should link to a temp file?  OTOH, that
660:             # implies a build environment error that shouldn't go undetected.
661:             mfinfo = self.manifest_get_embed_info(target_desc, ld_args)
662:             if mfinfo is not None:
663:                 mffilename, mfid = mfinfo
664:                 out_arg = '-outputresource:%s;%s' % (output_filename, mfid)
665:                 try:
666:                     self.spawn(['mt.exe', '-nologo', '-manifest',
667:                                 mffilename, out_arg])
668:                 except DistutilsExecError, msg:
669:                     raise LinkError(msg)
670:         else:
671:             log.debug("skipping %s (up-to-date)", output_filename)
672: 
673:     def manifest_setup_ldargs(self, output_filename, build_temp, ld_args):
674:         # If we need a manifest at all, an embedded manifest is recommended.
675:         # See MSDN article titled
676:         # "How to: Embed a Manifest Inside a C/C++ Application"
677:         # (currently at http://msdn2.microsoft.com/en-us/library/ms235591(VS.80).aspx)
678:         # Ask the linker to generate the manifest in the temp dir, so
679:         # we can check it, and possibly embed it, later.
680:         temp_manifest = os.path.join(
681:                 build_temp,
682:                 os.path.basename(output_filename) + ".manifest")
683:         ld_args.append('/MANIFESTFILE:' + temp_manifest)
684: 
685:     def manifest_get_embed_info(self, target_desc, ld_args):
686:         # If a manifest should be embedded, return a tuple of
687:         # (manifest_filename, resource_id).  Returns None if no manifest
688:         # should be embedded.  See http://bugs.python.org/issue7833 for why
689:         # we want to avoid any manifest for extension modules if we can)
690:         for arg in ld_args:
691:             if arg.startswith("/MANIFESTFILE:"):
692:                 temp_manifest = arg.split(":", 1)[1]
693:                 break
694:         else:
695:             # no /MANIFESTFILE so nothing to do.
696:             return None
697:         if target_desc == CCompiler.EXECUTABLE:
698:             # by default, executables always get the manifest with the
699:             # CRT referenced.
700:             mfid = 1
701:         else:
702:             # Extension modules try and avoid any manifest if possible.
703:             mfid = 2
704:             temp_manifest = self._remove_visual_c_ref(temp_manifest)
705:         if temp_manifest is None:
706:             return None
707:         return temp_manifest, mfid
708: 
709:     def _remove_visual_c_ref(self, manifest_file):
710:         try:
711:             # Remove references to the Visual C runtime, so they will
712:             # fall through to the Visual C dependency of Python.exe.
713:             # This way, when installed for a restricted user (e.g.
714:             # runtimes are not in WinSxS folder, but in Python's own
715:             # folder), the runtimes do not need to be in every folder
716:             # with .pyd's.
717:             # Returns either the filename of the modified manifest or
718:             # None if no manifest should be embedded.
719:             manifest_f = open(manifest_file)
720:             try:
721:                 manifest_buf = manifest_f.read()
722:             finally:
723:                 manifest_f.close()
724:             pattern = re.compile(
725:                 r'''<assemblyIdentity.*?name=("|')Microsoft\.'''\
726:                 r'''VC\d{2}\.CRT("|').*?(/>|</assemblyIdentity>)''',
727:                 re.DOTALL)
728:             manifest_buf = re.sub(pattern, "", manifest_buf)
729:             pattern = "<dependentAssembly>\s*</dependentAssembly>"
730:             manifest_buf = re.sub(pattern, "", manifest_buf)
731:             # Now see if any other assemblies are referenced - if not, we
732:             # don't want a manifest embedded.
733:             pattern = re.compile(
734:                 r'''<assemblyIdentity.*?name=(?:"|')(.+?)(?:"|')'''
735:                 r'''.*?(?:/>|</assemblyIdentity>)''', re.DOTALL)
736:             if re.search(pattern, manifest_buf) is None:
737:                 return None
738: 
739:             manifest_f = open(manifest_file, 'w')
740:             try:
741:                 manifest_f.write(manifest_buf)
742:                 return manifest_file
743:             finally:
744:                 manifest_f.close()
745:         except IOError:
746:             pass
747: 
748:     # -- Miscellaneous methods -----------------------------------------
749:     # These are all used by the 'gen_lib_options() function, in
750:     # ccompiler.py.
751: 
752:     def library_dir_option(self, dir):
753:         return "/LIBPATH:" + dir
754: 
755:     def runtime_library_dir_option(self, dir):
756:         raise DistutilsPlatformError(
757:               "don't know how to set runtime library search path for MSVC++")
758: 
759:     def library_option(self, lib):
760:         return self.library_filename(lib)
761: 
762: 
763:     def find_library_file(self, dirs, lib, debug=0):
764:         # Prefer a debugging library if found (and requested), but deal
765:         # with it if we don't have one.
766:         if debug:
767:             try_names = [lib + "_d", lib]
768:         else:
769:             try_names = [lib]
770:         for dir in dirs:
771:             for name in try_names:
772:                 libfile = os.path.join(dir, self.library_filename (name))
773:                 if os.path.exists(libfile):
774:                     return libfile
775:         else:
776:             # Oops, didn't find it in *any* of 'dirs'
777:             return None
778: 
779:     # Helper methods for using the MSVC registry settings
780: 
781:     def find_exe(self, exe):
782:         '''Return path to an MSVC executable program.
783: 
784:         Tries to find the program in several places: first, one of the
785:         MSVC program search paths from the registry; next, the directories
786:         in the PATH environment variable.  If any of those work, return an
787:         absolute path that is known to exist.  If none of them work, just
788:         return the original program name, 'exe'.
789:         '''
790:         for p in self.__paths:
791:             fn = os.path.join(os.path.abspath(p), exe)
792:             if os.path.isfile(fn):
793:                 return fn
794: 
795:         # didn't find it; try existing path
796:         for p in os.environ['Path'].split(';'):
797:             fn = os.path.join(os.path.abspath(p),exe)
798:             if os.path.isfile(fn):
799:                 return fn
800: 
801:         return exe
802: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_2896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, (-1)), 'str', 'distutils.msvc9compiler\n\nContains MSVCCompiler, an implementation of the abstract CCompiler class\nfor the Microsoft Visual Studio 2008.\n\nThe module is compatible with VS 2005 and VS 2008. You can find legacy support\nfor older versions of VS in distutils.msvccompiler.\n')

# Assigning a Str to a Name (line 15):

# Assigning a Str to a Name (line 15):
str_2897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), '__revision__', str_2897)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'import os' statement (line 17)
import os

import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'import subprocess' statement (line 18)
import subprocess

import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'subprocess', subprocess, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'import sys' statement (line 19)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import re' statement (line 20)
import re

import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from distutils.errors import DistutilsExecError, DistutilsPlatformError, CompileError, LibError, LinkError' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_2898 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.errors')

if (type(import_2898) is not StypyTypeError):

    if (import_2898 != 'pyd_module'):
        __import__(import_2898)
        sys_modules_2899 = sys.modules[import_2898]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.errors', sys_modules_2899.module_type_store, module_type_store, ['DistutilsExecError', 'DistutilsPlatformError', 'CompileError', 'LibError', 'LinkError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_2899, sys_modules_2899.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsExecError, DistutilsPlatformError, CompileError, LibError, LinkError

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.errors', None, module_type_store, ['DistutilsExecError', 'DistutilsPlatformError', 'CompileError', 'LibError', 'LinkError'], [DistutilsExecError, DistutilsPlatformError, CompileError, LibError, LinkError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.errors', import_2898)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from distutils.ccompiler import CCompiler, gen_lib_options' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_2900 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'distutils.ccompiler')

if (type(import_2900) is not StypyTypeError):

    if (import_2900 != 'pyd_module'):
        __import__(import_2900)
        sys_modules_2901 = sys.modules[import_2900]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'distutils.ccompiler', sys_modules_2901.module_type_store, module_type_store, ['CCompiler', 'gen_lib_options'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_2901, sys_modules_2901.module_type_store, module_type_store)
    else:
        from distutils.ccompiler import CCompiler, gen_lib_options

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'distutils.ccompiler', None, module_type_store, ['CCompiler', 'gen_lib_options'], [CCompiler, gen_lib_options])

else:
    # Assigning a type to the variable 'distutils.ccompiler' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'distutils.ccompiler', import_2900)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'from distutils import log' statement (line 25)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'distutils', None, module_type_store, ['log'], [log])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from distutils.util import get_platform' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_2902 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'distutils.util')

if (type(import_2902) is not StypyTypeError):

    if (import_2902 != 'pyd_module'):
        __import__(import_2902)
        sys_modules_2903 = sys.modules[import_2902]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'distutils.util', sys_modules_2903.module_type_store, module_type_store, ['get_platform'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_2903, sys_modules_2903.module_type_store, module_type_store)
    else:
        from distutils.util import get_platform

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'distutils.util', None, module_type_store, ['get_platform'], [get_platform])

else:
    # Assigning a type to the variable 'distutils.util' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'distutils.util', import_2902)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'import _winreg' statement (line 28)
import _winreg

import_module(stypy.reporting.localization.Localization(__file__, 28, 0), '_winreg', _winreg, module_type_store)


# Assigning a Attribute to a Name (line 30):

# Assigning a Attribute to a Name (line 30):
# Getting the type of '_winreg' (line 30)
_winreg_2904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 15), '_winreg')
# Obtaining the member 'OpenKeyEx' of a type (line 30)
OpenKeyEx_2905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 15), _winreg_2904, 'OpenKeyEx')
# Assigning a type to the variable 'RegOpenKeyEx' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'RegOpenKeyEx', OpenKeyEx_2905)

# Assigning a Attribute to a Name (line 31):

# Assigning a Attribute to a Name (line 31):
# Getting the type of '_winreg' (line 31)
_winreg_2906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 13), '_winreg')
# Obtaining the member 'EnumKey' of a type (line 31)
EnumKey_2907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 13), _winreg_2906, 'EnumKey')
# Assigning a type to the variable 'RegEnumKey' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'RegEnumKey', EnumKey_2907)

# Assigning a Attribute to a Name (line 32):

# Assigning a Attribute to a Name (line 32):
# Getting the type of '_winreg' (line 32)
_winreg_2908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), '_winreg')
# Obtaining the member 'EnumValue' of a type (line 32)
EnumValue_2909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 15), _winreg_2908, 'EnumValue')
# Assigning a type to the variable 'RegEnumValue' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'RegEnumValue', EnumValue_2909)

# Assigning a Attribute to a Name (line 33):

# Assigning a Attribute to a Name (line 33):
# Getting the type of '_winreg' (line 33)
_winreg_2910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 11), '_winreg')
# Obtaining the member 'error' of a type (line 33)
error_2911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 11), _winreg_2910, 'error')
# Assigning a type to the variable 'RegError' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'RegError', error_2911)

# Assigning a Tuple to a Name (line 35):

# Assigning a Tuple to a Name (line 35):

# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_2912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)
# Getting the type of '_winreg' (line 35)
_winreg_2913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 9), '_winreg')
# Obtaining the member 'HKEY_USERS' of a type (line 35)
HKEY_USERS_2914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 9), _winreg_2913, 'HKEY_USERS')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 9), tuple_2912, HKEY_USERS_2914)
# Adding element type (line 35)
# Getting the type of '_winreg' (line 36)
_winreg_2915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 9), '_winreg')
# Obtaining the member 'HKEY_CURRENT_USER' of a type (line 36)
HKEY_CURRENT_USER_2916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 9), _winreg_2915, 'HKEY_CURRENT_USER')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 9), tuple_2912, HKEY_CURRENT_USER_2916)
# Adding element type (line 35)
# Getting the type of '_winreg' (line 37)
_winreg_2917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 9), '_winreg')
# Obtaining the member 'HKEY_LOCAL_MACHINE' of a type (line 37)
HKEY_LOCAL_MACHINE_2918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 9), _winreg_2917, 'HKEY_LOCAL_MACHINE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 9), tuple_2912, HKEY_LOCAL_MACHINE_2918)
# Adding element type (line 35)
# Getting the type of '_winreg' (line 38)
_winreg_2919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 9), '_winreg')
# Obtaining the member 'HKEY_CLASSES_ROOT' of a type (line 38)
HKEY_CLASSES_ROOT_2920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 9), _winreg_2919, 'HKEY_CLASSES_ROOT')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 9), tuple_2912, HKEY_CLASSES_ROOT_2920)

# Assigning a type to the variable 'HKEYS' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'HKEYS', tuple_2912)

# Assigning a BoolOp to a Name (line 40):

# Assigning a BoolOp to a Name (line 40):

# Evaluating a boolean operation

# Getting the type of 'sys' (line 40)
sys_2921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'sys')
# Obtaining the member 'platform' of a type (line 40)
platform_2922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 16), sys_2921, 'platform')
str_2923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 32), 'str', 'win32')
# Applying the binary operator '==' (line 40)
result_eq_2924 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 16), '==', platform_2922, str_2923)


# Getting the type of 'sys' (line 40)
sys_2925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 44), 'sys')
# Obtaining the member 'maxsize' of a type (line 40)
maxsize_2926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 44), sys_2925, 'maxsize')
int_2927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 58), 'int')
int_2928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 61), 'int')
# Applying the binary operator '**' (line 40)
result_pow_2929 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 58), '**', int_2927, int_2928)

# Applying the binary operator '>' (line 40)
result_gt_2930 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 44), '>', maxsize_2926, result_pow_2929)

# Applying the binary operator 'and' (line 40)
result_and_keyword_2931 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 16), 'and', result_eq_2924, result_gt_2930)

# Assigning a type to the variable 'NATIVE_WIN64' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'NATIVE_WIN64', result_and_keyword_2931)

# Getting the type of 'NATIVE_WIN64' (line 41)
NATIVE_WIN64_2932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 3), 'NATIVE_WIN64')
# Testing the type of an if condition (line 41)
if_condition_2933 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 0), NATIVE_WIN64_2932)
# Assigning a type to the variable 'if_condition_2933' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'if_condition_2933', if_condition_2933)
# SSA begins for if statement (line 41)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Str to a Name (line 45):

# Assigning a Str to a Name (line 45):
str_2934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 14), 'str', 'Software\\Wow6432Node\\Microsoft\\VisualStudio\\%0.1f')
# Assigning a type to the variable 'VS_BASE' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'VS_BASE', str_2934)

# Assigning a Str to a Name (line 46):

# Assigning a Str to a Name (line 46):
str_2935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 21), 'str', 'Software\\Wow6432Node\\Microsoft\\VCExpress\\%0.1f')
# Assigning a type to the variable 'VSEXPRESS_BASE' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'VSEXPRESS_BASE', str_2935)

# Assigning a Str to a Name (line 47):

# Assigning a Str to a Name (line 47):
str_2936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 18), 'str', 'Software\\Wow6432Node\\Microsoft\\Microsoft SDKs\\Windows')
# Assigning a type to the variable 'WINSDK_BASE' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'WINSDK_BASE', str_2936)

# Assigning a Str to a Name (line 48):

# Assigning a Str to a Name (line 48):
str_2937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 15), 'str', 'Software\\Wow6432Node\\Microsoft\\.NETFramework')
# Assigning a type to the variable 'NET_BASE' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'NET_BASE', str_2937)
# SSA branch for the else part of an if statement (line 41)
module_type_store.open_ssa_branch('else')

# Assigning a Str to a Name (line 50):

# Assigning a Str to a Name (line 50):
str_2938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 14), 'str', 'Software\\Microsoft\\VisualStudio\\%0.1f')
# Assigning a type to the variable 'VS_BASE' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'VS_BASE', str_2938)

# Assigning a Str to a Name (line 51):

# Assigning a Str to a Name (line 51):
str_2939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 21), 'str', 'Software\\Microsoft\\VCExpress\\%0.1f')
# Assigning a type to the variable 'VSEXPRESS_BASE' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'VSEXPRESS_BASE', str_2939)

# Assigning a Str to a Name (line 52):

# Assigning a Str to a Name (line 52):
str_2940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 18), 'str', 'Software\\Microsoft\\Microsoft SDKs\\Windows')
# Assigning a type to the variable 'WINSDK_BASE' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'WINSDK_BASE', str_2940)

# Assigning a Str to a Name (line 53):

# Assigning a Str to a Name (line 53):
str_2941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 15), 'str', 'Software\\Microsoft\\.NETFramework')
# Assigning a type to the variable 'NET_BASE' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'NET_BASE', str_2941)
# SSA join for if statement (line 41)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Dict to a Name (line 58):

# Assigning a Dict to a Name (line 58):

# Obtaining an instance of the builtin type 'dict' (line 58)
dict_2942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 17), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 58)
# Adding element type (key, value) (line 58)
str_2943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 4), 'str', 'win32')
str_2944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 14), 'str', 'x86')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 17), dict_2942, (str_2943, str_2944))
# Adding element type (key, value) (line 58)
str_2945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 4), 'str', 'win-amd64')
str_2946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 18), 'str', 'amd64')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 17), dict_2942, (str_2945, str_2946))
# Adding element type (key, value) (line 58)
str_2947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 4), 'str', 'win-ia64')
str_2948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 17), 'str', 'ia64')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 17), dict_2942, (str_2947, str_2948))

# Assigning a type to the variable 'PLAT_TO_VCVARS' (line 58)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'PLAT_TO_VCVARS', dict_2942)
# Declaration of the 'Reg' class

class Reg:
    str_2949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, (-1)), 'str', 'Helper class to read values from the registry\n    ')

    @norecursion
    def get_value(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_value'
        module_type_store = module_type_store.open_function_context('get_value', 68, 4, False)
        # Assigning a type to the variable 'self' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Reg.get_value.__dict__.__setitem__('stypy_localization', localization)
        Reg.get_value.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Reg.get_value.__dict__.__setitem__('stypy_type_store', module_type_store)
        Reg.get_value.__dict__.__setitem__('stypy_function_name', 'Reg.get_value')
        Reg.get_value.__dict__.__setitem__('stypy_param_names_list', ['path', 'key'])
        Reg.get_value.__dict__.__setitem__('stypy_varargs_param_name', None)
        Reg.get_value.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Reg.get_value.__dict__.__setitem__('stypy_call_defaults', defaults)
        Reg.get_value.__dict__.__setitem__('stypy_call_varargs', varargs)
        Reg.get_value.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Reg.get_value.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Reg.get_value', ['path', 'key'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_value', localization, ['path', 'key'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_value(...)' code ##################

        
        # Getting the type of 'HKEYS' (line 69)
        HKEYS_2950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 20), 'HKEYS')
        # Testing the type of a for loop iterable (line 69)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 69, 8), HKEYS_2950)
        # Getting the type of the for loop variable (line 69)
        for_loop_var_2951 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 69, 8), HKEYS_2950)
        # Assigning a type to the variable 'base' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'base', for_loop_var_2951)
        # SSA begins for a for statement (line 69)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 70):
        
        # Assigning a Call to a Name (line 70):
        
        # Call to read_values(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'base' (line 70)
        base_2954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 32), 'base', False)
        # Getting the type of 'path' (line 70)
        path_2955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 38), 'path', False)
        # Processing the call keyword arguments (line 70)
        kwargs_2956 = {}
        # Getting the type of 'cls' (line 70)
        cls_2952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'cls', False)
        # Obtaining the member 'read_values' of a type (line 70)
        read_values_2953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 16), cls_2952, 'read_values')
        # Calling read_values(args, kwargs) (line 70)
        read_values_call_result_2957 = invoke(stypy.reporting.localization.Localization(__file__, 70, 16), read_values_2953, *[base_2954, path_2955], **kwargs_2956)
        
        # Assigning a type to the variable 'd' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'd', read_values_call_result_2957)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'd' (line 71)
        d_2958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 15), 'd')
        
        # Getting the type of 'key' (line 71)
        key_2959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 21), 'key')
        # Getting the type of 'd' (line 71)
        d_2960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 28), 'd')
        # Applying the binary operator 'in' (line 71)
        result_contains_2961 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 21), 'in', key_2959, d_2960)
        
        # Applying the binary operator 'and' (line 71)
        result_and_keyword_2962 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 15), 'and', d_2958, result_contains_2961)
        
        # Testing the type of an if condition (line 71)
        if_condition_2963 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 12), result_and_keyword_2962)
        # Assigning a type to the variable 'if_condition_2963' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'if_condition_2963', if_condition_2963)
        # SSA begins for if statement (line 71)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining the type of the subscript
        # Getting the type of 'key' (line 72)
        key_2964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 25), 'key')
        # Getting the type of 'd' (line 72)
        d_2965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 23), 'd')
        # Obtaining the member '__getitem__' of a type (line 72)
        getitem___2966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 23), d_2965, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 72)
        subscript_call_result_2967 = invoke(stypy.reporting.localization.Localization(__file__, 72, 23), getitem___2966, key_2964)
        
        # Assigning a type to the variable 'stypy_return_type' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'stypy_return_type', subscript_call_result_2967)
        # SSA join for if statement (line 71)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to KeyError(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'key' (line 73)
        key_2969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'key', False)
        # Processing the call keyword arguments (line 73)
        kwargs_2970 = {}
        # Getting the type of 'KeyError' (line 73)
        KeyError_2968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 14), 'KeyError', False)
        # Calling KeyError(args, kwargs) (line 73)
        KeyError_call_result_2971 = invoke(stypy.reporting.localization.Localization(__file__, 73, 14), KeyError_2968, *[key_2969], **kwargs_2970)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 73, 8), KeyError_call_result_2971, 'raise parameter', BaseException)
        
        # ################# End of 'get_value(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_value' in the type store
        # Getting the type of 'stypy_return_type' (line 68)
        stypy_return_type_2972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2972)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_value'
        return stypy_return_type_2972

    
    # Assigning a Call to a Name (line 74):

    @norecursion
    def read_keys(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'read_keys'
        module_type_store = module_type_store.open_function_context('read_keys', 76, 4, False)
        # Assigning a type to the variable 'self' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Reg.read_keys.__dict__.__setitem__('stypy_localization', localization)
        Reg.read_keys.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Reg.read_keys.__dict__.__setitem__('stypy_type_store', module_type_store)
        Reg.read_keys.__dict__.__setitem__('stypy_function_name', 'Reg.read_keys')
        Reg.read_keys.__dict__.__setitem__('stypy_param_names_list', ['base', 'key'])
        Reg.read_keys.__dict__.__setitem__('stypy_varargs_param_name', None)
        Reg.read_keys.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Reg.read_keys.__dict__.__setitem__('stypy_call_defaults', defaults)
        Reg.read_keys.__dict__.__setitem__('stypy_call_varargs', varargs)
        Reg.read_keys.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Reg.read_keys.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Reg.read_keys', ['base', 'key'], None, None, defaults, varargs, kwargs)

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

        str_2973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 8), 'str', 'Return list of registry keys.')
        
        
        # SSA begins for try-except statement (line 78)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 79):
        
        # Assigning a Call to a Name (line 79):
        
        # Call to RegOpenKeyEx(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'base' (line 79)
        base_2975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 34), 'base', False)
        # Getting the type of 'key' (line 79)
        key_2976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 40), 'key', False)
        # Processing the call keyword arguments (line 79)
        kwargs_2977 = {}
        # Getting the type of 'RegOpenKeyEx' (line 79)
        RegOpenKeyEx_2974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 21), 'RegOpenKeyEx', False)
        # Calling RegOpenKeyEx(args, kwargs) (line 79)
        RegOpenKeyEx_call_result_2978 = invoke(stypy.reporting.localization.Localization(__file__, 79, 21), RegOpenKeyEx_2974, *[base_2975, key_2976], **kwargs_2977)
        
        # Assigning a type to the variable 'handle' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'handle', RegOpenKeyEx_call_result_2978)
        # SSA branch for the except part of a try statement (line 78)
        # SSA branch for the except 'RegError' branch of a try statement (line 78)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'None' (line 81)
        None_2979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'stypy_return_type', None_2979)
        # SSA join for try-except statement (line 78)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 82):
        
        # Assigning a List to a Name (line 82):
        
        # Obtaining an instance of the builtin type 'list' (line 82)
        list_2980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 82)
        
        # Assigning a type to the variable 'L' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'L', list_2980)
        
        # Assigning a Num to a Name (line 83):
        
        # Assigning a Num to a Name (line 83):
        int_2981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 12), 'int')
        # Assigning a type to the variable 'i' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'i', int_2981)
        
        # Getting the type of 'True' (line 84)
        True_2982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 14), 'True')
        # Testing the type of an if condition (line 84)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 8), True_2982)
        # SSA begins for while statement (line 84)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        
        # SSA begins for try-except statement (line 85)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 86):
        
        # Assigning a Call to a Name (line 86):
        
        # Call to RegEnumKey(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'handle' (line 86)
        handle_2984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 31), 'handle', False)
        # Getting the type of 'i' (line 86)
        i_2985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 39), 'i', False)
        # Processing the call keyword arguments (line 86)
        kwargs_2986 = {}
        # Getting the type of 'RegEnumKey' (line 86)
        RegEnumKey_2983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 20), 'RegEnumKey', False)
        # Calling RegEnumKey(args, kwargs) (line 86)
        RegEnumKey_call_result_2987 = invoke(stypy.reporting.localization.Localization(__file__, 86, 20), RegEnumKey_2983, *[handle_2984, i_2985], **kwargs_2986)
        
        # Assigning a type to the variable 'k' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'k', RegEnumKey_call_result_2987)
        # SSA branch for the except part of a try statement (line 85)
        # SSA branch for the except 'RegError' branch of a try statement (line 85)
        module_type_store.open_ssa_branch('except')
        # SSA join for try-except statement (line 85)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'k' (line 89)
        k_2990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 21), 'k', False)
        # Processing the call keyword arguments (line 89)
        kwargs_2991 = {}
        # Getting the type of 'L' (line 89)
        L_2988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'L', False)
        # Obtaining the member 'append' of a type (line 89)
        append_2989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 12), L_2988, 'append')
        # Calling append(args, kwargs) (line 89)
        append_call_result_2992 = invoke(stypy.reporting.localization.Localization(__file__, 89, 12), append_2989, *[k_2990], **kwargs_2991)
        
        
        # Getting the type of 'i' (line 90)
        i_2993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'i')
        int_2994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 17), 'int')
        # Applying the binary operator '+=' (line 90)
        result_iadd_2995 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 12), '+=', i_2993, int_2994)
        # Assigning a type to the variable 'i' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'i', result_iadd_2995)
        
        # SSA join for while statement (line 84)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'L' (line 91)
        L_2996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 15), 'L')
        # Assigning a type to the variable 'stypy_return_type' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'stypy_return_type', L_2996)
        
        # ################# End of 'read_keys(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'read_keys' in the type store
        # Getting the type of 'stypy_return_type' (line 76)
        stypy_return_type_2997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2997)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'read_keys'
        return stypy_return_type_2997

    
    # Assigning a Call to a Name (line 92):

    @norecursion
    def read_values(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'read_values'
        module_type_store = module_type_store.open_function_context('read_values', 94, 4, False)
        # Assigning a type to the variable 'self' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Reg.read_values.__dict__.__setitem__('stypy_localization', localization)
        Reg.read_values.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Reg.read_values.__dict__.__setitem__('stypy_type_store', module_type_store)
        Reg.read_values.__dict__.__setitem__('stypy_function_name', 'Reg.read_values')
        Reg.read_values.__dict__.__setitem__('stypy_param_names_list', ['base', 'key'])
        Reg.read_values.__dict__.__setitem__('stypy_varargs_param_name', None)
        Reg.read_values.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Reg.read_values.__dict__.__setitem__('stypy_call_defaults', defaults)
        Reg.read_values.__dict__.__setitem__('stypy_call_varargs', varargs)
        Reg.read_values.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Reg.read_values.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Reg.read_values', ['base', 'key'], None, None, defaults, varargs, kwargs)

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

        str_2998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, (-1)), 'str', 'Return dict of registry keys and values.\n\n        All names are converted to lowercase.\n        ')
        
        
        # SSA begins for try-except statement (line 99)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 100):
        
        # Assigning a Call to a Name (line 100):
        
        # Call to RegOpenKeyEx(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'base' (line 100)
        base_3000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 34), 'base', False)
        # Getting the type of 'key' (line 100)
        key_3001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 40), 'key', False)
        # Processing the call keyword arguments (line 100)
        kwargs_3002 = {}
        # Getting the type of 'RegOpenKeyEx' (line 100)
        RegOpenKeyEx_2999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 21), 'RegOpenKeyEx', False)
        # Calling RegOpenKeyEx(args, kwargs) (line 100)
        RegOpenKeyEx_call_result_3003 = invoke(stypy.reporting.localization.Localization(__file__, 100, 21), RegOpenKeyEx_2999, *[base_3000, key_3001], **kwargs_3002)
        
        # Assigning a type to the variable 'handle' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'handle', RegOpenKeyEx_call_result_3003)
        # SSA branch for the except part of a try statement (line 99)
        # SSA branch for the except 'RegError' branch of a try statement (line 99)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'None' (line 102)
        None_3004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'stypy_return_type', None_3004)
        # SSA join for try-except statement (line 99)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Dict to a Name (line 103):
        
        # Assigning a Dict to a Name (line 103):
        
        # Obtaining an instance of the builtin type 'dict' (line 103)
        dict_3005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 12), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 103)
        
        # Assigning a type to the variable 'd' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'd', dict_3005)
        
        # Assigning a Num to a Name (line 104):
        
        # Assigning a Num to a Name (line 104):
        int_3006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 12), 'int')
        # Assigning a type to the variable 'i' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'i', int_3006)
        
        # Getting the type of 'True' (line 105)
        True_3007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 14), 'True')
        # Testing the type of an if condition (line 105)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 8), True_3007)
        # SSA begins for while statement (line 105)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        
        # SSA begins for try-except statement (line 106)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Tuple (line 107):
        
        # Assigning a Subscript to a Name (line 107):
        
        # Obtaining the type of the subscript
        int_3008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 16), 'int')
        
        # Call to RegEnumValue(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'handle' (line 107)
        handle_3010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 49), 'handle', False)
        # Getting the type of 'i' (line 107)
        i_3011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 57), 'i', False)
        # Processing the call keyword arguments (line 107)
        kwargs_3012 = {}
        # Getting the type of 'RegEnumValue' (line 107)
        RegEnumValue_3009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 36), 'RegEnumValue', False)
        # Calling RegEnumValue(args, kwargs) (line 107)
        RegEnumValue_call_result_3013 = invoke(stypy.reporting.localization.Localization(__file__, 107, 36), RegEnumValue_3009, *[handle_3010, i_3011], **kwargs_3012)
        
        # Obtaining the member '__getitem__' of a type (line 107)
        getitem___3014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 16), RegEnumValue_call_result_3013, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 107)
        subscript_call_result_3015 = invoke(stypy.reporting.localization.Localization(__file__, 107, 16), getitem___3014, int_3008)
        
        # Assigning a type to the variable 'tuple_var_assignment_2865' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'tuple_var_assignment_2865', subscript_call_result_3015)
        
        # Assigning a Subscript to a Name (line 107):
        
        # Obtaining the type of the subscript
        int_3016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 16), 'int')
        
        # Call to RegEnumValue(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'handle' (line 107)
        handle_3018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 49), 'handle', False)
        # Getting the type of 'i' (line 107)
        i_3019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 57), 'i', False)
        # Processing the call keyword arguments (line 107)
        kwargs_3020 = {}
        # Getting the type of 'RegEnumValue' (line 107)
        RegEnumValue_3017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 36), 'RegEnumValue', False)
        # Calling RegEnumValue(args, kwargs) (line 107)
        RegEnumValue_call_result_3021 = invoke(stypy.reporting.localization.Localization(__file__, 107, 36), RegEnumValue_3017, *[handle_3018, i_3019], **kwargs_3020)
        
        # Obtaining the member '__getitem__' of a type (line 107)
        getitem___3022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 16), RegEnumValue_call_result_3021, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 107)
        subscript_call_result_3023 = invoke(stypy.reporting.localization.Localization(__file__, 107, 16), getitem___3022, int_3016)
        
        # Assigning a type to the variable 'tuple_var_assignment_2866' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'tuple_var_assignment_2866', subscript_call_result_3023)
        
        # Assigning a Subscript to a Name (line 107):
        
        # Obtaining the type of the subscript
        int_3024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 16), 'int')
        
        # Call to RegEnumValue(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'handle' (line 107)
        handle_3026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 49), 'handle', False)
        # Getting the type of 'i' (line 107)
        i_3027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 57), 'i', False)
        # Processing the call keyword arguments (line 107)
        kwargs_3028 = {}
        # Getting the type of 'RegEnumValue' (line 107)
        RegEnumValue_3025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 36), 'RegEnumValue', False)
        # Calling RegEnumValue(args, kwargs) (line 107)
        RegEnumValue_call_result_3029 = invoke(stypy.reporting.localization.Localization(__file__, 107, 36), RegEnumValue_3025, *[handle_3026, i_3027], **kwargs_3028)
        
        # Obtaining the member '__getitem__' of a type (line 107)
        getitem___3030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 16), RegEnumValue_call_result_3029, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 107)
        subscript_call_result_3031 = invoke(stypy.reporting.localization.Localization(__file__, 107, 16), getitem___3030, int_3024)
        
        # Assigning a type to the variable 'tuple_var_assignment_2867' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'tuple_var_assignment_2867', subscript_call_result_3031)
        
        # Assigning a Name to a Name (line 107):
        # Getting the type of 'tuple_var_assignment_2865' (line 107)
        tuple_var_assignment_2865_3032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'tuple_var_assignment_2865')
        # Assigning a type to the variable 'name' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'name', tuple_var_assignment_2865_3032)
        
        # Assigning a Name to a Name (line 107):
        # Getting the type of 'tuple_var_assignment_2866' (line 107)
        tuple_var_assignment_2866_3033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'tuple_var_assignment_2866')
        # Assigning a type to the variable 'value' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 22), 'value', tuple_var_assignment_2866_3033)
        
        # Assigning a Name to a Name (line 107):
        # Getting the type of 'tuple_var_assignment_2867' (line 107)
        tuple_var_assignment_2867_3034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'tuple_var_assignment_2867')
        # Assigning a type to the variable 'type' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 29), 'type', tuple_var_assignment_2867_3034)
        # SSA branch for the except part of a try statement (line 106)
        # SSA branch for the except 'RegError' branch of a try statement (line 106)
        module_type_store.open_ssa_branch('except')
        # SSA join for try-except statement (line 106)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 110):
        
        # Assigning a Call to a Name (line 110):
        
        # Call to lower(...): (line 110)
        # Processing the call keyword arguments (line 110)
        kwargs_3037 = {}
        # Getting the type of 'name' (line 110)
        name_3035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 19), 'name', False)
        # Obtaining the member 'lower' of a type (line 110)
        lower_3036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 19), name_3035, 'lower')
        # Calling lower(args, kwargs) (line 110)
        lower_call_result_3038 = invoke(stypy.reporting.localization.Localization(__file__, 110, 19), lower_3036, *[], **kwargs_3037)
        
        # Assigning a type to the variable 'name' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'name', lower_call_result_3038)
        
        # Assigning a Call to a Subscript (line 111):
        
        # Assigning a Call to a Subscript (line 111):
        
        # Call to convert_mbcs(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'value' (line 111)
        value_3041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 57), 'value', False)
        # Processing the call keyword arguments (line 111)
        kwargs_3042 = {}
        # Getting the type of 'cls' (line 111)
        cls_3039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 40), 'cls', False)
        # Obtaining the member 'convert_mbcs' of a type (line 111)
        convert_mbcs_3040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 40), cls_3039, 'convert_mbcs')
        # Calling convert_mbcs(args, kwargs) (line 111)
        convert_mbcs_call_result_3043 = invoke(stypy.reporting.localization.Localization(__file__, 111, 40), convert_mbcs_3040, *[value_3041], **kwargs_3042)
        
        # Getting the type of 'd' (line 111)
        d_3044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'd')
        
        # Call to convert_mbcs(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'name' (line 111)
        name_3047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 31), 'name', False)
        # Processing the call keyword arguments (line 111)
        kwargs_3048 = {}
        # Getting the type of 'cls' (line 111)
        cls_3045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 14), 'cls', False)
        # Obtaining the member 'convert_mbcs' of a type (line 111)
        convert_mbcs_3046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 14), cls_3045, 'convert_mbcs')
        # Calling convert_mbcs(args, kwargs) (line 111)
        convert_mbcs_call_result_3049 = invoke(stypy.reporting.localization.Localization(__file__, 111, 14), convert_mbcs_3046, *[name_3047], **kwargs_3048)
        
        # Storing an element on a container (line 111)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 12), d_3044, (convert_mbcs_call_result_3049, convert_mbcs_call_result_3043))
        
        # Getting the type of 'i' (line 112)
        i_3050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'i')
        int_3051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 17), 'int')
        # Applying the binary operator '+=' (line 112)
        result_iadd_3052 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 12), '+=', i_3050, int_3051)
        # Assigning a type to the variable 'i' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'i', result_iadd_3052)
        
        # SSA join for while statement (line 105)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'd' (line 113)
        d_3053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 15), 'd')
        # Assigning a type to the variable 'stypy_return_type' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'stypy_return_type', d_3053)
        
        # ################# End of 'read_values(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'read_values' in the type store
        # Getting the type of 'stypy_return_type' (line 94)
        stypy_return_type_3054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3054)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'read_values'
        return stypy_return_type_3054

    
    # Assigning a Call to a Name (line 114):

    @norecursion
    def convert_mbcs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'convert_mbcs'
        module_type_store = module_type_store.open_function_context('convert_mbcs', 116, 4, False)
        # Assigning a type to the variable 'self' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Reg.convert_mbcs.__dict__.__setitem__('stypy_localization', localization)
        Reg.convert_mbcs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Reg.convert_mbcs.__dict__.__setitem__('stypy_type_store', module_type_store)
        Reg.convert_mbcs.__dict__.__setitem__('stypy_function_name', 'Reg.convert_mbcs')
        Reg.convert_mbcs.__dict__.__setitem__('stypy_param_names_list', [])
        Reg.convert_mbcs.__dict__.__setitem__('stypy_varargs_param_name', None)
        Reg.convert_mbcs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Reg.convert_mbcs.__dict__.__setitem__('stypy_call_defaults', defaults)
        Reg.convert_mbcs.__dict__.__setitem__('stypy_call_varargs', varargs)
        Reg.convert_mbcs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Reg.convert_mbcs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Reg.convert_mbcs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'convert_mbcs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'convert_mbcs(...)' code ##################

        
        # Assigning a Call to a Name (line 117):
        
        # Assigning a Call to a Name (line 117):
        
        # Call to getattr(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 's' (line 117)
        s_3056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 22), 's', False)
        str_3057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 25), 'str', 'decode')
        # Getting the type of 'None' (line 117)
        None_3058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 35), 'None', False)
        # Processing the call keyword arguments (line 117)
        kwargs_3059 = {}
        # Getting the type of 'getattr' (line 117)
        getattr_3055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 14), 'getattr', False)
        # Calling getattr(args, kwargs) (line 117)
        getattr_call_result_3060 = invoke(stypy.reporting.localization.Localization(__file__, 117, 14), getattr_3055, *[s_3056, str_3057, None_3058], **kwargs_3059)
        
        # Assigning a type to the variable 'dec' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'dec', getattr_call_result_3060)
        
        # Type idiom detected: calculating its left and rigth part (line 118)
        # Getting the type of 'dec' (line 118)
        dec_3061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'dec')
        # Getting the type of 'None' (line 118)
        None_3062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 22), 'None')
        
        (may_be_3063, more_types_in_union_3064) = may_not_be_none(dec_3061, None_3062)

        if may_be_3063:

            if more_types_in_union_3064:
                # Runtime conditional SSA (line 118)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # SSA begins for try-except statement (line 119)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Name (line 120):
            
            # Assigning a Call to a Name (line 120):
            
            # Call to dec(...): (line 120)
            # Processing the call arguments (line 120)
            str_3066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 24), 'str', 'mbcs')
            # Processing the call keyword arguments (line 120)
            kwargs_3067 = {}
            # Getting the type of 'dec' (line 120)
            dec_3065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 20), 'dec', False)
            # Calling dec(args, kwargs) (line 120)
            dec_call_result_3068 = invoke(stypy.reporting.localization.Localization(__file__, 120, 20), dec_3065, *[str_3066], **kwargs_3067)
            
            # Assigning a type to the variable 's' (line 120)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 's', dec_call_result_3068)
            # SSA branch for the except part of a try statement (line 119)
            # SSA branch for the except 'UnicodeError' branch of a try statement (line 119)
            module_type_store.open_ssa_branch('except')
            pass
            # SSA join for try-except statement (line 119)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_3064:
                # SSA join for if statement (line 118)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 's' (line 123)
        s_3069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 15), 's')
        # Assigning a type to the variable 'stypy_return_type' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'stypy_return_type', s_3069)
        
        # ################# End of 'convert_mbcs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'convert_mbcs' in the type store
        # Getting the type of 'stypy_return_type' (line 116)
        stypy_return_type_3070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3070)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'convert_mbcs'
        return stypy_return_type_3070

    
    # Assigning a Call to a Name (line 124):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 64, 0, False)
        # Assigning a type to the variable 'self' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Reg.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Reg' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'Reg', Reg)

# Assigning a Call to a Name (line 74):

# Call to classmethod(...): (line 74)
# Processing the call arguments (line 74)
# Getting the type of 'Reg'
Reg_3072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Reg', False)
# Obtaining the member 'get_value' of a type
get_value_3073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Reg_3072, 'get_value')
# Processing the call keyword arguments (line 74)
kwargs_3074 = {}
# Getting the type of 'classmethod' (line 74)
classmethod_3071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 16), 'classmethod', False)
# Calling classmethod(args, kwargs) (line 74)
classmethod_call_result_3075 = invoke(stypy.reporting.localization.Localization(__file__, 74, 16), classmethod_3071, *[get_value_3073], **kwargs_3074)

# Getting the type of 'Reg'
Reg_3076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Reg')
# Setting the type of the member 'get_value' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Reg_3076, 'get_value', classmethod_call_result_3075)

# Assigning a Call to a Name (line 92):

# Call to classmethod(...): (line 92)
# Processing the call arguments (line 92)
# Getting the type of 'Reg'
Reg_3078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Reg', False)
# Obtaining the member 'read_keys' of a type
read_keys_3079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Reg_3078, 'read_keys')
# Processing the call keyword arguments (line 92)
kwargs_3080 = {}
# Getting the type of 'classmethod' (line 92)
classmethod_3077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'classmethod', False)
# Calling classmethod(args, kwargs) (line 92)
classmethod_call_result_3081 = invoke(stypy.reporting.localization.Localization(__file__, 92, 16), classmethod_3077, *[read_keys_3079], **kwargs_3080)

# Getting the type of 'Reg'
Reg_3082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Reg')
# Setting the type of the member 'read_keys' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Reg_3082, 'read_keys', classmethod_call_result_3081)

# Assigning a Call to a Name (line 114):

# Call to classmethod(...): (line 114)
# Processing the call arguments (line 114)
# Getting the type of 'Reg'
Reg_3084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Reg', False)
# Obtaining the member 'read_values' of a type
read_values_3085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Reg_3084, 'read_values')
# Processing the call keyword arguments (line 114)
kwargs_3086 = {}
# Getting the type of 'classmethod' (line 114)
classmethod_3083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 18), 'classmethod', False)
# Calling classmethod(args, kwargs) (line 114)
classmethod_call_result_3087 = invoke(stypy.reporting.localization.Localization(__file__, 114, 18), classmethod_3083, *[read_values_3085], **kwargs_3086)

# Getting the type of 'Reg'
Reg_3088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Reg')
# Setting the type of the member 'read_values' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Reg_3088, 'read_values', classmethod_call_result_3087)

# Assigning a Call to a Name (line 124):

# Call to staticmethod(...): (line 124)
# Processing the call arguments (line 124)
# Getting the type of 'Reg'
Reg_3090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Reg', False)
# Obtaining the member 'convert_mbcs' of a type
convert_mbcs_3091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Reg_3090, 'convert_mbcs')
# Processing the call keyword arguments (line 124)
kwargs_3092 = {}
# Getting the type of 'staticmethod' (line 124)
staticmethod_3089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 19), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 124)
staticmethod_call_result_3093 = invoke(stypy.reporting.localization.Localization(__file__, 124, 19), staticmethod_3089, *[convert_mbcs_3091], **kwargs_3092)

# Getting the type of 'Reg'
Reg_3094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Reg')
# Setting the type of the member 'convert_mbcs' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Reg_3094, 'convert_mbcs', staticmethod_call_result_3093)
# Declaration of the 'MacroExpander' class

class MacroExpander:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 128, 4, False)
        # Assigning a type to the variable 'self' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'self', type_of_self)
        
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

        
        # Assigning a Dict to a Attribute (line 129):
        
        # Assigning a Dict to a Attribute (line 129):
        
        # Obtaining an instance of the builtin type 'dict' (line 129)
        dict_3095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 22), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 129)
        
        # Getting the type of 'self' (line 129)
        self_3096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'self')
        # Setting the type of the member 'macros' of a type (line 129)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), self_3096, 'macros', dict_3095)
        
        # Assigning a BinOp to a Attribute (line 130):
        
        # Assigning a BinOp to a Attribute (line 130):
        # Getting the type of 'VS_BASE' (line 130)
        VS_BASE_3097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 22), 'VS_BASE')
        # Getting the type of 'version' (line 130)
        version_3098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 32), 'version')
        # Applying the binary operator '%' (line 130)
        result_mod_3099 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 22), '%', VS_BASE_3097, version_3098)
        
        # Getting the type of 'self' (line 130)
        self_3100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'self')
        # Setting the type of the member 'vsbase' of a type (line 130)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), self_3100, 'vsbase', result_mod_3099)
        
        # Call to load_macros(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'version' (line 131)
        version_3103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 25), 'version', False)
        # Processing the call keyword arguments (line 131)
        kwargs_3104 = {}
        # Getting the type of 'self' (line 131)
        self_3101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'self', False)
        # Obtaining the member 'load_macros' of a type (line 131)
        load_macros_3102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), self_3101, 'load_macros')
        # Calling load_macros(args, kwargs) (line 131)
        load_macros_call_result_3105 = invoke(stypy.reporting.localization.Localization(__file__, 131, 8), load_macros_3102, *[version_3103], **kwargs_3104)
        
        
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
        module_type_store = module_type_store.open_function_context('set_macro', 133, 4, False)
        # Assigning a type to the variable 'self' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'self', type_of_self)
        
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

        
        # Assigning a Call to a Subscript (line 134):
        
        # Assigning a Call to a Subscript (line 134):
        
        # Call to get_value(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'path' (line 134)
        path_3108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 53), 'path', False)
        # Getting the type of 'key' (line 134)
        key_3109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 59), 'key', False)
        # Processing the call keyword arguments (line 134)
        kwargs_3110 = {}
        # Getting the type of 'Reg' (line 134)
        Reg_3106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 39), 'Reg', False)
        # Obtaining the member 'get_value' of a type (line 134)
        get_value_3107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 39), Reg_3106, 'get_value')
        # Calling get_value(args, kwargs) (line 134)
        get_value_call_result_3111 = invoke(stypy.reporting.localization.Localization(__file__, 134, 39), get_value_3107, *[path_3108, key_3109], **kwargs_3110)
        
        # Getting the type of 'self' (line 134)
        self_3112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'self')
        # Obtaining the member 'macros' of a type (line 134)
        macros_3113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), self_3112, 'macros')
        str_3114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 20), 'str', '$(%s)')
        # Getting the type of 'macro' (line 134)
        macro_3115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 30), 'macro')
        # Applying the binary operator '%' (line 134)
        result_mod_3116 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 20), '%', str_3114, macro_3115)
        
        # Storing an element on a container (line 134)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 8), macros_3113, (result_mod_3116, get_value_call_result_3111))
        
        # ################# End of 'set_macro(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_macro' in the type store
        # Getting the type of 'stypy_return_type' (line 133)
        stypy_return_type_3117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3117)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_macro'
        return stypy_return_type_3117


    @norecursion
    def load_macros(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'load_macros'
        module_type_store = module_type_store.open_function_context('load_macros', 136, 4, False)
        # Assigning a type to the variable 'self' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'self', type_of_self)
        
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

        
        # Call to set_macro(...): (line 137)
        # Processing the call arguments (line 137)
        str_3120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 23), 'str', 'VCInstallDir')
        # Getting the type of 'self' (line 137)
        self_3121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 39), 'self', False)
        # Obtaining the member 'vsbase' of a type (line 137)
        vsbase_3122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 39), self_3121, 'vsbase')
        str_3123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 53), 'str', '\\Setup\\VC')
        # Applying the binary operator '+' (line 137)
        result_add_3124 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 39), '+', vsbase_3122, str_3123)
        
        str_3125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 67), 'str', 'productdir')
        # Processing the call keyword arguments (line 137)
        kwargs_3126 = {}
        # Getting the type of 'self' (line 137)
        self_3118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'self', False)
        # Obtaining the member 'set_macro' of a type (line 137)
        set_macro_3119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), self_3118, 'set_macro')
        # Calling set_macro(args, kwargs) (line 137)
        set_macro_call_result_3127 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), set_macro_3119, *[str_3120, result_add_3124, str_3125], **kwargs_3126)
        
        
        # Call to set_macro(...): (line 138)
        # Processing the call arguments (line 138)
        str_3130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 23), 'str', 'VSInstallDir')
        # Getting the type of 'self' (line 138)
        self_3131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 39), 'self', False)
        # Obtaining the member 'vsbase' of a type (line 138)
        vsbase_3132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 39), self_3131, 'vsbase')
        str_3133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 53), 'str', '\\Setup\\VS')
        # Applying the binary operator '+' (line 138)
        result_add_3134 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 39), '+', vsbase_3132, str_3133)
        
        str_3135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 67), 'str', 'productdir')
        # Processing the call keyword arguments (line 138)
        kwargs_3136 = {}
        # Getting the type of 'self' (line 138)
        self_3128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'self', False)
        # Obtaining the member 'set_macro' of a type (line 138)
        set_macro_3129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), self_3128, 'set_macro')
        # Calling set_macro(args, kwargs) (line 138)
        set_macro_call_result_3137 = invoke(stypy.reporting.localization.Localization(__file__, 138, 8), set_macro_3129, *[str_3130, result_add_3134, str_3135], **kwargs_3136)
        
        
        # Call to set_macro(...): (line 139)
        # Processing the call arguments (line 139)
        str_3140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 23), 'str', 'FrameworkDir')
        # Getting the type of 'NET_BASE' (line 139)
        NET_BASE_3141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 39), 'NET_BASE', False)
        str_3142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 49), 'str', 'installroot')
        # Processing the call keyword arguments (line 139)
        kwargs_3143 = {}
        # Getting the type of 'self' (line 139)
        self_3138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'self', False)
        # Obtaining the member 'set_macro' of a type (line 139)
        set_macro_3139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), self_3138, 'set_macro')
        # Calling set_macro(args, kwargs) (line 139)
        set_macro_call_result_3144 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), set_macro_3139, *[str_3140, NET_BASE_3141, str_3142], **kwargs_3143)
        
        
        
        # SSA begins for try-except statement (line 140)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        
        # Getting the type of 'version' (line 141)
        version_3145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'version')
        float_3146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 26), 'float')
        # Applying the binary operator '>=' (line 141)
        result_ge_3147 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 15), '>=', version_3145, float_3146)
        
        # Testing the type of an if condition (line 141)
        if_condition_3148 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 12), result_ge_3147)
        # Assigning a type to the variable 'if_condition_3148' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'if_condition_3148', if_condition_3148)
        # SSA begins for if statement (line 141)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_macro(...): (line 142)
        # Processing the call arguments (line 142)
        str_3151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 31), 'str', 'FrameworkSDKDir')
        # Getting the type of 'NET_BASE' (line 142)
        NET_BASE_3152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 50), 'NET_BASE', False)
        str_3153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 31), 'str', 'sdkinstallrootv2.0')
        # Processing the call keyword arguments (line 142)
        kwargs_3154 = {}
        # Getting the type of 'self' (line 142)
        self_3149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 16), 'self', False)
        # Obtaining the member 'set_macro' of a type (line 142)
        set_macro_3150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 16), self_3149, 'set_macro')
        # Calling set_macro(args, kwargs) (line 142)
        set_macro_call_result_3155 = invoke(stypy.reporting.localization.Localization(__file__, 142, 16), set_macro_3150, *[str_3151, NET_BASE_3152, str_3153], **kwargs_3154)
        
        # SSA branch for the else part of an if statement (line 141)
        module_type_store.open_ssa_branch('else')
        
        # Call to KeyError(...): (line 145)
        # Processing the call arguments (line 145)
        str_3157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 31), 'str', 'sdkinstallrootv2.0')
        # Processing the call keyword arguments (line 145)
        kwargs_3158 = {}
        # Getting the type of 'KeyError' (line 145)
        KeyError_3156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 22), 'KeyError', False)
        # Calling KeyError(args, kwargs) (line 145)
        KeyError_call_result_3159 = invoke(stypy.reporting.localization.Localization(__file__, 145, 22), KeyError_3156, *[str_3157], **kwargs_3158)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 145, 16), KeyError_call_result_3159, 'raise parameter', BaseException)
        # SSA join for if statement (line 141)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the except part of a try statement (line 140)
        # SSA branch for the except 'KeyError' branch of a try statement (line 140)
        module_type_store.open_ssa_branch('except')
        
        # Call to DistutilsPlatformError(...): (line 147)
        # Processing the call arguments (line 147)
        str_3161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, (-1)), 'str', 'Python was built with Visual Studio 2008;\nextensions must be built with a compiler than can generate compatible binaries.\nVisual Studio 2008 was not found on this system. If you have Cygwin installed,\nyou can try compiling with MingW32, by passing "-c mingw32" to setup.py.')
        # Processing the call keyword arguments (line 147)
        kwargs_3162 = {}
        # Getting the type of 'DistutilsPlatformError' (line 147)
        DistutilsPlatformError_3160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 18), 'DistutilsPlatformError', False)
        # Calling DistutilsPlatformError(args, kwargs) (line 147)
        DistutilsPlatformError_call_result_3163 = invoke(stypy.reporting.localization.Localization(__file__, 147, 18), DistutilsPlatformError_3160, *[str_3161], **kwargs_3162)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 147, 12), DistutilsPlatformError_call_result_3163, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 140)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'version' (line 153)
        version_3164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 11), 'version')
        float_3165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 22), 'float')
        # Applying the binary operator '>=' (line 153)
        result_ge_3166 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 11), '>=', version_3164, float_3165)
        
        # Testing the type of an if condition (line 153)
        if_condition_3167 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 8), result_ge_3166)
        # Assigning a type to the variable 'if_condition_3167' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'if_condition_3167', if_condition_3167)
        # SSA begins for if statement (line 153)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_macro(...): (line 154)
        # Processing the call arguments (line 154)
        str_3170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 27), 'str', 'FrameworkVersion')
        # Getting the type of 'self' (line 154)
        self_3171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 47), 'self', False)
        # Obtaining the member 'vsbase' of a type (line 154)
        vsbase_3172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 47), self_3171, 'vsbase')
        str_3173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 60), 'str', 'clr version')
        # Processing the call keyword arguments (line 154)
        kwargs_3174 = {}
        # Getting the type of 'self' (line 154)
        self_3168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'self', False)
        # Obtaining the member 'set_macro' of a type (line 154)
        set_macro_3169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 12), self_3168, 'set_macro')
        # Calling set_macro(args, kwargs) (line 154)
        set_macro_call_result_3175 = invoke(stypy.reporting.localization.Localization(__file__, 154, 12), set_macro_3169, *[str_3170, vsbase_3172, str_3173], **kwargs_3174)
        
        
        # Call to set_macro(...): (line 155)
        # Processing the call arguments (line 155)
        str_3178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 27), 'str', 'WindowsSdkDir')
        # Getting the type of 'WINSDK_BASE' (line 155)
        WINSDK_BASE_3179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 44), 'WINSDK_BASE', False)
        str_3180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 57), 'str', 'currentinstallfolder')
        # Processing the call keyword arguments (line 155)
        kwargs_3181 = {}
        # Getting the type of 'self' (line 155)
        self_3176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'self', False)
        # Obtaining the member 'set_macro' of a type (line 155)
        set_macro_3177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 12), self_3176, 'set_macro')
        # Calling set_macro(args, kwargs) (line 155)
        set_macro_call_result_3182 = invoke(stypy.reporting.localization.Localization(__file__, 155, 12), set_macro_3177, *[str_3178, WINSDK_BASE_3179, str_3180], **kwargs_3181)
        
        # SSA branch for the else part of an if statement (line 153)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 157):
        
        # Assigning a Str to a Name (line 157):
        str_3183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 16), 'str', 'Software\\Microsoft\\NET Framework Setup\\Product')
        # Assigning a type to the variable 'p' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'p', str_3183)
        
        # Getting the type of 'HKEYS' (line 158)
        HKEYS_3184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 24), 'HKEYS')
        # Testing the type of a for loop iterable (line 158)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 158, 12), HKEYS_3184)
        # Getting the type of the for loop variable (line 158)
        for_loop_var_3185 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 158, 12), HKEYS_3184)
        # Assigning a type to the variable 'base' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'base', for_loop_var_3185)
        # SSA begins for a for statement (line 158)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # SSA begins for try-except statement (line 159)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 160):
        
        # Assigning a Call to a Name (line 160):
        
        # Call to RegOpenKeyEx(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'base' (line 160)
        base_3187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 37), 'base', False)
        # Getting the type of 'p' (line 160)
        p_3188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 43), 'p', False)
        # Processing the call keyword arguments (line 160)
        kwargs_3189 = {}
        # Getting the type of 'RegOpenKeyEx' (line 160)
        RegOpenKeyEx_3186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'RegOpenKeyEx', False)
        # Calling RegOpenKeyEx(args, kwargs) (line 160)
        RegOpenKeyEx_call_result_3190 = invoke(stypy.reporting.localization.Localization(__file__, 160, 24), RegOpenKeyEx_3186, *[base_3187, p_3188], **kwargs_3189)
        
        # Assigning a type to the variable 'h' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), 'h', RegOpenKeyEx_call_result_3190)
        # SSA branch for the except part of a try statement (line 159)
        # SSA branch for the except 'RegError' branch of a try statement (line 159)
        module_type_store.open_ssa_branch('except')
        # SSA join for try-except statement (line 159)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 163):
        
        # Assigning a Call to a Name (line 163):
        
        # Call to RegEnumKey(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'h' (line 163)
        h_3192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 33), 'h', False)
        int_3193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 36), 'int')
        # Processing the call keyword arguments (line 163)
        kwargs_3194 = {}
        # Getting the type of 'RegEnumKey' (line 163)
        RegEnumKey_3191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 22), 'RegEnumKey', False)
        # Calling RegEnumKey(args, kwargs) (line 163)
        RegEnumKey_call_result_3195 = invoke(stypy.reporting.localization.Localization(__file__, 163, 22), RegEnumKey_3191, *[h_3192, int_3193], **kwargs_3194)
        
        # Assigning a type to the variable 'key' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 16), 'key', RegEnumKey_call_result_3195)
        
        # Assigning a Call to a Name (line 164):
        
        # Assigning a Call to a Name (line 164):
        
        # Call to get_value(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'base' (line 164)
        base_3198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 34), 'base', False)
        str_3199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 40), 'str', '%s\\%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 164)
        tuple_3200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 164)
        # Adding element type (line 164)
        # Getting the type of 'p' (line 164)
        p_3201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 52), 'p', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 52), tuple_3200, p_3201)
        # Adding element type (line 164)
        # Getting the type of 'key' (line 164)
        key_3202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 55), 'key', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 52), tuple_3200, key_3202)
        
        # Applying the binary operator '%' (line 164)
        result_mod_3203 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 40), '%', str_3199, tuple_3200)
        
        # Processing the call keyword arguments (line 164)
        kwargs_3204 = {}
        # Getting the type of 'Reg' (line 164)
        Reg_3196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 20), 'Reg', False)
        # Obtaining the member 'get_value' of a type (line 164)
        get_value_3197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 20), Reg_3196, 'get_value')
        # Calling get_value(args, kwargs) (line 164)
        get_value_call_result_3205 = invoke(stypy.reporting.localization.Localization(__file__, 164, 20), get_value_3197, *[base_3198, result_mod_3203], **kwargs_3204)
        
        # Assigning a type to the variable 'd' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'd', get_value_call_result_3205)
        
        # Assigning a Subscript to a Subscript (line 165):
        
        # Assigning a Subscript to a Subscript (line 165):
        
        # Obtaining the type of the subscript
        str_3206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 55), 'str', 'version')
        # Getting the type of 'd' (line 165)
        d_3207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 53), 'd')
        # Obtaining the member '__getitem__' of a type (line 165)
        getitem___3208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 53), d_3207, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 165)
        subscript_call_result_3209 = invoke(stypy.reporting.localization.Localization(__file__, 165, 53), getitem___3208, str_3206)
        
        # Getting the type of 'self' (line 165)
        self_3210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'self')
        # Obtaining the member 'macros' of a type (line 165)
        macros_3211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 16), self_3210, 'macros')
        str_3212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 28), 'str', '$(FrameworkVersion)')
        # Storing an element on a container (line 165)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 16), macros_3211, (str_3212, subscript_call_result_3209))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 153)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'load_macros(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'load_macros' in the type store
        # Getting the type of 'stypy_return_type' (line 136)
        stypy_return_type_3213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3213)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'load_macros'
        return stypy_return_type_3213


    @norecursion
    def sub(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'sub'
        module_type_store = module_type_store.open_function_context('sub', 167, 4, False)
        # Assigning a type to the variable 'self' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'self', type_of_self)
        
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

        
        
        # Call to items(...): (line 168)
        # Processing the call keyword arguments (line 168)
        kwargs_3217 = {}
        # Getting the type of 'self' (line 168)
        self_3214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 20), 'self', False)
        # Obtaining the member 'macros' of a type (line 168)
        macros_3215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 20), self_3214, 'macros')
        # Obtaining the member 'items' of a type (line 168)
        items_3216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 20), macros_3215, 'items')
        # Calling items(args, kwargs) (line 168)
        items_call_result_3218 = invoke(stypy.reporting.localization.Localization(__file__, 168, 20), items_3216, *[], **kwargs_3217)
        
        # Testing the type of a for loop iterable (line 168)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 168, 8), items_call_result_3218)
        # Getting the type of the for loop variable (line 168)
        for_loop_var_3219 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 168, 8), items_call_result_3218)
        # Assigning a type to the variable 'k' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), for_loop_var_3219))
        # Assigning a type to the variable 'v' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), for_loop_var_3219))
        # SSA begins for a for statement (line 168)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 169):
        
        # Assigning a Call to a Name (line 169):
        
        # Call to replace(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'k' (line 169)
        k_3222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 26), 'k', False)
        # Getting the type of 'v' (line 169)
        v_3223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 29), 'v', False)
        # Processing the call keyword arguments (line 169)
        kwargs_3224 = {}
        # Getting the type of 's' (line 169)
        s_3220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 's', False)
        # Obtaining the member 'replace' of a type (line 169)
        replace_3221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 16), s_3220, 'replace')
        # Calling replace(args, kwargs) (line 169)
        replace_call_result_3225 = invoke(stypy.reporting.localization.Localization(__file__, 169, 16), replace_3221, *[k_3222, v_3223], **kwargs_3224)
        
        # Assigning a type to the variable 's' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 's', replace_call_result_3225)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 's' (line 170)
        s_3226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 15), 's')
        # Assigning a type to the variable 'stypy_return_type' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'stypy_return_type', s_3226)
        
        # ################# End of 'sub(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sub' in the type store
        # Getting the type of 'stypy_return_type' (line 167)
        stypy_return_type_3227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3227)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sub'
        return stypy_return_type_3227


# Assigning a type to the variable 'MacroExpander' (line 126)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 0), 'MacroExpander', MacroExpander)

@norecursion
def get_build_version(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_build_version'
    module_type_store = module_type_store.open_function_context('get_build_version', 172, 0, False)
    
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

    str_3228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, (-1)), 'str', 'Return the version of MSVC that was used to build Python.\n\n    For Python 2.3 and up, the version number is included in\n    sys.version.  For earlier versions, assume the compiler is MSVC 6.\n    ')
    
    # Assigning a Str to a Name (line 178):
    
    # Assigning a Str to a Name (line 178):
    str_3229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 13), 'str', 'MSC v.')
    # Assigning a type to the variable 'prefix' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'prefix', str_3229)
    
    # Assigning a Call to a Name (line 179):
    
    # Assigning a Call to a Name (line 179):
    
    # Call to find(...): (line 179)
    # Processing the call arguments (line 179)
    # Getting the type of 'prefix' (line 179)
    prefix_3233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 25), 'prefix', False)
    # Processing the call keyword arguments (line 179)
    kwargs_3234 = {}
    # Getting the type of 'sys' (line 179)
    sys_3230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'sys', False)
    # Obtaining the member 'version' of a type (line 179)
    version_3231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), sys_3230, 'version')
    # Obtaining the member 'find' of a type (line 179)
    find_3232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), version_3231, 'find')
    # Calling find(args, kwargs) (line 179)
    find_call_result_3235 = invoke(stypy.reporting.localization.Localization(__file__, 179, 8), find_3232, *[prefix_3233], **kwargs_3234)
    
    # Assigning a type to the variable 'i' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'i', find_call_result_3235)
    
    
    # Getting the type of 'i' (line 180)
    i_3236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 7), 'i')
    int_3237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 12), 'int')
    # Applying the binary operator '==' (line 180)
    result_eq_3238 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 7), '==', i_3236, int_3237)
    
    # Testing the type of an if condition (line 180)
    if_condition_3239 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 180, 4), result_eq_3238)
    # Assigning a type to the variable 'if_condition_3239' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'if_condition_3239', if_condition_3239)
    # SSA begins for if statement (line 180)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_3240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'stypy_return_type', int_3240)
    # SSA join for if statement (line 180)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 182):
    
    # Assigning a BinOp to a Name (line 182):
    # Getting the type of 'i' (line 182)
    i_3241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'i')
    
    # Call to len(...): (line 182)
    # Processing the call arguments (line 182)
    # Getting the type of 'prefix' (line 182)
    prefix_3243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 16), 'prefix', False)
    # Processing the call keyword arguments (line 182)
    kwargs_3244 = {}
    # Getting the type of 'len' (line 182)
    len_3242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'len', False)
    # Calling len(args, kwargs) (line 182)
    len_call_result_3245 = invoke(stypy.reporting.localization.Localization(__file__, 182, 12), len_3242, *[prefix_3243], **kwargs_3244)
    
    # Applying the binary operator '+' (line 182)
    result_add_3246 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 8), '+', i_3241, len_call_result_3245)
    
    # Assigning a type to the variable 'i' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'i', result_add_3246)
    
    # Assigning a Call to a Tuple (line 183):
    
    # Assigning a Subscript to a Name (line 183):
    
    # Obtaining the type of the subscript
    int_3247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 4), 'int')
    
    # Call to split(...): (line 183)
    # Processing the call arguments (line 183)
    str_3255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 36), 'str', ' ')
    int_3256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 41), 'int')
    # Processing the call keyword arguments (line 183)
    kwargs_3257 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 183)
    i_3248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 26), 'i', False)
    slice_3249 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 183, 14), i_3248, None, None)
    # Getting the type of 'sys' (line 183)
    sys_3250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 14), 'sys', False)
    # Obtaining the member 'version' of a type (line 183)
    version_3251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 14), sys_3250, 'version')
    # Obtaining the member '__getitem__' of a type (line 183)
    getitem___3252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 14), version_3251, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 183)
    subscript_call_result_3253 = invoke(stypy.reporting.localization.Localization(__file__, 183, 14), getitem___3252, slice_3249)
    
    # Obtaining the member 'split' of a type (line 183)
    split_3254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 14), subscript_call_result_3253, 'split')
    # Calling split(args, kwargs) (line 183)
    split_call_result_3258 = invoke(stypy.reporting.localization.Localization(__file__, 183, 14), split_3254, *[str_3255, int_3256], **kwargs_3257)
    
    # Obtaining the member '__getitem__' of a type (line 183)
    getitem___3259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 4), split_call_result_3258, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 183)
    subscript_call_result_3260 = invoke(stypy.reporting.localization.Localization(__file__, 183, 4), getitem___3259, int_3247)
    
    # Assigning a type to the variable 'tuple_var_assignment_2868' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'tuple_var_assignment_2868', subscript_call_result_3260)
    
    # Assigning a Subscript to a Name (line 183):
    
    # Obtaining the type of the subscript
    int_3261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 4), 'int')
    
    # Call to split(...): (line 183)
    # Processing the call arguments (line 183)
    str_3269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 36), 'str', ' ')
    int_3270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 41), 'int')
    # Processing the call keyword arguments (line 183)
    kwargs_3271 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 183)
    i_3262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 26), 'i', False)
    slice_3263 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 183, 14), i_3262, None, None)
    # Getting the type of 'sys' (line 183)
    sys_3264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 14), 'sys', False)
    # Obtaining the member 'version' of a type (line 183)
    version_3265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 14), sys_3264, 'version')
    # Obtaining the member '__getitem__' of a type (line 183)
    getitem___3266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 14), version_3265, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 183)
    subscript_call_result_3267 = invoke(stypy.reporting.localization.Localization(__file__, 183, 14), getitem___3266, slice_3263)
    
    # Obtaining the member 'split' of a type (line 183)
    split_3268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 14), subscript_call_result_3267, 'split')
    # Calling split(args, kwargs) (line 183)
    split_call_result_3272 = invoke(stypy.reporting.localization.Localization(__file__, 183, 14), split_3268, *[str_3269, int_3270], **kwargs_3271)
    
    # Obtaining the member '__getitem__' of a type (line 183)
    getitem___3273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 4), split_call_result_3272, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 183)
    subscript_call_result_3274 = invoke(stypy.reporting.localization.Localization(__file__, 183, 4), getitem___3273, int_3261)
    
    # Assigning a type to the variable 'tuple_var_assignment_2869' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'tuple_var_assignment_2869', subscript_call_result_3274)
    
    # Assigning a Name to a Name (line 183):
    # Getting the type of 'tuple_var_assignment_2868' (line 183)
    tuple_var_assignment_2868_3275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'tuple_var_assignment_2868')
    # Assigning a type to the variable 's' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 's', tuple_var_assignment_2868_3275)
    
    # Assigning a Name to a Name (line 183):
    # Getting the type of 'tuple_var_assignment_2869' (line 183)
    tuple_var_assignment_2869_3276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'tuple_var_assignment_2869')
    # Assigning a type to the variable 'rest' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 7), 'rest', tuple_var_assignment_2869_3276)
    
    # Assigning a BinOp to a Name (line 184):
    
    # Assigning a BinOp to a Name (line 184):
    
    # Call to int(...): (line 184)
    # Processing the call arguments (line 184)
    
    # Obtaining the type of the subscript
    int_3278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 26), 'int')
    slice_3279 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 184, 23), None, int_3278, None)
    # Getting the type of 's' (line 184)
    s_3280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 23), 's', False)
    # Obtaining the member '__getitem__' of a type (line 184)
    getitem___3281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 23), s_3280, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 184)
    subscript_call_result_3282 = invoke(stypy.reporting.localization.Localization(__file__, 184, 23), getitem___3281, slice_3279)
    
    # Processing the call keyword arguments (line 184)
    kwargs_3283 = {}
    # Getting the type of 'int' (line 184)
    int_3277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 19), 'int', False)
    # Calling int(args, kwargs) (line 184)
    int_call_result_3284 = invoke(stypy.reporting.localization.Localization(__file__, 184, 19), int_3277, *[subscript_call_result_3282], **kwargs_3283)
    
    int_3285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 33), 'int')
    # Applying the binary operator '-' (line 184)
    result_sub_3286 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 19), '-', int_call_result_3284, int_3285)
    
    # Assigning a type to the variable 'majorVersion' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'majorVersion', result_sub_3286)
    
    # Assigning a BinOp to a Name (line 185):
    
    # Assigning a BinOp to a Name (line 185):
    
    # Call to int(...): (line 185)
    # Processing the call arguments (line 185)
    
    # Obtaining the type of the subscript
    int_3288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 25), 'int')
    int_3289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 27), 'int')
    slice_3290 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 185, 23), int_3288, int_3289, None)
    # Getting the type of 's' (line 185)
    s_3291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 23), 's', False)
    # Obtaining the member '__getitem__' of a type (line 185)
    getitem___3292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 23), s_3291, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 185)
    subscript_call_result_3293 = invoke(stypy.reporting.localization.Localization(__file__, 185, 23), getitem___3292, slice_3290)
    
    # Processing the call keyword arguments (line 185)
    kwargs_3294 = {}
    # Getting the type of 'int' (line 185)
    int_3287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 19), 'int', False)
    # Calling int(args, kwargs) (line 185)
    int_call_result_3295 = invoke(stypy.reporting.localization.Localization(__file__, 185, 19), int_3287, *[subscript_call_result_3293], **kwargs_3294)
    
    float_3296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 33), 'float')
    # Applying the binary operator 'div' (line 185)
    result_div_3297 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 19), 'div', int_call_result_3295, float_3296)
    
    # Assigning a type to the variable 'minorVersion' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'minorVersion', result_div_3297)
    
    
    # Getting the type of 'majorVersion' (line 187)
    majorVersion_3298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 7), 'majorVersion')
    int_3299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 23), 'int')
    # Applying the binary operator '==' (line 187)
    result_eq_3300 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 7), '==', majorVersion_3298, int_3299)
    
    # Testing the type of an if condition (line 187)
    if_condition_3301 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 187, 4), result_eq_3300)
    # Assigning a type to the variable 'if_condition_3301' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'if_condition_3301', if_condition_3301)
    # SSA begins for if statement (line 187)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 188):
    
    # Assigning a Num to a Name (line 188):
    int_3302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 23), 'int')
    # Assigning a type to the variable 'minorVersion' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'minorVersion', int_3302)
    # SSA join for if statement (line 187)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'majorVersion' (line 189)
    majorVersion_3303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 7), 'majorVersion')
    int_3304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 23), 'int')
    # Applying the binary operator '>=' (line 189)
    result_ge_3305 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 7), '>=', majorVersion_3303, int_3304)
    
    # Testing the type of an if condition (line 189)
    if_condition_3306 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 189, 4), result_ge_3305)
    # Assigning a type to the variable 'if_condition_3306' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'if_condition_3306', if_condition_3306)
    # SSA begins for if statement (line 189)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'majorVersion' (line 190)
    majorVersion_3307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 15), 'majorVersion')
    # Getting the type of 'minorVersion' (line 190)
    minorVersion_3308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 30), 'minorVersion')
    # Applying the binary operator '+' (line 190)
    result_add_3309 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 15), '+', majorVersion_3307, minorVersion_3308)
    
    # Assigning a type to the variable 'stypy_return_type' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'stypy_return_type', result_add_3309)
    # SSA join for if statement (line 189)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'None' (line 192)
    None_3310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 11), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'stypy_return_type', None_3310)
    
    # ################# End of 'get_build_version(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_build_version' in the type store
    # Getting the type of 'stypy_return_type' (line 172)
    stypy_return_type_3311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3311)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_build_version'
    return stypy_return_type_3311

# Assigning a type to the variable 'get_build_version' (line 172)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 0), 'get_build_version', get_build_version)

@norecursion
def normalize_and_reduce_paths(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'normalize_and_reduce_paths'
    module_type_store = module_type_store.open_function_context('normalize_and_reduce_paths', 194, 0, False)
    
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

    str_3312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, (-1)), 'str', 'Return a list of normalized paths with duplicates removed.\n\n    The current order of paths is maintained.\n    ')
    
    # Assigning a List to a Name (line 200):
    
    # Assigning a List to a Name (line 200):
    
    # Obtaining an instance of the builtin type 'list' (line 200)
    list_3313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 200)
    
    # Assigning a type to the variable 'reduced_paths' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'reduced_paths', list_3313)
    
    # Getting the type of 'paths' (line 201)
    paths_3314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 13), 'paths')
    # Testing the type of a for loop iterable (line 201)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 201, 4), paths_3314)
    # Getting the type of the for loop variable (line 201)
    for_loop_var_3315 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 201, 4), paths_3314)
    # Assigning a type to the variable 'p' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'p', for_loop_var_3315)
    # SSA begins for a for statement (line 201)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 202):
    
    # Assigning a Call to a Name (line 202):
    
    # Call to normpath(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'p' (line 202)
    p_3319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 30), 'p', False)
    # Processing the call keyword arguments (line 202)
    kwargs_3320 = {}
    # Getting the type of 'os' (line 202)
    os_3316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 13), 'os', False)
    # Obtaining the member 'path' of a type (line 202)
    path_3317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 13), os_3316, 'path')
    # Obtaining the member 'normpath' of a type (line 202)
    normpath_3318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 13), path_3317, 'normpath')
    # Calling normpath(args, kwargs) (line 202)
    normpath_call_result_3321 = invoke(stypy.reporting.localization.Localization(__file__, 202, 13), normpath_3318, *[p_3319], **kwargs_3320)
    
    # Assigning a type to the variable 'np' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'np', normpath_call_result_3321)
    
    
    # Getting the type of 'np' (line 204)
    np_3322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 11), 'np')
    # Getting the type of 'reduced_paths' (line 204)
    reduced_paths_3323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 21), 'reduced_paths')
    # Applying the binary operator 'notin' (line 204)
    result_contains_3324 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 11), 'notin', np_3322, reduced_paths_3323)
    
    # Testing the type of an if condition (line 204)
    if_condition_3325 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 204, 8), result_contains_3324)
    # Assigning a type to the variable 'if_condition_3325' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'if_condition_3325', if_condition_3325)
    # SSA begins for if statement (line 204)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'np' (line 205)
    np_3328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 33), 'np', False)
    # Processing the call keyword arguments (line 205)
    kwargs_3329 = {}
    # Getting the type of 'reduced_paths' (line 205)
    reduced_paths_3326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'reduced_paths', False)
    # Obtaining the member 'append' of a type (line 205)
    append_3327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 12), reduced_paths_3326, 'append')
    # Calling append(args, kwargs) (line 205)
    append_call_result_3330 = invoke(stypy.reporting.localization.Localization(__file__, 205, 12), append_3327, *[np_3328], **kwargs_3329)
    
    # SSA join for if statement (line 204)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'reduced_paths' (line 206)
    reduced_paths_3331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 11), 'reduced_paths')
    # Assigning a type to the variable 'stypy_return_type' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'stypy_return_type', reduced_paths_3331)
    
    # ################# End of 'normalize_and_reduce_paths(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'normalize_and_reduce_paths' in the type store
    # Getting the type of 'stypy_return_type' (line 194)
    stypy_return_type_3332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3332)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'normalize_and_reduce_paths'
    return stypy_return_type_3332

# Assigning a type to the variable 'normalize_and_reduce_paths' (line 194)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 0), 'normalize_and_reduce_paths', normalize_and_reduce_paths)

@norecursion
def removeDuplicates(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'removeDuplicates'
    module_type_store = module_type_store.open_function_context('removeDuplicates', 208, 0, False)
    
    # Passed parameters checking function
    removeDuplicates.stypy_localization = localization
    removeDuplicates.stypy_type_of_self = None
    removeDuplicates.stypy_type_store = module_type_store
    removeDuplicates.stypy_function_name = 'removeDuplicates'
    removeDuplicates.stypy_param_names_list = ['variable']
    removeDuplicates.stypy_varargs_param_name = None
    removeDuplicates.stypy_kwargs_param_name = None
    removeDuplicates.stypy_call_defaults = defaults
    removeDuplicates.stypy_call_varargs = varargs
    removeDuplicates.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'removeDuplicates', ['variable'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'removeDuplicates', localization, ['variable'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'removeDuplicates(...)' code ##################

    str_3333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, (-1)), 'str', 'Remove duplicate values of an environment variable.\n    ')
    
    # Assigning a Call to a Name (line 211):
    
    # Assigning a Call to a Name (line 211):
    
    # Call to split(...): (line 211)
    # Processing the call arguments (line 211)
    # Getting the type of 'os' (line 211)
    os_3336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 29), 'os', False)
    # Obtaining the member 'pathsep' of a type (line 211)
    pathsep_3337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 29), os_3336, 'pathsep')
    # Processing the call keyword arguments (line 211)
    kwargs_3338 = {}
    # Getting the type of 'variable' (line 211)
    variable_3334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 14), 'variable', False)
    # Obtaining the member 'split' of a type (line 211)
    split_3335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 14), variable_3334, 'split')
    # Calling split(args, kwargs) (line 211)
    split_call_result_3339 = invoke(stypy.reporting.localization.Localization(__file__, 211, 14), split_3335, *[pathsep_3337], **kwargs_3338)
    
    # Assigning a type to the variable 'oldList' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'oldList', split_call_result_3339)
    
    # Assigning a List to a Name (line 212):
    
    # Assigning a List to a Name (line 212):
    
    # Obtaining an instance of the builtin type 'list' (line 212)
    list_3340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 212)
    
    # Assigning a type to the variable 'newList' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'newList', list_3340)
    
    # Getting the type of 'oldList' (line 213)
    oldList_3341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 13), 'oldList')
    # Testing the type of a for loop iterable (line 213)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 213, 4), oldList_3341)
    # Getting the type of the for loop variable (line 213)
    for_loop_var_3342 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 213, 4), oldList_3341)
    # Assigning a type to the variable 'i' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'i', for_loop_var_3342)
    # SSA begins for a for statement (line 213)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'i' (line 214)
    i_3343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 11), 'i')
    # Getting the type of 'newList' (line 214)
    newList_3344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 20), 'newList')
    # Applying the binary operator 'notin' (line 214)
    result_contains_3345 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 11), 'notin', i_3343, newList_3344)
    
    # Testing the type of an if condition (line 214)
    if_condition_3346 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 214, 8), result_contains_3345)
    # Assigning a type to the variable 'if_condition_3346' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'if_condition_3346', if_condition_3346)
    # SSA begins for if statement (line 214)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 215)
    # Processing the call arguments (line 215)
    # Getting the type of 'i' (line 215)
    i_3349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 27), 'i', False)
    # Processing the call keyword arguments (line 215)
    kwargs_3350 = {}
    # Getting the type of 'newList' (line 215)
    newList_3347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'newList', False)
    # Obtaining the member 'append' of a type (line 215)
    append_3348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 12), newList_3347, 'append')
    # Calling append(args, kwargs) (line 215)
    append_call_result_3351 = invoke(stypy.reporting.localization.Localization(__file__, 215, 12), append_3348, *[i_3349], **kwargs_3350)
    
    # SSA join for if statement (line 214)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 216):
    
    # Assigning a Call to a Name (line 216):
    
    # Call to join(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'newList' (line 216)
    newList_3355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 34), 'newList', False)
    # Processing the call keyword arguments (line 216)
    kwargs_3356 = {}
    # Getting the type of 'os' (line 216)
    os_3352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 18), 'os', False)
    # Obtaining the member 'pathsep' of a type (line 216)
    pathsep_3353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 18), os_3352, 'pathsep')
    # Obtaining the member 'join' of a type (line 216)
    join_3354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 18), pathsep_3353, 'join')
    # Calling join(args, kwargs) (line 216)
    join_call_result_3357 = invoke(stypy.reporting.localization.Localization(__file__, 216, 18), join_3354, *[newList_3355], **kwargs_3356)
    
    # Assigning a type to the variable 'newVariable' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'newVariable', join_call_result_3357)
    # Getting the type of 'newVariable' (line 217)
    newVariable_3358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 11), 'newVariable')
    # Assigning a type to the variable 'stypy_return_type' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'stypy_return_type', newVariable_3358)
    
    # ################# End of 'removeDuplicates(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'removeDuplicates' in the type store
    # Getting the type of 'stypy_return_type' (line 208)
    stypy_return_type_3359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3359)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'removeDuplicates'
    return stypy_return_type_3359

# Assigning a type to the variable 'removeDuplicates' (line 208)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 0), 'removeDuplicates', removeDuplicates)

@norecursion
def find_vcvarsall(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'find_vcvarsall'
    module_type_store = module_type_store.open_function_context('find_vcvarsall', 219, 0, False)
    
    # Passed parameters checking function
    find_vcvarsall.stypy_localization = localization
    find_vcvarsall.stypy_type_of_self = None
    find_vcvarsall.stypy_type_store = module_type_store
    find_vcvarsall.stypy_function_name = 'find_vcvarsall'
    find_vcvarsall.stypy_param_names_list = ['version']
    find_vcvarsall.stypy_varargs_param_name = None
    find_vcvarsall.stypy_kwargs_param_name = None
    find_vcvarsall.stypy_call_defaults = defaults
    find_vcvarsall.stypy_call_varargs = varargs
    find_vcvarsall.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'find_vcvarsall', ['version'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'find_vcvarsall', localization, ['version'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'find_vcvarsall(...)' code ##################

    str_3360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, (-1)), 'str', 'Find the vcvarsall.bat file\n\n    At first it tries to find the productdir of VS 2008 in the registry. If\n    that fails it falls back to the VS90COMNTOOLS env var.\n    ')
    
    # Assigning a BinOp to a Name (line 225):
    
    # Assigning a BinOp to a Name (line 225):
    # Getting the type of 'VS_BASE' (line 225)
    VS_BASE_3361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 13), 'VS_BASE')
    # Getting the type of 'version' (line 225)
    version_3362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 23), 'version')
    # Applying the binary operator '%' (line 225)
    result_mod_3363 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 13), '%', VS_BASE_3361, version_3362)
    
    # Assigning a type to the variable 'vsbase' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'vsbase', result_mod_3363)
    
    
    # SSA begins for try-except statement (line 226)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 227):
    
    # Assigning a Call to a Name (line 227):
    
    # Call to get_value(...): (line 227)
    # Processing the call arguments (line 227)
    str_3366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 35), 'str', '%s\\Setup\\VC')
    # Getting the type of 'vsbase' (line 227)
    vsbase_3367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 52), 'vsbase', False)
    # Applying the binary operator '%' (line 227)
    result_mod_3368 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 35), '%', str_3366, vsbase_3367)
    
    str_3369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 35), 'str', 'productdir')
    # Processing the call keyword arguments (line 227)
    kwargs_3370 = {}
    # Getting the type of 'Reg' (line 227)
    Reg_3364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 21), 'Reg', False)
    # Obtaining the member 'get_value' of a type (line 227)
    get_value_3365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 21), Reg_3364, 'get_value')
    # Calling get_value(args, kwargs) (line 227)
    get_value_call_result_3371 = invoke(stypy.reporting.localization.Localization(__file__, 227, 21), get_value_3365, *[result_mod_3368, str_3369], **kwargs_3370)
    
    # Assigning a type to the variable 'productdir' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'productdir', get_value_call_result_3371)
    # SSA branch for the except part of a try statement (line 226)
    # SSA branch for the except 'KeyError' branch of a try statement (line 226)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 230):
    
    # Assigning a Name to a Name (line 230):
    # Getting the type of 'None' (line 230)
    None_3372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 21), 'None')
    # Assigning a type to the variable 'productdir' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'productdir', None_3372)
    # SSA join for try-except statement (line 226)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 233)
    # Getting the type of 'productdir' (line 233)
    productdir_3373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 7), 'productdir')
    # Getting the type of 'None' (line 233)
    None_3374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 21), 'None')
    
    (may_be_3375, more_types_in_union_3376) = may_be_none(productdir_3373, None_3374)

    if may_be_3375:

        if more_types_in_union_3376:
            # Runtime conditional SSA (line 233)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 234):
        
        # Assigning a BinOp to a Name (line 234):
        # Getting the type of 'VSEXPRESS_BASE' (line 234)
        VSEXPRESS_BASE_3377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 17), 'VSEXPRESS_BASE')
        # Getting the type of 'version' (line 234)
        version_3378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 34), 'version')
        # Applying the binary operator '%' (line 234)
        result_mod_3379 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 17), '%', VSEXPRESS_BASE_3377, version_3378)
        
        # Assigning a type to the variable 'vsbase' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'vsbase', result_mod_3379)
        
        
        # SSA begins for try-except statement (line 235)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 236):
        
        # Assigning a Call to a Name (line 236):
        
        # Call to get_value(...): (line 236)
        # Processing the call arguments (line 236)
        str_3382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 39), 'str', '%s\\Setup\\VC')
        # Getting the type of 'vsbase' (line 236)
        vsbase_3383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 56), 'vsbase', False)
        # Applying the binary operator '%' (line 236)
        result_mod_3384 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 39), '%', str_3382, vsbase_3383)
        
        str_3385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 39), 'str', 'productdir')
        # Processing the call keyword arguments (line 236)
        kwargs_3386 = {}
        # Getting the type of 'Reg' (line 236)
        Reg_3380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 25), 'Reg', False)
        # Obtaining the member 'get_value' of a type (line 236)
        get_value_3381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 25), Reg_3380, 'get_value')
        # Calling get_value(args, kwargs) (line 236)
        get_value_call_result_3387 = invoke(stypy.reporting.localization.Localization(__file__, 236, 25), get_value_3381, *[result_mod_3384, str_3385], **kwargs_3386)
        
        # Assigning a type to the variable 'productdir' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'productdir', get_value_call_result_3387)
        # SSA branch for the except part of a try statement (line 235)
        # SSA branch for the except 'KeyError' branch of a try statement (line 235)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Name to a Name (line 239):
        
        # Assigning a Name to a Name (line 239):
        # Getting the type of 'None' (line 239)
        None_3388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 25), 'None')
        # Assigning a type to the variable 'productdir' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'productdir', None_3388)
        
        # Call to debug(...): (line 240)
        # Processing the call arguments (line 240)
        str_3391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 22), 'str', 'Unable to find productdir in registry')
        # Processing the call keyword arguments (line 240)
        kwargs_3392 = {}
        # Getting the type of 'log' (line 240)
        log_3389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'log', False)
        # Obtaining the member 'debug' of a type (line 240)
        debug_3390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 12), log_3389, 'debug')
        # Calling debug(args, kwargs) (line 240)
        debug_call_result_3393 = invoke(stypy.reporting.localization.Localization(__file__, 240, 12), debug_3390, *[str_3391], **kwargs_3392)
        
        # SSA join for try-except statement (line 235)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_3376:
            # SSA join for if statement (line 233)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'productdir' (line 242)
    productdir_3394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 11), 'productdir')
    # Applying the 'not' unary operator (line 242)
    result_not__3395 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 7), 'not', productdir_3394)
    
    
    
    # Call to isdir(...): (line 242)
    # Processing the call arguments (line 242)
    # Getting the type of 'productdir' (line 242)
    productdir_3399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 43), 'productdir', False)
    # Processing the call keyword arguments (line 242)
    kwargs_3400 = {}
    # Getting the type of 'os' (line 242)
    os_3396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 29), 'os', False)
    # Obtaining the member 'path' of a type (line 242)
    path_3397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 29), os_3396, 'path')
    # Obtaining the member 'isdir' of a type (line 242)
    isdir_3398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 29), path_3397, 'isdir')
    # Calling isdir(args, kwargs) (line 242)
    isdir_call_result_3401 = invoke(stypy.reporting.localization.Localization(__file__, 242, 29), isdir_3398, *[productdir_3399], **kwargs_3400)
    
    # Applying the 'not' unary operator (line 242)
    result_not__3402 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 25), 'not', isdir_call_result_3401)
    
    # Applying the binary operator 'or' (line 242)
    result_or_keyword_3403 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 7), 'or', result_not__3395, result_not__3402)
    
    # Testing the type of an if condition (line 242)
    if_condition_3404 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 242, 4), result_or_keyword_3403)
    # Assigning a type to the variable 'if_condition_3404' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'if_condition_3404', if_condition_3404)
    # SSA begins for if statement (line 242)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 243):
    
    # Assigning a BinOp to a Name (line 243):
    str_3405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 19), 'str', 'VS%0.f0COMNTOOLS')
    # Getting the type of 'version' (line 243)
    version_3406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 40), 'version')
    # Applying the binary operator '%' (line 243)
    result_mod_3407 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 19), '%', str_3405, version_3406)
    
    # Assigning a type to the variable 'toolskey' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'toolskey', result_mod_3407)
    
    # Assigning a Call to a Name (line 244):
    
    # Assigning a Call to a Name (line 244):
    
    # Call to get(...): (line 244)
    # Processing the call arguments (line 244)
    # Getting the type of 'toolskey' (line 244)
    toolskey_3411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 34), 'toolskey', False)
    # Getting the type of 'None' (line 244)
    None_3412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 44), 'None', False)
    # Processing the call keyword arguments (line 244)
    kwargs_3413 = {}
    # Getting the type of 'os' (line 244)
    os_3408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 19), 'os', False)
    # Obtaining the member 'environ' of a type (line 244)
    environ_3409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 19), os_3408, 'environ')
    # Obtaining the member 'get' of a type (line 244)
    get_3410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 19), environ_3409, 'get')
    # Calling get(args, kwargs) (line 244)
    get_call_result_3414 = invoke(stypy.reporting.localization.Localization(__file__, 244, 19), get_3410, *[toolskey_3411, None_3412], **kwargs_3413)
    
    # Assigning a type to the variable 'toolsdir' (line 244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'toolsdir', get_call_result_3414)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'toolsdir' (line 246)
    toolsdir_3415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 11), 'toolsdir')
    
    # Call to isdir(...): (line 246)
    # Processing the call arguments (line 246)
    # Getting the type of 'toolsdir' (line 246)
    toolsdir_3419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 38), 'toolsdir', False)
    # Processing the call keyword arguments (line 246)
    kwargs_3420 = {}
    # Getting the type of 'os' (line 246)
    os_3416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 246)
    path_3417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 24), os_3416, 'path')
    # Obtaining the member 'isdir' of a type (line 246)
    isdir_3418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 24), path_3417, 'isdir')
    # Calling isdir(args, kwargs) (line 246)
    isdir_call_result_3421 = invoke(stypy.reporting.localization.Localization(__file__, 246, 24), isdir_3418, *[toolsdir_3419], **kwargs_3420)
    
    # Applying the binary operator 'and' (line 246)
    result_and_keyword_3422 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 11), 'and', toolsdir_3415, isdir_call_result_3421)
    
    # Testing the type of an if condition (line 246)
    if_condition_3423 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 246, 8), result_and_keyword_3422)
    # Assigning a type to the variable 'if_condition_3423' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'if_condition_3423', if_condition_3423)
    # SSA begins for if statement (line 246)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 247):
    
    # Assigning a Call to a Name (line 247):
    
    # Call to join(...): (line 247)
    # Processing the call arguments (line 247)
    # Getting the type of 'toolsdir' (line 247)
    toolsdir_3427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 38), 'toolsdir', False)
    # Getting the type of 'os' (line 247)
    os_3428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 48), 'os', False)
    # Obtaining the member 'pardir' of a type (line 247)
    pardir_3429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 48), os_3428, 'pardir')
    # Getting the type of 'os' (line 247)
    os_3430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 59), 'os', False)
    # Obtaining the member 'pardir' of a type (line 247)
    pardir_3431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 59), os_3430, 'pardir')
    str_3432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 70), 'str', 'VC')
    # Processing the call keyword arguments (line 247)
    kwargs_3433 = {}
    # Getting the type of 'os' (line 247)
    os_3424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 25), 'os', False)
    # Obtaining the member 'path' of a type (line 247)
    path_3425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 25), os_3424, 'path')
    # Obtaining the member 'join' of a type (line 247)
    join_3426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 25), path_3425, 'join')
    # Calling join(args, kwargs) (line 247)
    join_call_result_3434 = invoke(stypy.reporting.localization.Localization(__file__, 247, 25), join_3426, *[toolsdir_3427, pardir_3429, pardir_3431, str_3432], **kwargs_3433)
    
    # Assigning a type to the variable 'productdir' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'productdir', join_call_result_3434)
    
    # Assigning a Call to a Name (line 248):
    
    # Assigning a Call to a Name (line 248):
    
    # Call to abspath(...): (line 248)
    # Processing the call arguments (line 248)
    # Getting the type of 'productdir' (line 248)
    productdir_3438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 41), 'productdir', False)
    # Processing the call keyword arguments (line 248)
    kwargs_3439 = {}
    # Getting the type of 'os' (line 248)
    os_3435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 25), 'os', False)
    # Obtaining the member 'path' of a type (line 248)
    path_3436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 25), os_3435, 'path')
    # Obtaining the member 'abspath' of a type (line 248)
    abspath_3437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 25), path_3436, 'abspath')
    # Calling abspath(args, kwargs) (line 248)
    abspath_call_result_3440 = invoke(stypy.reporting.localization.Localization(__file__, 248, 25), abspath_3437, *[productdir_3438], **kwargs_3439)
    
    # Assigning a type to the variable 'productdir' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'productdir', abspath_call_result_3440)
    
    
    
    # Call to isdir(...): (line 249)
    # Processing the call arguments (line 249)
    # Getting the type of 'productdir' (line 249)
    productdir_3444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 33), 'productdir', False)
    # Processing the call keyword arguments (line 249)
    kwargs_3445 = {}
    # Getting the type of 'os' (line 249)
    os_3441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 19), 'os', False)
    # Obtaining the member 'path' of a type (line 249)
    path_3442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 19), os_3441, 'path')
    # Obtaining the member 'isdir' of a type (line 249)
    isdir_3443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 19), path_3442, 'isdir')
    # Calling isdir(args, kwargs) (line 249)
    isdir_call_result_3446 = invoke(stypy.reporting.localization.Localization(__file__, 249, 19), isdir_3443, *[productdir_3444], **kwargs_3445)
    
    # Applying the 'not' unary operator (line 249)
    result_not__3447 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 15), 'not', isdir_call_result_3446)
    
    # Testing the type of an if condition (line 249)
    if_condition_3448 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 249, 12), result_not__3447)
    # Assigning a type to the variable 'if_condition_3448' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'if_condition_3448', if_condition_3448)
    # SSA begins for if statement (line 249)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to debug(...): (line 250)
    # Processing the call arguments (line 250)
    str_3451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 26), 'str', '%s is not a valid directory')
    # Getting the type of 'productdir' (line 250)
    productdir_3452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 58), 'productdir', False)
    # Applying the binary operator '%' (line 250)
    result_mod_3453 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 26), '%', str_3451, productdir_3452)
    
    # Processing the call keyword arguments (line 250)
    kwargs_3454 = {}
    # Getting the type of 'log' (line 250)
    log_3449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'log', False)
    # Obtaining the member 'debug' of a type (line 250)
    debug_3450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 16), log_3449, 'debug')
    # Calling debug(args, kwargs) (line 250)
    debug_call_result_3455 = invoke(stypy.reporting.localization.Localization(__file__, 250, 16), debug_3450, *[result_mod_3453], **kwargs_3454)
    
    # Getting the type of 'None' (line 251)
    None_3456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 23), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 16), 'stypy_return_type', None_3456)
    # SSA join for if statement (line 249)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 246)
    module_type_store.open_ssa_branch('else')
    
    # Call to debug(...): (line 253)
    # Processing the call arguments (line 253)
    str_3459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 22), 'str', 'Env var %s is not set or invalid')
    # Getting the type of 'toolskey' (line 253)
    toolskey_3460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 59), 'toolskey', False)
    # Applying the binary operator '%' (line 253)
    result_mod_3461 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 22), '%', str_3459, toolskey_3460)
    
    # Processing the call keyword arguments (line 253)
    kwargs_3462 = {}
    # Getting the type of 'log' (line 253)
    log_3457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'log', False)
    # Obtaining the member 'debug' of a type (line 253)
    debug_3458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 12), log_3457, 'debug')
    # Calling debug(args, kwargs) (line 253)
    debug_call_result_3463 = invoke(stypy.reporting.localization.Localization(__file__, 253, 12), debug_3458, *[result_mod_3461], **kwargs_3462)
    
    # SSA join for if statement (line 246)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 242)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'productdir' (line 254)
    productdir_3464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 11), 'productdir')
    # Applying the 'not' unary operator (line 254)
    result_not__3465 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 7), 'not', productdir_3464)
    
    # Testing the type of an if condition (line 254)
    if_condition_3466 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 254, 4), result_not__3465)
    # Assigning a type to the variable 'if_condition_3466' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'if_condition_3466', if_condition_3466)
    # SSA begins for if statement (line 254)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to debug(...): (line 255)
    # Processing the call arguments (line 255)
    str_3469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 18), 'str', 'No productdir found')
    # Processing the call keyword arguments (line 255)
    kwargs_3470 = {}
    # Getting the type of 'log' (line 255)
    log_3467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'log', False)
    # Obtaining the member 'debug' of a type (line 255)
    debug_3468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 8), log_3467, 'debug')
    # Calling debug(args, kwargs) (line 255)
    debug_call_result_3471 = invoke(stypy.reporting.localization.Localization(__file__, 255, 8), debug_3468, *[str_3469], **kwargs_3470)
    
    # Getting the type of 'None' (line 256)
    None_3472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'stypy_return_type', None_3472)
    # SSA join for if statement (line 254)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 257):
    
    # Assigning a Call to a Name (line 257):
    
    # Call to join(...): (line 257)
    # Processing the call arguments (line 257)
    # Getting the type of 'productdir' (line 257)
    productdir_3476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 29), 'productdir', False)
    str_3477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 41), 'str', 'vcvarsall.bat')
    # Processing the call keyword arguments (line 257)
    kwargs_3478 = {}
    # Getting the type of 'os' (line 257)
    os_3473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), 'os', False)
    # Obtaining the member 'path' of a type (line 257)
    path_3474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 16), os_3473, 'path')
    # Obtaining the member 'join' of a type (line 257)
    join_3475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 16), path_3474, 'join')
    # Calling join(args, kwargs) (line 257)
    join_call_result_3479 = invoke(stypy.reporting.localization.Localization(__file__, 257, 16), join_3475, *[productdir_3476, str_3477], **kwargs_3478)
    
    # Assigning a type to the variable 'vcvarsall' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'vcvarsall', join_call_result_3479)
    
    
    # Call to isfile(...): (line 258)
    # Processing the call arguments (line 258)
    # Getting the type of 'vcvarsall' (line 258)
    vcvarsall_3483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 22), 'vcvarsall', False)
    # Processing the call keyword arguments (line 258)
    kwargs_3484 = {}
    # Getting the type of 'os' (line 258)
    os_3480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 7), 'os', False)
    # Obtaining the member 'path' of a type (line 258)
    path_3481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 7), os_3480, 'path')
    # Obtaining the member 'isfile' of a type (line 258)
    isfile_3482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 7), path_3481, 'isfile')
    # Calling isfile(args, kwargs) (line 258)
    isfile_call_result_3485 = invoke(stypy.reporting.localization.Localization(__file__, 258, 7), isfile_3482, *[vcvarsall_3483], **kwargs_3484)
    
    # Testing the type of an if condition (line 258)
    if_condition_3486 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 258, 4), isfile_call_result_3485)
    # Assigning a type to the variable 'if_condition_3486' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'if_condition_3486', if_condition_3486)
    # SSA begins for if statement (line 258)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'vcvarsall' (line 259)
    vcvarsall_3487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 15), 'vcvarsall')
    # Assigning a type to the variable 'stypy_return_type' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'stypy_return_type', vcvarsall_3487)
    # SSA join for if statement (line 258)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to debug(...): (line 260)
    # Processing the call arguments (line 260)
    str_3490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 14), 'str', 'Unable to find vcvarsall.bat')
    # Processing the call keyword arguments (line 260)
    kwargs_3491 = {}
    # Getting the type of 'log' (line 260)
    log_3488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'log', False)
    # Obtaining the member 'debug' of a type (line 260)
    debug_3489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 4), log_3488, 'debug')
    # Calling debug(args, kwargs) (line 260)
    debug_call_result_3492 = invoke(stypy.reporting.localization.Localization(__file__, 260, 4), debug_3489, *[str_3490], **kwargs_3491)
    
    # Getting the type of 'None' (line 261)
    None_3493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 11), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'stypy_return_type', None_3493)
    
    # ################# End of 'find_vcvarsall(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'find_vcvarsall' in the type store
    # Getting the type of 'stypy_return_type' (line 219)
    stypy_return_type_3494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3494)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'find_vcvarsall'
    return stypy_return_type_3494

# Assigning a type to the variable 'find_vcvarsall' (line 219)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 0), 'find_vcvarsall', find_vcvarsall)

@norecursion
def query_vcvarsall(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_3495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 34), 'str', 'x86')
    defaults = [str_3495]
    # Create a new context for function 'query_vcvarsall'
    module_type_store = module_type_store.open_function_context('query_vcvarsall', 263, 0, False)
    
    # Passed parameters checking function
    query_vcvarsall.stypy_localization = localization
    query_vcvarsall.stypy_type_of_self = None
    query_vcvarsall.stypy_type_store = module_type_store
    query_vcvarsall.stypy_function_name = 'query_vcvarsall'
    query_vcvarsall.stypy_param_names_list = ['version', 'arch']
    query_vcvarsall.stypy_varargs_param_name = None
    query_vcvarsall.stypy_kwargs_param_name = None
    query_vcvarsall.stypy_call_defaults = defaults
    query_vcvarsall.stypy_call_varargs = varargs
    query_vcvarsall.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'query_vcvarsall', ['version', 'arch'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'query_vcvarsall', localization, ['version', 'arch'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'query_vcvarsall(...)' code ##################

    str_3496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, (-1)), 'str', 'Launch vcvarsall.bat and read the settings from its environment\n    ')
    
    # Assigning a Call to a Name (line 266):
    
    # Assigning a Call to a Name (line 266):
    
    # Call to find_vcvarsall(...): (line 266)
    # Processing the call arguments (line 266)
    # Getting the type of 'version' (line 266)
    version_3498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 31), 'version', False)
    # Processing the call keyword arguments (line 266)
    kwargs_3499 = {}
    # Getting the type of 'find_vcvarsall' (line 266)
    find_vcvarsall_3497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'find_vcvarsall', False)
    # Calling find_vcvarsall(args, kwargs) (line 266)
    find_vcvarsall_call_result_3500 = invoke(stypy.reporting.localization.Localization(__file__, 266, 16), find_vcvarsall_3497, *[version_3498], **kwargs_3499)
    
    # Assigning a type to the variable 'vcvarsall' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'vcvarsall', find_vcvarsall_call_result_3500)
    
    # Assigning a Call to a Name (line 267):
    
    # Assigning a Call to a Name (line 267):
    
    # Call to set(...): (line 267)
    # Processing the call arguments (line 267)
    
    # Obtaining an instance of the builtin type 'tuple' (line 267)
    tuple_3502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 267)
    # Adding element type (line 267)
    str_3503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 23), 'str', 'include')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 23), tuple_3502, str_3503)
    # Adding element type (line 267)
    str_3504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 34), 'str', 'lib')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 23), tuple_3502, str_3504)
    # Adding element type (line 267)
    str_3505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 41), 'str', 'libpath')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 23), tuple_3502, str_3505)
    # Adding element type (line 267)
    str_3506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 52), 'str', 'path')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 23), tuple_3502, str_3506)
    
    # Processing the call keyword arguments (line 267)
    kwargs_3507 = {}
    # Getting the type of 'set' (line 267)
    set_3501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 18), 'set', False)
    # Calling set(args, kwargs) (line 267)
    set_call_result_3508 = invoke(stypy.reporting.localization.Localization(__file__, 267, 18), set_3501, *[tuple_3502], **kwargs_3507)
    
    # Assigning a type to the variable 'interesting' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'interesting', set_call_result_3508)
    
    # Assigning a Dict to a Name (line 268):
    
    # Assigning a Dict to a Name (line 268):
    
    # Obtaining an instance of the builtin type 'dict' (line 268)
    dict_3509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 13), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 268)
    
    # Assigning a type to the variable 'result' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'result', dict_3509)
    
    # Type idiom detected: calculating its left and rigth part (line 270)
    # Getting the type of 'vcvarsall' (line 270)
    vcvarsall_3510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 7), 'vcvarsall')
    # Getting the type of 'None' (line 270)
    None_3511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 20), 'None')
    
    (may_be_3512, more_types_in_union_3513) = may_be_none(vcvarsall_3510, None_3511)

    if may_be_3512:

        if more_types_in_union_3513:
            # Runtime conditional SSA (line 270)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to DistutilsPlatformError(...): (line 271)
        # Processing the call arguments (line 271)
        str_3515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 37), 'str', 'Unable to find vcvarsall.bat')
        # Processing the call keyword arguments (line 271)
        kwargs_3516 = {}
        # Getting the type of 'DistutilsPlatformError' (line 271)
        DistutilsPlatformError_3514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 14), 'DistutilsPlatformError', False)
        # Calling DistutilsPlatformError(args, kwargs) (line 271)
        DistutilsPlatformError_call_result_3517 = invoke(stypy.reporting.localization.Localization(__file__, 271, 14), DistutilsPlatformError_3514, *[str_3515], **kwargs_3516)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 271, 8), DistutilsPlatformError_call_result_3517, 'raise parameter', BaseException)

        if more_types_in_union_3513:
            # SSA join for if statement (line 270)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to debug(...): (line 272)
    # Processing the call arguments (line 272)
    str_3520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 14), 'str', "Calling 'vcvarsall.bat %s' (version=%s)")
    # Getting the type of 'arch' (line 272)
    arch_3521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 57), 'arch', False)
    # Getting the type of 'version' (line 272)
    version_3522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 63), 'version', False)
    # Processing the call keyword arguments (line 272)
    kwargs_3523 = {}
    # Getting the type of 'log' (line 272)
    log_3518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'log', False)
    # Obtaining the member 'debug' of a type (line 272)
    debug_3519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 4), log_3518, 'debug')
    # Calling debug(args, kwargs) (line 272)
    debug_call_result_3524 = invoke(stypy.reporting.localization.Localization(__file__, 272, 4), debug_3519, *[str_3520, arch_3521, version_3522], **kwargs_3523)
    
    
    # Assigning a Call to a Name (line 273):
    
    # Assigning a Call to a Name (line 273):
    
    # Call to Popen(...): (line 273)
    # Processing the call arguments (line 273)
    str_3527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 29), 'str', '"%s" %s & set')
    
    # Obtaining an instance of the builtin type 'tuple' (line 273)
    tuple_3528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 273)
    # Adding element type (line 273)
    # Getting the type of 'vcvarsall' (line 273)
    vcvarsall_3529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 48), 'vcvarsall', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 48), tuple_3528, vcvarsall_3529)
    # Adding element type (line 273)
    # Getting the type of 'arch' (line 273)
    arch_3530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 59), 'arch', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 48), tuple_3528, arch_3530)
    
    # Applying the binary operator '%' (line 273)
    result_mod_3531 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 29), '%', str_3527, tuple_3528)
    
    # Processing the call keyword arguments (line 273)
    # Getting the type of 'subprocess' (line 274)
    subprocess_3532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 36), 'subprocess', False)
    # Obtaining the member 'PIPE' of a type (line 274)
    PIPE_3533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 36), subprocess_3532, 'PIPE')
    keyword_3534 = PIPE_3533
    # Getting the type of 'subprocess' (line 275)
    subprocess_3535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 36), 'subprocess', False)
    # Obtaining the member 'PIPE' of a type (line 275)
    PIPE_3536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 36), subprocess_3535, 'PIPE')
    keyword_3537 = PIPE_3536
    kwargs_3538 = {'stderr': keyword_3537, 'stdout': keyword_3534}
    # Getting the type of 'subprocess' (line 273)
    subprocess_3525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'subprocess', False)
    # Obtaining the member 'Popen' of a type (line 273)
    Popen_3526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 12), subprocess_3525, 'Popen')
    # Calling Popen(args, kwargs) (line 273)
    Popen_call_result_3539 = invoke(stypy.reporting.localization.Localization(__file__, 273, 12), Popen_3526, *[result_mod_3531], **kwargs_3538)
    
    # Assigning a type to the variable 'popen' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'popen', Popen_call_result_3539)
    
    # Try-finally block (line 276)
    
    # Assigning a Call to a Tuple (line 277):
    
    # Assigning a Subscript to a Name (line 277):
    
    # Obtaining the type of the subscript
    int_3540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 8), 'int')
    
    # Call to communicate(...): (line 277)
    # Processing the call keyword arguments (line 277)
    kwargs_3543 = {}
    # Getting the type of 'popen' (line 277)
    popen_3541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 25), 'popen', False)
    # Obtaining the member 'communicate' of a type (line 277)
    communicate_3542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 25), popen_3541, 'communicate')
    # Calling communicate(args, kwargs) (line 277)
    communicate_call_result_3544 = invoke(stypy.reporting.localization.Localization(__file__, 277, 25), communicate_3542, *[], **kwargs_3543)
    
    # Obtaining the member '__getitem__' of a type (line 277)
    getitem___3545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 8), communicate_call_result_3544, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 277)
    subscript_call_result_3546 = invoke(stypy.reporting.localization.Localization(__file__, 277, 8), getitem___3545, int_3540)
    
    # Assigning a type to the variable 'tuple_var_assignment_2870' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'tuple_var_assignment_2870', subscript_call_result_3546)
    
    # Assigning a Subscript to a Name (line 277):
    
    # Obtaining the type of the subscript
    int_3547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 8), 'int')
    
    # Call to communicate(...): (line 277)
    # Processing the call keyword arguments (line 277)
    kwargs_3550 = {}
    # Getting the type of 'popen' (line 277)
    popen_3548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 25), 'popen', False)
    # Obtaining the member 'communicate' of a type (line 277)
    communicate_3549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 25), popen_3548, 'communicate')
    # Calling communicate(args, kwargs) (line 277)
    communicate_call_result_3551 = invoke(stypy.reporting.localization.Localization(__file__, 277, 25), communicate_3549, *[], **kwargs_3550)
    
    # Obtaining the member '__getitem__' of a type (line 277)
    getitem___3552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 8), communicate_call_result_3551, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 277)
    subscript_call_result_3553 = invoke(stypy.reporting.localization.Localization(__file__, 277, 8), getitem___3552, int_3547)
    
    # Assigning a type to the variable 'tuple_var_assignment_2871' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'tuple_var_assignment_2871', subscript_call_result_3553)
    
    # Assigning a Name to a Name (line 277):
    # Getting the type of 'tuple_var_assignment_2870' (line 277)
    tuple_var_assignment_2870_3554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'tuple_var_assignment_2870')
    # Assigning a type to the variable 'stdout' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'stdout', tuple_var_assignment_2870_3554)
    
    # Assigning a Name to a Name (line 277):
    # Getting the type of 'tuple_var_assignment_2871' (line 277)
    tuple_var_assignment_2871_3555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'tuple_var_assignment_2871')
    # Assigning a type to the variable 'stderr' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 16), 'stderr', tuple_var_assignment_2871_3555)
    
    
    
    # Call to wait(...): (line 278)
    # Processing the call keyword arguments (line 278)
    kwargs_3558 = {}
    # Getting the type of 'popen' (line 278)
    popen_3556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 11), 'popen', False)
    # Obtaining the member 'wait' of a type (line 278)
    wait_3557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 11), popen_3556, 'wait')
    # Calling wait(args, kwargs) (line 278)
    wait_call_result_3559 = invoke(stypy.reporting.localization.Localization(__file__, 278, 11), wait_3557, *[], **kwargs_3558)
    
    int_3560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 27), 'int')
    # Applying the binary operator '!=' (line 278)
    result_ne_3561 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 11), '!=', wait_call_result_3559, int_3560)
    
    # Testing the type of an if condition (line 278)
    if_condition_3562 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 278, 8), result_ne_3561)
    # Assigning a type to the variable 'if_condition_3562' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'if_condition_3562', if_condition_3562)
    # SSA begins for if statement (line 278)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to DistutilsPlatformError(...): (line 279)
    # Processing the call arguments (line 279)
    
    # Call to decode(...): (line 279)
    # Processing the call arguments (line 279)
    str_3566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 55), 'str', 'mbcs')
    # Processing the call keyword arguments (line 279)
    kwargs_3567 = {}
    # Getting the type of 'stderr' (line 279)
    stderr_3564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 41), 'stderr', False)
    # Obtaining the member 'decode' of a type (line 279)
    decode_3565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 41), stderr_3564, 'decode')
    # Calling decode(args, kwargs) (line 279)
    decode_call_result_3568 = invoke(stypy.reporting.localization.Localization(__file__, 279, 41), decode_3565, *[str_3566], **kwargs_3567)
    
    # Processing the call keyword arguments (line 279)
    kwargs_3569 = {}
    # Getting the type of 'DistutilsPlatformError' (line 279)
    DistutilsPlatformError_3563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 18), 'DistutilsPlatformError', False)
    # Calling DistutilsPlatformError(args, kwargs) (line 279)
    DistutilsPlatformError_call_result_3570 = invoke(stypy.reporting.localization.Localization(__file__, 279, 18), DistutilsPlatformError_3563, *[decode_call_result_3568], **kwargs_3569)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 279, 12), DistutilsPlatformError_call_result_3570, 'raise parameter', BaseException)
    # SSA join for if statement (line 278)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 281):
    
    # Assigning a Call to a Name (line 281):
    
    # Call to decode(...): (line 281)
    # Processing the call arguments (line 281)
    str_3573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 31), 'str', 'mbcs')
    # Processing the call keyword arguments (line 281)
    kwargs_3574 = {}
    # Getting the type of 'stdout' (line 281)
    stdout_3571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 17), 'stdout', False)
    # Obtaining the member 'decode' of a type (line 281)
    decode_3572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 17), stdout_3571, 'decode')
    # Calling decode(args, kwargs) (line 281)
    decode_call_result_3575 = invoke(stypy.reporting.localization.Localization(__file__, 281, 17), decode_3572, *[str_3573], **kwargs_3574)
    
    # Assigning a type to the variable 'stdout' (line 281)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'stdout', decode_call_result_3575)
    
    
    # Call to split(...): (line 282)
    # Processing the call arguments (line 282)
    str_3578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 33), 'str', '\n')
    # Processing the call keyword arguments (line 282)
    kwargs_3579 = {}
    # Getting the type of 'stdout' (line 282)
    stdout_3576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 20), 'stdout', False)
    # Obtaining the member 'split' of a type (line 282)
    split_3577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 20), stdout_3576, 'split')
    # Calling split(args, kwargs) (line 282)
    split_call_result_3580 = invoke(stypy.reporting.localization.Localization(__file__, 282, 20), split_3577, *[str_3578], **kwargs_3579)
    
    # Testing the type of a for loop iterable (line 282)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 282, 8), split_call_result_3580)
    # Getting the type of the for loop variable (line 282)
    for_loop_var_3581 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 282, 8), split_call_result_3580)
    # Assigning a type to the variable 'line' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'line', for_loop_var_3581)
    # SSA begins for a for statement (line 282)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 283):
    
    # Assigning a Call to a Name (line 283):
    
    # Call to convert_mbcs(...): (line 283)
    # Processing the call arguments (line 283)
    # Getting the type of 'line' (line 283)
    line_3584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 36), 'line', False)
    # Processing the call keyword arguments (line 283)
    kwargs_3585 = {}
    # Getting the type of 'Reg' (line 283)
    Reg_3582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 19), 'Reg', False)
    # Obtaining the member 'convert_mbcs' of a type (line 283)
    convert_mbcs_3583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 19), Reg_3582, 'convert_mbcs')
    # Calling convert_mbcs(args, kwargs) (line 283)
    convert_mbcs_call_result_3586 = invoke(stypy.reporting.localization.Localization(__file__, 283, 19), convert_mbcs_3583, *[line_3584], **kwargs_3585)
    
    # Assigning a type to the variable 'line' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'line', convert_mbcs_call_result_3586)
    
    
    str_3587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 15), 'str', '=')
    # Getting the type of 'line' (line 284)
    line_3588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 26), 'line')
    # Applying the binary operator 'notin' (line 284)
    result_contains_3589 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 15), 'notin', str_3587, line_3588)
    
    # Testing the type of an if condition (line 284)
    if_condition_3590 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 284, 12), result_contains_3589)
    # Assigning a type to the variable 'if_condition_3590' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'if_condition_3590', if_condition_3590)
    # SSA begins for if statement (line 284)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 284)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 286):
    
    # Assigning a Call to a Name (line 286):
    
    # Call to strip(...): (line 286)
    # Processing the call keyword arguments (line 286)
    kwargs_3593 = {}
    # Getting the type of 'line' (line 286)
    line_3591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 19), 'line', False)
    # Obtaining the member 'strip' of a type (line 286)
    strip_3592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 19), line_3591, 'strip')
    # Calling strip(args, kwargs) (line 286)
    strip_call_result_3594 = invoke(stypy.reporting.localization.Localization(__file__, 286, 19), strip_3592, *[], **kwargs_3593)
    
    # Assigning a type to the variable 'line' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'line', strip_call_result_3594)
    
    # Assigning a Call to a Tuple (line 287):
    
    # Assigning a Subscript to a Name (line 287):
    
    # Obtaining the type of the subscript
    int_3595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 12), 'int')
    
    # Call to split(...): (line 287)
    # Processing the call arguments (line 287)
    str_3598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 36), 'str', '=')
    int_3599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 41), 'int')
    # Processing the call keyword arguments (line 287)
    kwargs_3600 = {}
    # Getting the type of 'line' (line 287)
    line_3596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 25), 'line', False)
    # Obtaining the member 'split' of a type (line 287)
    split_3597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 25), line_3596, 'split')
    # Calling split(args, kwargs) (line 287)
    split_call_result_3601 = invoke(stypy.reporting.localization.Localization(__file__, 287, 25), split_3597, *[str_3598, int_3599], **kwargs_3600)
    
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___3602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 12), split_call_result_3601, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_3603 = invoke(stypy.reporting.localization.Localization(__file__, 287, 12), getitem___3602, int_3595)
    
    # Assigning a type to the variable 'tuple_var_assignment_2872' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'tuple_var_assignment_2872', subscript_call_result_3603)
    
    # Assigning a Subscript to a Name (line 287):
    
    # Obtaining the type of the subscript
    int_3604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 12), 'int')
    
    # Call to split(...): (line 287)
    # Processing the call arguments (line 287)
    str_3607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 36), 'str', '=')
    int_3608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 41), 'int')
    # Processing the call keyword arguments (line 287)
    kwargs_3609 = {}
    # Getting the type of 'line' (line 287)
    line_3605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 25), 'line', False)
    # Obtaining the member 'split' of a type (line 287)
    split_3606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 25), line_3605, 'split')
    # Calling split(args, kwargs) (line 287)
    split_call_result_3610 = invoke(stypy.reporting.localization.Localization(__file__, 287, 25), split_3606, *[str_3607, int_3608], **kwargs_3609)
    
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___3611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 12), split_call_result_3610, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_3612 = invoke(stypy.reporting.localization.Localization(__file__, 287, 12), getitem___3611, int_3604)
    
    # Assigning a type to the variable 'tuple_var_assignment_2873' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'tuple_var_assignment_2873', subscript_call_result_3612)
    
    # Assigning a Name to a Name (line 287):
    # Getting the type of 'tuple_var_assignment_2872' (line 287)
    tuple_var_assignment_2872_3613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'tuple_var_assignment_2872')
    # Assigning a type to the variable 'key' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'key', tuple_var_assignment_2872_3613)
    
    # Assigning a Name to a Name (line 287):
    # Getting the type of 'tuple_var_assignment_2873' (line 287)
    tuple_var_assignment_2873_3614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'tuple_var_assignment_2873')
    # Assigning a type to the variable 'value' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 17), 'value', tuple_var_assignment_2873_3614)
    
    # Assigning a Call to a Name (line 288):
    
    # Assigning a Call to a Name (line 288):
    
    # Call to lower(...): (line 288)
    # Processing the call keyword arguments (line 288)
    kwargs_3617 = {}
    # Getting the type of 'key' (line 288)
    key_3615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 18), 'key', False)
    # Obtaining the member 'lower' of a type (line 288)
    lower_3616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 18), key_3615, 'lower')
    # Calling lower(args, kwargs) (line 288)
    lower_call_result_3618 = invoke(stypy.reporting.localization.Localization(__file__, 288, 18), lower_3616, *[], **kwargs_3617)
    
    # Assigning a type to the variable 'key' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'key', lower_call_result_3618)
    
    
    # Getting the type of 'key' (line 289)
    key_3619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 15), 'key')
    # Getting the type of 'interesting' (line 289)
    interesting_3620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 22), 'interesting')
    # Applying the binary operator 'in' (line 289)
    result_contains_3621 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 15), 'in', key_3619, interesting_3620)
    
    # Testing the type of an if condition (line 289)
    if_condition_3622 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 289, 12), result_contains_3621)
    # Assigning a type to the variable 'if_condition_3622' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'if_condition_3622', if_condition_3622)
    # SSA begins for if statement (line 289)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to endswith(...): (line 290)
    # Processing the call arguments (line 290)
    # Getting the type of 'os' (line 290)
    os_3625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 34), 'os', False)
    # Obtaining the member 'pathsep' of a type (line 290)
    pathsep_3626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 34), os_3625, 'pathsep')
    # Processing the call keyword arguments (line 290)
    kwargs_3627 = {}
    # Getting the type of 'value' (line 290)
    value_3623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 19), 'value', False)
    # Obtaining the member 'endswith' of a type (line 290)
    endswith_3624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 19), value_3623, 'endswith')
    # Calling endswith(args, kwargs) (line 290)
    endswith_call_result_3628 = invoke(stypy.reporting.localization.Localization(__file__, 290, 19), endswith_3624, *[pathsep_3626], **kwargs_3627)
    
    # Testing the type of an if condition (line 290)
    if_condition_3629 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 290, 16), endswith_call_result_3628)
    # Assigning a type to the variable 'if_condition_3629' (line 290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 16), 'if_condition_3629', if_condition_3629)
    # SSA begins for if statement (line 290)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 291):
    
    # Assigning a Subscript to a Name (line 291):
    
    # Obtaining the type of the subscript
    int_3630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 35), 'int')
    slice_3631 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 291, 28), None, int_3630, None)
    # Getting the type of 'value' (line 291)
    value_3632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 28), 'value')
    # Obtaining the member '__getitem__' of a type (line 291)
    getitem___3633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 28), value_3632, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 291)
    subscript_call_result_3634 = invoke(stypy.reporting.localization.Localization(__file__, 291, 28), getitem___3633, slice_3631)
    
    # Assigning a type to the variable 'value' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 20), 'value', subscript_call_result_3634)
    # SSA join for if statement (line 290)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Subscript (line 292):
    
    # Assigning a Call to a Subscript (line 292):
    
    # Call to removeDuplicates(...): (line 292)
    # Processing the call arguments (line 292)
    # Getting the type of 'value' (line 292)
    value_3636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 47), 'value', False)
    # Processing the call keyword arguments (line 292)
    kwargs_3637 = {}
    # Getting the type of 'removeDuplicates' (line 292)
    removeDuplicates_3635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 30), 'removeDuplicates', False)
    # Calling removeDuplicates(args, kwargs) (line 292)
    removeDuplicates_call_result_3638 = invoke(stypy.reporting.localization.Localization(__file__, 292, 30), removeDuplicates_3635, *[value_3636], **kwargs_3637)
    
    # Getting the type of 'result' (line 292)
    result_3639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 16), 'result')
    # Getting the type of 'key' (line 292)
    key_3640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 23), 'key')
    # Storing an element on a container (line 292)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 16), result_3639, (key_3640, removeDuplicates_call_result_3638))
    # SSA join for if statement (line 289)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # finally branch of the try-finally block (line 276)
    
    # Call to close(...): (line 295)
    # Processing the call keyword arguments (line 295)
    kwargs_3644 = {}
    # Getting the type of 'popen' (line 295)
    popen_3641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'popen', False)
    # Obtaining the member 'stdout' of a type (line 295)
    stdout_3642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 8), popen_3641, 'stdout')
    # Obtaining the member 'close' of a type (line 295)
    close_3643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 8), stdout_3642, 'close')
    # Calling close(args, kwargs) (line 295)
    close_call_result_3645 = invoke(stypy.reporting.localization.Localization(__file__, 295, 8), close_3643, *[], **kwargs_3644)
    
    
    # Call to close(...): (line 296)
    # Processing the call keyword arguments (line 296)
    kwargs_3649 = {}
    # Getting the type of 'popen' (line 296)
    popen_3646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'popen', False)
    # Obtaining the member 'stderr' of a type (line 296)
    stderr_3647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 8), popen_3646, 'stderr')
    # Obtaining the member 'close' of a type (line 296)
    close_3648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 8), stderr_3647, 'close')
    # Calling close(args, kwargs) (line 296)
    close_call_result_3650 = invoke(stypy.reporting.localization.Localization(__file__, 296, 8), close_3648, *[], **kwargs_3649)
    
    
    
    
    
    # Call to len(...): (line 298)
    # Processing the call arguments (line 298)
    # Getting the type of 'result' (line 298)
    result_3652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 11), 'result', False)
    # Processing the call keyword arguments (line 298)
    kwargs_3653 = {}
    # Getting the type of 'len' (line 298)
    len_3651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 7), 'len', False)
    # Calling len(args, kwargs) (line 298)
    len_call_result_3654 = invoke(stypy.reporting.localization.Localization(__file__, 298, 7), len_3651, *[result_3652], **kwargs_3653)
    
    
    # Call to len(...): (line 298)
    # Processing the call arguments (line 298)
    # Getting the type of 'interesting' (line 298)
    interesting_3656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 26), 'interesting', False)
    # Processing the call keyword arguments (line 298)
    kwargs_3657 = {}
    # Getting the type of 'len' (line 298)
    len_3655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 22), 'len', False)
    # Calling len(args, kwargs) (line 298)
    len_call_result_3658 = invoke(stypy.reporting.localization.Localization(__file__, 298, 22), len_3655, *[interesting_3656], **kwargs_3657)
    
    # Applying the binary operator '!=' (line 298)
    result_ne_3659 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 7), '!=', len_call_result_3654, len_call_result_3658)
    
    # Testing the type of an if condition (line 298)
    if_condition_3660 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 298, 4), result_ne_3659)
    # Assigning a type to the variable 'if_condition_3660' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'if_condition_3660', if_condition_3660)
    # SSA begins for if statement (line 298)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 299)
    # Processing the call arguments (line 299)
    
    # Call to str(...): (line 299)
    # Processing the call arguments (line 299)
    
    # Call to list(...): (line 299)
    # Processing the call arguments (line 299)
    
    # Call to keys(...): (line 299)
    # Processing the call keyword arguments (line 299)
    kwargs_3666 = {}
    # Getting the type of 'result' (line 299)
    result_3664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 34), 'result', False)
    # Obtaining the member 'keys' of a type (line 299)
    keys_3665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 34), result_3664, 'keys')
    # Calling keys(args, kwargs) (line 299)
    keys_call_result_3667 = invoke(stypy.reporting.localization.Localization(__file__, 299, 34), keys_3665, *[], **kwargs_3666)
    
    # Processing the call keyword arguments (line 299)
    kwargs_3668 = {}
    # Getting the type of 'list' (line 299)
    list_3663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 29), 'list', False)
    # Calling list(args, kwargs) (line 299)
    list_call_result_3669 = invoke(stypy.reporting.localization.Localization(__file__, 299, 29), list_3663, *[keys_call_result_3667], **kwargs_3668)
    
    # Processing the call keyword arguments (line 299)
    kwargs_3670 = {}
    # Getting the type of 'str' (line 299)
    str_3662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 25), 'str', False)
    # Calling str(args, kwargs) (line 299)
    str_call_result_3671 = invoke(stypy.reporting.localization.Localization(__file__, 299, 25), str_3662, *[list_call_result_3669], **kwargs_3670)
    
    # Processing the call keyword arguments (line 299)
    kwargs_3672 = {}
    # Getting the type of 'ValueError' (line 299)
    ValueError_3661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 299)
    ValueError_call_result_3673 = invoke(stypy.reporting.localization.Localization(__file__, 299, 14), ValueError_3661, *[str_call_result_3671], **kwargs_3672)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 299, 8), ValueError_call_result_3673, 'raise parameter', BaseException)
    # SSA join for if statement (line 298)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'result' (line 301)
    result_3674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'stypy_return_type', result_3674)
    
    # ################# End of 'query_vcvarsall(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'query_vcvarsall' in the type store
    # Getting the type of 'stypy_return_type' (line 263)
    stypy_return_type_3675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3675)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'query_vcvarsall'
    return stypy_return_type_3675

# Assigning a type to the variable 'query_vcvarsall' (line 263)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'query_vcvarsall', query_vcvarsall)

# Assigning a Call to a Name (line 304):

# Assigning a Call to a Name (line 304):

# Call to get_build_version(...): (line 304)
# Processing the call keyword arguments (line 304)
kwargs_3677 = {}
# Getting the type of 'get_build_version' (line 304)
get_build_version_3676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 10), 'get_build_version', False)
# Calling get_build_version(args, kwargs) (line 304)
get_build_version_call_result_3678 = invoke(stypy.reporting.localization.Localization(__file__, 304, 10), get_build_version_3676, *[], **kwargs_3677)

# Assigning a type to the variable 'VERSION' (line 304)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 0), 'VERSION', get_build_version_call_result_3678)


# Getting the type of 'VERSION' (line 305)
VERSION_3679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 3), 'VERSION')
float_3680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 13), 'float')
# Applying the binary operator '<' (line 305)
result_lt_3681 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 3), '<', VERSION_3679, float_3680)

# Testing the type of an if condition (line 305)
if_condition_3682 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 305, 0), result_lt_3681)
# Assigning a type to the variable 'if_condition_3682' (line 305)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 0), 'if_condition_3682', if_condition_3682)
# SSA begins for if statement (line 305)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to DistutilsPlatformError(...): (line 306)
# Processing the call arguments (line 306)
str_3684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 33), 'str', 'VC %0.1f is not supported by this module')
# Getting the type of 'VERSION' (line 306)
VERSION_3685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 78), 'VERSION', False)
# Applying the binary operator '%' (line 306)
result_mod_3686 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 33), '%', str_3684, VERSION_3685)

# Processing the call keyword arguments (line 306)
kwargs_3687 = {}
# Getting the type of 'DistutilsPlatformError' (line 306)
DistutilsPlatformError_3683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 10), 'DistutilsPlatformError', False)
# Calling DistutilsPlatformError(args, kwargs) (line 306)
DistutilsPlatformError_call_result_3688 = invoke(stypy.reporting.localization.Localization(__file__, 306, 10), DistutilsPlatformError_3683, *[result_mod_3686], **kwargs_3687)

ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 306, 4), DistutilsPlatformError_call_result_3688, 'raise parameter', BaseException)
# SSA join for if statement (line 305)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'MSVCCompiler' class
# Getting the type of 'CCompiler' (line 309)
CCompiler_3689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 19), 'CCompiler')

class MSVCCompiler(CCompiler_3689, ):
    str_3690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, (-1)), 'str', 'Concrete class that implements an interface to Microsoft Visual C++,\n       as defined by the CCompiler abstract class.')
    
    # Assigning a Str to a Name (line 313):
    
    # Assigning a Dict to a Name (line 320):
    
    # Assigning a List to a Name (line 323):
    
    # Assigning a List to a Name (line 324):
    
    # Assigning a List to a Name (line 325):
    
    # Assigning a List to a Name (line 326):
    
    # Assigning a BinOp to a Name (line 330):
    
    # Assigning a Str to a Name (line 332):
    
    # Assigning a Str to a Name (line 333):
    
    # Assigning a Str to a Name (line 334):
    
    # Assigning a Str to a Name (line 335):
    
    # Multiple assignment of 2 elements.
    
    # Assigning a Str to a Name (line 337):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_3691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 31), 'int')
        int_3692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 42), 'int')
        int_3693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 51), 'int')
        defaults = [int_3691, int_3692, int_3693]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 339, 4, False)
        # Assigning a type to the variable 'self' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'self', type_of_self)
        
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

        
        # Call to __init__(...): (line 340)
        # Processing the call arguments (line 340)
        # Getting the type of 'self' (line 340)
        self_3696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 28), 'self', False)
        # Getting the type of 'verbose' (line 340)
        verbose_3697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 34), 'verbose', False)
        # Getting the type of 'dry_run' (line 340)
        dry_run_3698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 43), 'dry_run', False)
        # Getting the type of 'force' (line 340)
        force_3699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 52), 'force', False)
        # Processing the call keyword arguments (line 340)
        kwargs_3700 = {}
        # Getting the type of 'CCompiler' (line 340)
        CCompiler_3694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'CCompiler', False)
        # Obtaining the member '__init__' of a type (line 340)
        init___3695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 8), CCompiler_3694, '__init__')
        # Calling __init__(args, kwargs) (line 340)
        init___call_result_3701 = invoke(stypy.reporting.localization.Localization(__file__, 340, 8), init___3695, *[self_3696, verbose_3697, dry_run_3698, force_3699], **kwargs_3700)
        
        
        # Assigning a Name to a Attribute (line 341):
        
        # Assigning a Name to a Attribute (line 341):
        # Getting the type of 'VERSION' (line 341)
        VERSION_3702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 25), 'VERSION')
        # Getting the type of 'self' (line 341)
        self_3703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'self')
        # Setting the type of the member '__version' of a type (line 341)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 8), self_3703, '__version', VERSION_3702)
        
        # Assigning a Str to a Attribute (line 342):
        
        # Assigning a Str to a Attribute (line 342):
        str_3704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 22), 'str', 'Software\\Microsoft\\VisualStudio')
        # Getting the type of 'self' (line 342)
        self_3705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'self')
        # Setting the type of the member '__root' of a type (line 342)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 8), self_3705, '__root', str_3704)
        
        # Assigning a List to a Attribute (line 344):
        
        # Assigning a List to a Attribute (line 344):
        
        # Obtaining an instance of the builtin type 'list' (line 344)
        list_3706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 344)
        
        # Getting the type of 'self' (line 344)
        self_3707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'self')
        # Setting the type of the member '__paths' of a type (line 344)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 8), self_3707, '__paths', list_3706)
        
        # Assigning a Name to a Attribute (line 346):
        
        # Assigning a Name to a Attribute (line 346):
        # Getting the type of 'None' (line 346)
        None_3708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 25), 'None')
        # Getting the type of 'self' (line 346)
        self_3709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'self')
        # Setting the type of the member 'plat_name' of a type (line 346)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 8), self_3709, 'plat_name', None_3708)
        
        # Assigning a Name to a Attribute (line 347):
        
        # Assigning a Name to a Attribute (line 347):
        # Getting the type of 'None' (line 347)
        None_3710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 22), 'None')
        # Getting the type of 'self' (line 347)
        self_3711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'self')
        # Setting the type of the member '__arch' of a type (line 347)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 8), self_3711, '__arch', None_3710)
        
        # Assigning a Name to a Attribute (line 348):
        
        # Assigning a Name to a Attribute (line 348):
        # Getting the type of 'False' (line 348)
        False_3712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 27), 'False')
        # Getting the type of 'self' (line 348)
        self_3713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'self')
        # Setting the type of the member 'initialized' of a type (line 348)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 8), self_3713, 'initialized', False_3712)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def initialize(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 350)
        None_3714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 35), 'None')
        defaults = [None_3714]
        # Create a new context for function 'initialize'
        module_type_store = module_type_store.open_function_context('initialize', 350, 4, False)
        # Assigning a type to the variable 'self' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_localization', localization)
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_type_store', module_type_store)
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_function_name', 'MSVCCompiler.initialize')
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_param_names_list', ['plat_name'])
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_varargs_param_name', None)
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_call_defaults', defaults)
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_call_varargs', varargs)
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MSVCCompiler.initialize', ['plat_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'initialize', localization, ['plat_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'initialize(...)' code ##################

        # Evaluating assert statement condition
        
        # Getting the type of 'self' (line 352)
        self_3715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 19), 'self')
        # Obtaining the member 'initialized' of a type (line 352)
        initialized_3716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 19), self_3715, 'initialized')
        # Applying the 'not' unary operator (line 352)
        result_not__3717 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 15), 'not', initialized_3716)
        
        
        # Type idiom detected: calculating its left and rigth part (line 353)
        # Getting the type of 'plat_name' (line 353)
        plat_name_3718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 11), 'plat_name')
        # Getting the type of 'None' (line 353)
        None_3719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 24), 'None')
        
        (may_be_3720, more_types_in_union_3721) = may_be_none(plat_name_3718, None_3719)

        if may_be_3720:

            if more_types_in_union_3721:
                # Runtime conditional SSA (line 353)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 354):
            
            # Assigning a Call to a Name (line 354):
            
            # Call to get_platform(...): (line 354)
            # Processing the call keyword arguments (line 354)
            kwargs_3723 = {}
            # Getting the type of 'get_platform' (line 354)
            get_platform_3722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 24), 'get_platform', False)
            # Calling get_platform(args, kwargs) (line 354)
            get_platform_call_result_3724 = invoke(stypy.reporting.localization.Localization(__file__, 354, 24), get_platform_3722, *[], **kwargs_3723)
            
            # Assigning a type to the variable 'plat_name' (line 354)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 12), 'plat_name', get_platform_call_result_3724)

            if more_types_in_union_3721:
                # SSA join for if statement (line 353)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Tuple to a Name (line 356):
        
        # Assigning a Tuple to a Name (line 356):
        
        # Obtaining an instance of the builtin type 'tuple' (line 356)
        tuple_3725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 356)
        # Adding element type (line 356)
        str_3726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 19), 'str', 'win32')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 19), tuple_3725, str_3726)
        # Adding element type (line 356)
        str_3727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 28), 'str', 'win-amd64')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 19), tuple_3725, str_3727)
        # Adding element type (line 356)
        str_3728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 41), 'str', 'win-ia64')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 19), tuple_3725, str_3728)
        
        # Assigning a type to the variable 'ok_plats' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'ok_plats', tuple_3725)
        
        
        # Getting the type of 'plat_name' (line 357)
        plat_name_3729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 11), 'plat_name')
        # Getting the type of 'ok_plats' (line 357)
        ok_plats_3730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 28), 'ok_plats')
        # Applying the binary operator 'notin' (line 357)
        result_contains_3731 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 11), 'notin', plat_name_3729, ok_plats_3730)
        
        # Testing the type of an if condition (line 357)
        if_condition_3732 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 357, 8), result_contains_3731)
        # Assigning a type to the variable 'if_condition_3732' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'if_condition_3732', if_condition_3732)
        # SSA begins for if statement (line 357)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to DistutilsPlatformError(...): (line 358)
        # Processing the call arguments (line 358)
        str_3734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 41), 'str', '--plat-name must be one of %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 359)
        tuple_3735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 359)
        # Adding element type (line 359)
        # Getting the type of 'ok_plats' (line 359)
        ok_plats_3736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 42), 'ok_plats', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 42), tuple_3735, ok_plats_3736)
        
        # Applying the binary operator '%' (line 358)
        result_mod_3737 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 41), '%', str_3734, tuple_3735)
        
        # Processing the call keyword arguments (line 358)
        kwargs_3738 = {}
        # Getting the type of 'DistutilsPlatformError' (line 358)
        DistutilsPlatformError_3733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 18), 'DistutilsPlatformError', False)
        # Calling DistutilsPlatformError(args, kwargs) (line 358)
        DistutilsPlatformError_call_result_3739 = invoke(stypy.reporting.localization.Localization(__file__, 358, 18), DistutilsPlatformError_3733, *[result_mod_3737], **kwargs_3738)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 358, 12), DistutilsPlatformError_call_result_3739, 'raise parameter', BaseException)
        # SSA join for if statement (line 357)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        str_3740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 11), 'str', 'DISTUTILS_USE_SDK')
        # Getting the type of 'os' (line 361)
        os_3741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 34), 'os')
        # Obtaining the member 'environ' of a type (line 361)
        environ_3742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 34), os_3741, 'environ')
        # Applying the binary operator 'in' (line 361)
        result_contains_3743 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 11), 'in', str_3740, environ_3742)
        
        
        str_3744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 49), 'str', 'MSSdk')
        # Getting the type of 'os' (line 361)
        os_3745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 60), 'os')
        # Obtaining the member 'environ' of a type (line 361)
        environ_3746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 60), os_3745, 'environ')
        # Applying the binary operator 'in' (line 361)
        result_contains_3747 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 49), 'in', str_3744, environ_3746)
        
        # Applying the binary operator 'and' (line 361)
        result_and_keyword_3748 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 11), 'and', result_contains_3743, result_contains_3747)
        
        # Call to find_exe(...): (line 361)
        # Processing the call arguments (line 361)
        str_3751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 89), 'str', 'cl.exe')
        # Processing the call keyword arguments (line 361)
        kwargs_3752 = {}
        # Getting the type of 'self' (line 361)
        self_3749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 75), 'self', False)
        # Obtaining the member 'find_exe' of a type (line 361)
        find_exe_3750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 75), self_3749, 'find_exe')
        # Calling find_exe(args, kwargs) (line 361)
        find_exe_call_result_3753 = invoke(stypy.reporting.localization.Localization(__file__, 361, 75), find_exe_3750, *[str_3751], **kwargs_3752)
        
        # Applying the binary operator 'and' (line 361)
        result_and_keyword_3754 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 11), 'and', result_and_keyword_3748, find_exe_call_result_3753)
        
        # Testing the type of an if condition (line 361)
        if_condition_3755 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 361, 8), result_and_keyword_3754)
        # Assigning a type to the variable 'if_condition_3755' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'if_condition_3755', if_condition_3755)
        # SSA begins for if statement (line 361)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Attribute (line 364):
        
        # Assigning a Str to a Attribute (line 364):
        str_3756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 22), 'str', 'cl.exe')
        # Getting the type of 'self' (line 364)
        self_3757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 12), 'self')
        # Setting the type of the member 'cc' of a type (line 364)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 12), self_3757, 'cc', str_3756)
        
        # Assigning a Str to a Attribute (line 365):
        
        # Assigning a Str to a Attribute (line 365):
        str_3758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 26), 'str', 'link.exe')
        # Getting the type of 'self' (line 365)
        self_3759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'self')
        # Setting the type of the member 'linker' of a type (line 365)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 12), self_3759, 'linker', str_3758)
        
        # Assigning a Str to a Attribute (line 366):
        
        # Assigning a Str to a Attribute (line 366):
        str_3760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 23), 'str', 'lib.exe')
        # Getting the type of 'self' (line 366)
        self_3761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'self')
        # Setting the type of the member 'lib' of a type (line 366)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 12), self_3761, 'lib', str_3760)
        
        # Assigning a Str to a Attribute (line 367):
        
        # Assigning a Str to a Attribute (line 367):
        str_3762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 22), 'str', 'rc.exe')
        # Getting the type of 'self' (line 367)
        self_3763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'self')
        # Setting the type of the member 'rc' of a type (line 367)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 12), self_3763, 'rc', str_3762)
        
        # Assigning a Str to a Attribute (line 368):
        
        # Assigning a Str to a Attribute (line 368):
        str_3764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 22), 'str', 'mc.exe')
        # Getting the type of 'self' (line 368)
        self_3765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 12), 'self')
        # Setting the type of the member 'mc' of a type (line 368)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 12), self_3765, 'mc', str_3764)
        # SSA branch for the else part of an if statement (line 361)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'plat_name' (line 375)
        plat_name_3766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 15), 'plat_name')
        
        # Call to get_platform(...): (line 375)
        # Processing the call keyword arguments (line 375)
        kwargs_3768 = {}
        # Getting the type of 'get_platform' (line 375)
        get_platform_3767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 28), 'get_platform', False)
        # Calling get_platform(args, kwargs) (line 375)
        get_platform_call_result_3769 = invoke(stypy.reporting.localization.Localization(__file__, 375, 28), get_platform_3767, *[], **kwargs_3768)
        
        # Applying the binary operator '==' (line 375)
        result_eq_3770 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 15), '==', plat_name_3766, get_platform_call_result_3769)
        
        
        # Getting the type of 'plat_name' (line 375)
        plat_name_3771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 46), 'plat_name')
        str_3772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 59), 'str', 'win32')
        # Applying the binary operator '==' (line 375)
        result_eq_3773 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 46), '==', plat_name_3771, str_3772)
        
        # Applying the binary operator 'or' (line 375)
        result_or_keyword_3774 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 15), 'or', result_eq_3770, result_eq_3773)
        
        # Testing the type of an if condition (line 375)
        if_condition_3775 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 375, 12), result_or_keyword_3774)
        # Assigning a type to the variable 'if_condition_3775' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 12), 'if_condition_3775', if_condition_3775)
        # SSA begins for if statement (line 375)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 377):
        
        # Assigning a Subscript to a Name (line 377):
        
        # Obtaining the type of the subscript
        # Getting the type of 'plat_name' (line 377)
        plat_name_3776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 43), 'plat_name')
        # Getting the type of 'PLAT_TO_VCVARS' (line 377)
        PLAT_TO_VCVARS_3777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 28), 'PLAT_TO_VCVARS')
        # Obtaining the member '__getitem__' of a type (line 377)
        getitem___3778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 28), PLAT_TO_VCVARS_3777, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 377)
        subscript_call_result_3779 = invoke(stypy.reporting.localization.Localization(__file__, 377, 28), getitem___3778, plat_name_3776)
        
        # Assigning a type to the variable 'plat_spec' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 16), 'plat_spec', subscript_call_result_3779)
        # SSA branch for the else part of an if statement (line 375)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 380):
        
        # Assigning a BinOp to a Name (line 380):
        
        # Obtaining the type of the subscript
        
        # Call to get_platform(...): (line 380)
        # Processing the call keyword arguments (line 380)
        kwargs_3781 = {}
        # Getting the type of 'get_platform' (line 380)
        get_platform_3780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 43), 'get_platform', False)
        # Calling get_platform(args, kwargs) (line 380)
        get_platform_call_result_3782 = invoke(stypy.reporting.localization.Localization(__file__, 380, 43), get_platform_3780, *[], **kwargs_3781)
        
        # Getting the type of 'PLAT_TO_VCVARS' (line 380)
        PLAT_TO_VCVARS_3783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 28), 'PLAT_TO_VCVARS')
        # Obtaining the member '__getitem__' of a type (line 380)
        getitem___3784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 28), PLAT_TO_VCVARS_3783, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 380)
        subscript_call_result_3785 = invoke(stypy.reporting.localization.Localization(__file__, 380, 28), getitem___3784, get_platform_call_result_3782)
        
        str_3786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 61), 'str', '_')
        # Applying the binary operator '+' (line 380)
        result_add_3787 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 28), '+', subscript_call_result_3785, str_3786)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'plat_name' (line 381)
        plat_name_3788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 43), 'plat_name')
        # Getting the type of 'PLAT_TO_VCVARS' (line 381)
        PLAT_TO_VCVARS_3789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 28), 'PLAT_TO_VCVARS')
        # Obtaining the member '__getitem__' of a type (line 381)
        getitem___3790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 28), PLAT_TO_VCVARS_3789, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 381)
        subscript_call_result_3791 = invoke(stypy.reporting.localization.Localization(__file__, 381, 28), getitem___3790, plat_name_3788)
        
        # Applying the binary operator '+' (line 380)
        result_add_3792 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 65), '+', result_add_3787, subscript_call_result_3791)
        
        # Assigning a type to the variable 'plat_spec' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 16), 'plat_spec', result_add_3792)
        # SSA join for if statement (line 375)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 383):
        
        # Assigning a Call to a Name (line 383):
        
        # Call to query_vcvarsall(...): (line 383)
        # Processing the call arguments (line 383)
        # Getting the type of 'VERSION' (line 383)
        VERSION_3794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 37), 'VERSION', False)
        # Getting the type of 'plat_spec' (line 383)
        plat_spec_3795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 46), 'plat_spec', False)
        # Processing the call keyword arguments (line 383)
        kwargs_3796 = {}
        # Getting the type of 'query_vcvarsall' (line 383)
        query_vcvarsall_3793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 21), 'query_vcvarsall', False)
        # Calling query_vcvarsall(args, kwargs) (line 383)
        query_vcvarsall_call_result_3797 = invoke(stypy.reporting.localization.Localization(__file__, 383, 21), query_vcvarsall_3793, *[VERSION_3794, plat_spec_3795], **kwargs_3796)
        
        # Assigning a type to the variable 'vc_env' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'vc_env', query_vcvarsall_call_result_3797)
        
        # Assigning a Call to a Attribute (line 386):
        
        # Assigning a Call to a Attribute (line 386):
        
        # Call to split(...): (line 386)
        # Processing the call arguments (line 386)
        # Getting the type of 'os' (line 386)
        os_3807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 63), 'os', False)
        # Obtaining the member 'pathsep' of a type (line 386)
        pathsep_3808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 63), os_3807, 'pathsep')
        # Processing the call keyword arguments (line 386)
        kwargs_3809 = {}
        
        # Call to encode(...): (line 386)
        # Processing the call arguments (line 386)
        str_3803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 49), 'str', 'mbcs')
        # Processing the call keyword arguments (line 386)
        kwargs_3804 = {}
        
        # Obtaining the type of the subscript
        str_3798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 34), 'str', 'path')
        # Getting the type of 'vc_env' (line 386)
        vc_env_3799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 27), 'vc_env', False)
        # Obtaining the member '__getitem__' of a type (line 386)
        getitem___3800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 27), vc_env_3799, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 386)
        subscript_call_result_3801 = invoke(stypy.reporting.localization.Localization(__file__, 386, 27), getitem___3800, str_3798)
        
        # Obtaining the member 'encode' of a type (line 386)
        encode_3802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 27), subscript_call_result_3801, 'encode')
        # Calling encode(args, kwargs) (line 386)
        encode_call_result_3805 = invoke(stypy.reporting.localization.Localization(__file__, 386, 27), encode_3802, *[str_3803], **kwargs_3804)
        
        # Obtaining the member 'split' of a type (line 386)
        split_3806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 27), encode_call_result_3805, 'split')
        # Calling split(args, kwargs) (line 386)
        split_call_result_3810 = invoke(stypy.reporting.localization.Localization(__file__, 386, 27), split_3806, *[pathsep_3808], **kwargs_3809)
        
        # Getting the type of 'self' (line 386)
        self_3811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'self')
        # Setting the type of the member '__paths' of a type (line 386)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 12), self_3811, '__paths', split_call_result_3810)
        
        # Assigning a Call to a Subscript (line 387):
        
        # Assigning a Call to a Subscript (line 387):
        
        # Call to encode(...): (line 387)
        # Processing the call arguments (line 387)
        str_3817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 53), 'str', 'mbcs')
        # Processing the call keyword arguments (line 387)
        kwargs_3818 = {}
        
        # Obtaining the type of the subscript
        str_3812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 39), 'str', 'lib')
        # Getting the type of 'vc_env' (line 387)
        vc_env_3813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 32), 'vc_env', False)
        # Obtaining the member '__getitem__' of a type (line 387)
        getitem___3814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 32), vc_env_3813, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 387)
        subscript_call_result_3815 = invoke(stypy.reporting.localization.Localization(__file__, 387, 32), getitem___3814, str_3812)
        
        # Obtaining the member 'encode' of a type (line 387)
        encode_3816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 32), subscript_call_result_3815, 'encode')
        # Calling encode(args, kwargs) (line 387)
        encode_call_result_3819 = invoke(stypy.reporting.localization.Localization(__file__, 387, 32), encode_3816, *[str_3817], **kwargs_3818)
        
        # Getting the type of 'os' (line 387)
        os_3820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'os')
        # Obtaining the member 'environ' of a type (line 387)
        environ_3821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 12), os_3820, 'environ')
        str_3822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 23), 'str', 'lib')
        # Storing an element on a container (line 387)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 12), environ_3821, (str_3822, encode_call_result_3819))
        
        # Assigning a Call to a Subscript (line 388):
        
        # Assigning a Call to a Subscript (line 388):
        
        # Call to encode(...): (line 388)
        # Processing the call arguments (line 388)
        str_3828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 61), 'str', 'mbcs')
        # Processing the call keyword arguments (line 388)
        kwargs_3829 = {}
        
        # Obtaining the type of the subscript
        str_3823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 43), 'str', 'include')
        # Getting the type of 'vc_env' (line 388)
        vc_env_3824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 36), 'vc_env', False)
        # Obtaining the member '__getitem__' of a type (line 388)
        getitem___3825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 36), vc_env_3824, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 388)
        subscript_call_result_3826 = invoke(stypy.reporting.localization.Localization(__file__, 388, 36), getitem___3825, str_3823)
        
        # Obtaining the member 'encode' of a type (line 388)
        encode_3827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 36), subscript_call_result_3826, 'encode')
        # Calling encode(args, kwargs) (line 388)
        encode_call_result_3830 = invoke(stypy.reporting.localization.Localization(__file__, 388, 36), encode_3827, *[str_3828], **kwargs_3829)
        
        # Getting the type of 'os' (line 388)
        os_3831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'os')
        # Obtaining the member 'environ' of a type (line 388)
        environ_3832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 12), os_3831, 'environ')
        str_3833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 23), 'str', 'include')
        # Storing an element on a container (line 388)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 12), environ_3832, (str_3833, encode_call_result_3830))
        
        
        
        # Call to len(...): (line 390)
        # Processing the call arguments (line 390)
        # Getting the type of 'self' (line 390)
        self_3835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 19), 'self', False)
        # Obtaining the member '__paths' of a type (line 390)
        paths_3836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 19), self_3835, '__paths')
        # Processing the call keyword arguments (line 390)
        kwargs_3837 = {}
        # Getting the type of 'len' (line 390)
        len_3834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 15), 'len', False)
        # Calling len(args, kwargs) (line 390)
        len_call_result_3838 = invoke(stypy.reporting.localization.Localization(__file__, 390, 15), len_3834, *[paths_3836], **kwargs_3837)
        
        int_3839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 36), 'int')
        # Applying the binary operator '==' (line 390)
        result_eq_3840 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 15), '==', len_call_result_3838, int_3839)
        
        # Testing the type of an if condition (line 390)
        if_condition_3841 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 390, 12), result_eq_3840)
        # Assigning a type to the variable 'if_condition_3841' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'if_condition_3841', if_condition_3841)
        # SSA begins for if statement (line 390)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to DistutilsPlatformError(...): (line 391)
        # Processing the call arguments (line 391)
        str_3843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 45), 'str', "Python was built with %s, and extensions need to be built with the same version of the compiler, but it isn't installed.")
        # Getting the type of 'self' (line 394)
        self_3844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 25), 'self', False)
        # Obtaining the member '__product' of a type (line 394)
        product_3845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 25), self_3844, '__product')
        # Applying the binary operator '%' (line 391)
        result_mod_3846 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 45), '%', str_3843, product_3845)
        
        # Processing the call keyword arguments (line 391)
        kwargs_3847 = {}
        # Getting the type of 'DistutilsPlatformError' (line 391)
        DistutilsPlatformError_3842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 22), 'DistutilsPlatformError', False)
        # Calling DistutilsPlatformError(args, kwargs) (line 391)
        DistutilsPlatformError_call_result_3848 = invoke(stypy.reporting.localization.Localization(__file__, 391, 22), DistutilsPlatformError_3842, *[result_mod_3846], **kwargs_3847)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 391, 16), DistutilsPlatformError_call_result_3848, 'raise parameter', BaseException)
        # SSA join for if statement (line 390)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 396):
        
        # Assigning a Call to a Attribute (line 396):
        
        # Call to find_exe(...): (line 396)
        # Processing the call arguments (line 396)
        str_3851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 36), 'str', 'cl.exe')
        # Processing the call keyword arguments (line 396)
        kwargs_3852 = {}
        # Getting the type of 'self' (line 396)
        self_3849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 22), 'self', False)
        # Obtaining the member 'find_exe' of a type (line 396)
        find_exe_3850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 22), self_3849, 'find_exe')
        # Calling find_exe(args, kwargs) (line 396)
        find_exe_call_result_3853 = invoke(stypy.reporting.localization.Localization(__file__, 396, 22), find_exe_3850, *[str_3851], **kwargs_3852)
        
        # Getting the type of 'self' (line 396)
        self_3854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'self')
        # Setting the type of the member 'cc' of a type (line 396)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 12), self_3854, 'cc', find_exe_call_result_3853)
        
        # Assigning a Call to a Attribute (line 397):
        
        # Assigning a Call to a Attribute (line 397):
        
        # Call to find_exe(...): (line 397)
        # Processing the call arguments (line 397)
        str_3857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 40), 'str', 'link.exe')
        # Processing the call keyword arguments (line 397)
        kwargs_3858 = {}
        # Getting the type of 'self' (line 397)
        self_3855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 26), 'self', False)
        # Obtaining the member 'find_exe' of a type (line 397)
        find_exe_3856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 26), self_3855, 'find_exe')
        # Calling find_exe(args, kwargs) (line 397)
        find_exe_call_result_3859 = invoke(stypy.reporting.localization.Localization(__file__, 397, 26), find_exe_3856, *[str_3857], **kwargs_3858)
        
        # Getting the type of 'self' (line 397)
        self_3860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'self')
        # Setting the type of the member 'linker' of a type (line 397)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 12), self_3860, 'linker', find_exe_call_result_3859)
        
        # Assigning a Call to a Attribute (line 398):
        
        # Assigning a Call to a Attribute (line 398):
        
        # Call to find_exe(...): (line 398)
        # Processing the call arguments (line 398)
        str_3863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 37), 'str', 'lib.exe')
        # Processing the call keyword arguments (line 398)
        kwargs_3864 = {}
        # Getting the type of 'self' (line 398)
        self_3861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 23), 'self', False)
        # Obtaining the member 'find_exe' of a type (line 398)
        find_exe_3862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 23), self_3861, 'find_exe')
        # Calling find_exe(args, kwargs) (line 398)
        find_exe_call_result_3865 = invoke(stypy.reporting.localization.Localization(__file__, 398, 23), find_exe_3862, *[str_3863], **kwargs_3864)
        
        # Getting the type of 'self' (line 398)
        self_3866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'self')
        # Setting the type of the member 'lib' of a type (line 398)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 12), self_3866, 'lib', find_exe_call_result_3865)
        
        # Assigning a Call to a Attribute (line 399):
        
        # Assigning a Call to a Attribute (line 399):
        
        # Call to find_exe(...): (line 399)
        # Processing the call arguments (line 399)
        str_3869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 36), 'str', 'rc.exe')
        # Processing the call keyword arguments (line 399)
        kwargs_3870 = {}
        # Getting the type of 'self' (line 399)
        self_3867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 22), 'self', False)
        # Obtaining the member 'find_exe' of a type (line 399)
        find_exe_3868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 22), self_3867, 'find_exe')
        # Calling find_exe(args, kwargs) (line 399)
        find_exe_call_result_3871 = invoke(stypy.reporting.localization.Localization(__file__, 399, 22), find_exe_3868, *[str_3869], **kwargs_3870)
        
        # Getting the type of 'self' (line 399)
        self_3872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'self')
        # Setting the type of the member 'rc' of a type (line 399)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 12), self_3872, 'rc', find_exe_call_result_3871)
        
        # Assigning a Call to a Attribute (line 400):
        
        # Assigning a Call to a Attribute (line 400):
        
        # Call to find_exe(...): (line 400)
        # Processing the call arguments (line 400)
        str_3875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 36), 'str', 'mc.exe')
        # Processing the call keyword arguments (line 400)
        kwargs_3876 = {}
        # Getting the type of 'self' (line 400)
        self_3873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 22), 'self', False)
        # Obtaining the member 'find_exe' of a type (line 400)
        find_exe_3874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 22), self_3873, 'find_exe')
        # Calling find_exe(args, kwargs) (line 400)
        find_exe_call_result_3877 = invoke(stypy.reporting.localization.Localization(__file__, 400, 22), find_exe_3874, *[str_3875], **kwargs_3876)
        
        # Getting the type of 'self' (line 400)
        self_3878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 12), 'self')
        # Setting the type of the member 'mc' of a type (line 400)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 12), self_3878, 'mc', find_exe_call_result_3877)
        # SSA join for if statement (line 361)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 405)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        
        # Call to split(...): (line 406)
        # Processing the call arguments (line 406)
        str_3885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 46), 'str', ';')
        # Processing the call keyword arguments (line 406)
        kwargs_3886 = {}
        
        # Obtaining the type of the subscript
        str_3879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 32), 'str', 'path')
        # Getting the type of 'os' (line 406)
        os_3880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 21), 'os', False)
        # Obtaining the member 'environ' of a type (line 406)
        environ_3881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 21), os_3880, 'environ')
        # Obtaining the member '__getitem__' of a type (line 406)
        getitem___3882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 21), environ_3881, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 406)
        subscript_call_result_3883 = invoke(stypy.reporting.localization.Localization(__file__, 406, 21), getitem___3882, str_3879)
        
        # Obtaining the member 'split' of a type (line 406)
        split_3884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 21), subscript_call_result_3883, 'split')
        # Calling split(args, kwargs) (line 406)
        split_call_result_3887 = invoke(stypy.reporting.localization.Localization(__file__, 406, 21), split_3884, *[str_3885], **kwargs_3886)
        
        # Testing the type of a for loop iterable (line 406)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 406, 12), split_call_result_3887)
        # Getting the type of the for loop variable (line 406)
        for_loop_var_3888 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 406, 12), split_call_result_3887)
        # Assigning a type to the variable 'p' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 12), 'p', for_loop_var_3888)
        # SSA begins for a for statement (line 406)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 407)
        # Processing the call arguments (line 407)
        # Getting the type of 'p' (line 407)
        p_3892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 36), 'p', False)
        # Processing the call keyword arguments (line 407)
        kwargs_3893 = {}
        # Getting the type of 'self' (line 407)
        self_3889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 16), 'self', False)
        # Obtaining the member '__paths' of a type (line 407)
        paths_3890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 16), self_3889, '__paths')
        # Obtaining the member 'append' of a type (line 407)
        append_3891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 16), paths_3890, 'append')
        # Calling append(args, kwargs) (line 407)
        append_call_result_3894 = invoke(stypy.reporting.localization.Localization(__file__, 407, 16), append_3891, *[p_3892], **kwargs_3893)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the except part of a try statement (line 405)
        # SSA branch for the except 'KeyError' branch of a try statement (line 405)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 405)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 410):
        
        # Assigning a Call to a Attribute (line 410):
        
        # Call to normalize_and_reduce_paths(...): (line 410)
        # Processing the call arguments (line 410)
        # Getting the type of 'self' (line 410)
        self_3896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 50), 'self', False)
        # Obtaining the member '__paths' of a type (line 410)
        paths_3897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 50), self_3896, '__paths')
        # Processing the call keyword arguments (line 410)
        kwargs_3898 = {}
        # Getting the type of 'normalize_and_reduce_paths' (line 410)
        normalize_and_reduce_paths_3895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 23), 'normalize_and_reduce_paths', False)
        # Calling normalize_and_reduce_paths(args, kwargs) (line 410)
        normalize_and_reduce_paths_call_result_3899 = invoke(stypy.reporting.localization.Localization(__file__, 410, 23), normalize_and_reduce_paths_3895, *[paths_3897], **kwargs_3898)
        
        # Getting the type of 'self' (line 410)
        self_3900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'self')
        # Setting the type of the member '__paths' of a type (line 410)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 8), self_3900, '__paths', normalize_and_reduce_paths_call_result_3899)
        
        # Assigning a Call to a Subscript (line 411):
        
        # Assigning a Call to a Subscript (line 411):
        
        # Call to join(...): (line 411)
        # Processing the call arguments (line 411)
        # Getting the type of 'self' (line 411)
        self_3903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 38), 'self', False)
        # Obtaining the member '__paths' of a type (line 411)
        paths_3904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 38), self_3903, '__paths')
        # Processing the call keyword arguments (line 411)
        kwargs_3905 = {}
        str_3901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 29), 'str', ';')
        # Obtaining the member 'join' of a type (line 411)
        join_3902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 29), str_3901, 'join')
        # Calling join(args, kwargs) (line 411)
        join_call_result_3906 = invoke(stypy.reporting.localization.Localization(__file__, 411, 29), join_3902, *[paths_3904], **kwargs_3905)
        
        # Getting the type of 'os' (line 411)
        os_3907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'os')
        # Obtaining the member 'environ' of a type (line 411)
        environ_3908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 8), os_3907, 'environ')
        str_3909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 19), 'str', 'path')
        # Storing an element on a container (line 411)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 8), environ_3908, (str_3909, join_call_result_3906))
        
        # Assigning a Name to a Attribute (line 413):
        
        # Assigning a Name to a Attribute (line 413):
        # Getting the type of 'None' (line 413)
        None_3910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 34), 'None')
        # Getting the type of 'self' (line 413)
        self_3911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'self')
        # Setting the type of the member 'preprocess_options' of a type (line 413)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 8), self_3911, 'preprocess_options', None_3910)
        
        
        # Getting the type of 'self' (line 414)
        self_3912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 11), 'self')
        # Obtaining the member '__arch' of a type (line 414)
        arch_3913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 11), self_3912, '__arch')
        str_3914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 26), 'str', 'x86')
        # Applying the binary operator '==' (line 414)
        result_eq_3915 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 11), '==', arch_3913, str_3914)
        
        # Testing the type of an if condition (line 414)
        if_condition_3916 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 414, 8), result_eq_3915)
        # Assigning a type to the variable 'if_condition_3916' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'if_condition_3916', if_condition_3916)
        # SSA begins for if statement (line 414)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Attribute (line 415):
        
        # Assigning a List to a Attribute (line 415):
        
        # Obtaining an instance of the builtin type 'list' (line 415)
        list_3917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 415)
        # Adding element type (line 415)
        str_3918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 37), 'str', '/nologo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 35), list_3917, str_3918)
        # Adding element type (line 415)
        str_3919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 48), 'str', '/Ox')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 35), list_3917, str_3919)
        # Adding element type (line 415)
        str_3920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 55), 'str', '/MD')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 35), list_3917, str_3920)
        # Adding element type (line 415)
        str_3921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 62), 'str', '/W3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 35), list_3917, str_3921)
        # Adding element type (line 415)
        str_3922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 37), 'str', '/DNDEBUG')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 35), list_3917, str_3922)
        
        # Getting the type of 'self' (line 415)
        self_3923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 12), 'self')
        # Setting the type of the member 'compile_options' of a type (line 415)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 12), self_3923, 'compile_options', list_3917)
        
        # Assigning a List to a Attribute (line 417):
        
        # Assigning a List to a Attribute (line 417):
        
        # Obtaining an instance of the builtin type 'list' (line 417)
        list_3924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 417)
        # Adding element type (line 417)
        str_3925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 42), 'str', '/nologo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 417, 41), list_3924, str_3925)
        # Adding element type (line 417)
        str_3926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 53), 'str', '/Od')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 417, 41), list_3924, str_3926)
        # Adding element type (line 417)
        str_3927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 60), 'str', '/MDd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 417, 41), list_3924, str_3927)
        # Adding element type (line 417)
        str_3928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 68), 'str', '/W3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 417, 41), list_3924, str_3928)
        # Adding element type (line 417)
        str_3929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 42), 'str', '/Z7')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 417, 41), list_3924, str_3929)
        # Adding element type (line 417)
        str_3930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 49), 'str', '/D_DEBUG')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 417, 41), list_3924, str_3930)
        
        # Getting the type of 'self' (line 417)
        self_3931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 12), 'self')
        # Setting the type of the member 'compile_options_debug' of a type (line 417)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 12), self_3931, 'compile_options_debug', list_3924)
        # SSA branch for the else part of an if statement (line 414)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Attribute (line 421):
        
        # Assigning a List to a Attribute (line 421):
        
        # Obtaining an instance of the builtin type 'list' (line 421)
        list_3932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 421)
        # Adding element type (line 421)
        str_3933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 37), 'str', '/nologo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 35), list_3932, str_3933)
        # Adding element type (line 421)
        str_3934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 48), 'str', '/Ox')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 35), list_3932, str_3934)
        # Adding element type (line 421)
        str_3935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 55), 'str', '/MD')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 35), list_3932, str_3935)
        # Adding element type (line 421)
        str_3936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 62), 'str', '/W3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 35), list_3932, str_3936)
        # Adding element type (line 421)
        str_3937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 69), 'str', '/GS-')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 35), list_3932, str_3937)
        # Adding element type (line 421)
        str_3938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 37), 'str', '/DNDEBUG')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 35), list_3932, str_3938)
        
        # Getting the type of 'self' (line 421)
        self_3939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 12), 'self')
        # Setting the type of the member 'compile_options' of a type (line 421)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 12), self_3939, 'compile_options', list_3932)
        
        # Assigning a List to a Attribute (line 423):
        
        # Assigning a List to a Attribute (line 423):
        
        # Obtaining an instance of the builtin type 'list' (line 423)
        list_3940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 423)
        # Adding element type (line 423)
        str_3941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 42), 'str', '/nologo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 41), list_3940, str_3941)
        # Adding element type (line 423)
        str_3942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 53), 'str', '/Od')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 41), list_3940, str_3942)
        # Adding element type (line 423)
        str_3943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 60), 'str', '/MDd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 41), list_3940, str_3943)
        # Adding element type (line 423)
        str_3944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 68), 'str', '/W3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 41), list_3940, str_3944)
        # Adding element type (line 423)
        str_3945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 75), 'str', '/GS-')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 41), list_3940, str_3945)
        # Adding element type (line 423)
        str_3946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 42), 'str', '/Z7')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 41), list_3940, str_3946)
        # Adding element type (line 423)
        str_3947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 49), 'str', '/D_DEBUG')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 41), list_3940, str_3947)
        
        # Getting the type of 'self' (line 423)
        self_3948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'self')
        # Setting the type of the member 'compile_options_debug' of a type (line 423)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 12), self_3948, 'compile_options_debug', list_3940)
        # SSA join for if statement (line 414)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Attribute (line 426):
        
        # Assigning a List to a Attribute (line 426):
        
        # Obtaining an instance of the builtin type 'list' (line 426)
        list_3949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 426)
        # Adding element type (line 426)
        str_3950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 31), 'str', '/DLL')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 30), list_3949, str_3950)
        # Adding element type (line 426)
        str_3951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 39), 'str', '/nologo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 30), list_3949, str_3951)
        # Adding element type (line 426)
        str_3952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 50), 'str', '/INCREMENTAL:NO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 30), list_3949, str_3952)
        
        # Getting the type of 'self' (line 426)
        self_3953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'self')
        # Setting the type of the member 'ldflags_shared' of a type (line 426)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 8), self_3953, 'ldflags_shared', list_3949)
        
        
        # Getting the type of 'self' (line 427)
        self_3954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 11), 'self')
        # Obtaining the member '__version' of a type (line 427)
        version_3955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 11), self_3954, '__version')
        int_3956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 29), 'int')
        # Applying the binary operator '>=' (line 427)
        result_ge_3957 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 11), '>=', version_3955, int_3956)
        
        # Testing the type of an if condition (line 427)
        if_condition_3958 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 427, 8), result_ge_3957)
        # Assigning a type to the variable 'if_condition_3958' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'if_condition_3958', if_condition_3958)
        # SSA begins for if statement (line 427)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Attribute (line 428):
        
        # Assigning a List to a Attribute (line 428):
        
        # Obtaining an instance of the builtin type 'list' (line 428)
        list_3959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 428)
        # Adding element type (line 428)
        str_3960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 16), 'str', '/DLL')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 40), list_3959, str_3960)
        # Adding element type (line 428)
        str_3961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 24), 'str', '/nologo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 40), list_3959, str_3961)
        # Adding element type (line 428)
        str_3962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 35), 'str', '/INCREMENTAL:no')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 40), list_3959, str_3962)
        # Adding element type (line 428)
        str_3963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 54), 'str', '/DEBUG')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 40), list_3959, str_3963)
        
        # Getting the type of 'self' (line 428)
        self_3964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 12), 'self')
        # Setting the type of the member 'ldflags_shared_debug' of a type (line 428)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 12), self_3964, 'ldflags_shared_debug', list_3959)
        # SSA join for if statement (line 427)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Attribute (line 431):
        
        # Assigning a List to a Attribute (line 431):
        
        # Obtaining an instance of the builtin type 'list' (line 431)
        list_3965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 431)
        # Adding element type (line 431)
        str_3966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 32), 'str', '/nologo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 30), list_3965, str_3966)
        
        # Getting the type of 'self' (line 431)
        self_3967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'self')
        # Setting the type of the member 'ldflags_static' of a type (line 431)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 8), self_3967, 'ldflags_static', list_3965)
        
        # Assigning a Name to a Attribute (line 433):
        
        # Assigning a Name to a Attribute (line 433):
        # Getting the type of 'True' (line 433)
        True_3968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 27), 'True')
        # Getting the type of 'self' (line 433)
        self_3969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'self')
        # Setting the type of the member 'initialized' of a type (line 433)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 8), self_3969, 'initialized', True_3968)
        
        # ################# End of 'initialize(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize' in the type store
        # Getting the type of 'stypy_return_type' (line 350)
        stypy_return_type_3970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3970)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize'
        return stypy_return_type_3970


    @norecursion
    def object_filenames(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_3971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 35), 'int')
        str_3972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 36), 'str', '')
        defaults = [int_3971, str_3972]
        # Create a new context for function 'object_filenames'
        module_type_store = module_type_store.open_function_context('object_filenames', 437, 4, False)
        # Assigning a type to the variable 'self' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 4), 'self', type_of_self)
        
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

        
        # Type idiom detected: calculating its left and rigth part (line 443)
        # Getting the type of 'output_dir' (line 443)
        output_dir_3973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 11), 'output_dir')
        # Getting the type of 'None' (line 443)
        None_3974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 25), 'None')
        
        (may_be_3975, more_types_in_union_3976) = may_be_none(output_dir_3973, None_3974)

        if may_be_3975:

            if more_types_in_union_3976:
                # Runtime conditional SSA (line 443)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Str to a Name (line 443):
            
            # Assigning a Str to a Name (line 443):
            str_3977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 44), 'str', '')
            # Assigning a type to the variable 'output_dir' (line 443)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 31), 'output_dir', str_3977)

            if more_types_in_union_3976:
                # SSA join for if statement (line 443)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a List to a Name (line 444):
        
        # Assigning a List to a Name (line 444):
        
        # Obtaining an instance of the builtin type 'list' (line 444)
        list_3978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 444)
        
        # Assigning a type to the variable 'obj_names' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'obj_names', list_3978)
        
        # Getting the type of 'source_filenames' (line 445)
        source_filenames_3979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 24), 'source_filenames')
        # Testing the type of a for loop iterable (line 445)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 445, 8), source_filenames_3979)
        # Getting the type of the for loop variable (line 445)
        for_loop_var_3980 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 445, 8), source_filenames_3979)
        # Assigning a type to the variable 'src_name' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'src_name', for_loop_var_3980)
        # SSA begins for a for statement (line 445)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 446):
        
        # Assigning a Subscript to a Name (line 446):
        
        # Obtaining the type of the subscript
        int_3981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 12), 'int')
        
        # Call to splitext(...): (line 446)
        # Processing the call arguments (line 446)
        # Getting the type of 'src_name' (line 446)
        src_name_3985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 44), 'src_name', False)
        # Processing the call keyword arguments (line 446)
        kwargs_3986 = {}
        # Getting the type of 'os' (line 446)
        os_3982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 446)
        path_3983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 26), os_3982, 'path')
        # Obtaining the member 'splitext' of a type (line 446)
        splitext_3984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 26), path_3983, 'splitext')
        # Calling splitext(args, kwargs) (line 446)
        splitext_call_result_3987 = invoke(stypy.reporting.localization.Localization(__file__, 446, 26), splitext_3984, *[src_name_3985], **kwargs_3986)
        
        # Obtaining the member '__getitem__' of a type (line 446)
        getitem___3988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 12), splitext_call_result_3987, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 446)
        subscript_call_result_3989 = invoke(stypy.reporting.localization.Localization(__file__, 446, 12), getitem___3988, int_3981)
        
        # Assigning a type to the variable 'tuple_var_assignment_2874' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 12), 'tuple_var_assignment_2874', subscript_call_result_3989)
        
        # Assigning a Subscript to a Name (line 446):
        
        # Obtaining the type of the subscript
        int_3990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 12), 'int')
        
        # Call to splitext(...): (line 446)
        # Processing the call arguments (line 446)
        # Getting the type of 'src_name' (line 446)
        src_name_3994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 44), 'src_name', False)
        # Processing the call keyword arguments (line 446)
        kwargs_3995 = {}
        # Getting the type of 'os' (line 446)
        os_3991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 446)
        path_3992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 26), os_3991, 'path')
        # Obtaining the member 'splitext' of a type (line 446)
        splitext_3993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 26), path_3992, 'splitext')
        # Calling splitext(args, kwargs) (line 446)
        splitext_call_result_3996 = invoke(stypy.reporting.localization.Localization(__file__, 446, 26), splitext_3993, *[src_name_3994], **kwargs_3995)
        
        # Obtaining the member '__getitem__' of a type (line 446)
        getitem___3997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 12), splitext_call_result_3996, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 446)
        subscript_call_result_3998 = invoke(stypy.reporting.localization.Localization(__file__, 446, 12), getitem___3997, int_3990)
        
        # Assigning a type to the variable 'tuple_var_assignment_2875' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 12), 'tuple_var_assignment_2875', subscript_call_result_3998)
        
        # Assigning a Name to a Name (line 446):
        # Getting the type of 'tuple_var_assignment_2874' (line 446)
        tuple_var_assignment_2874_3999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 12), 'tuple_var_assignment_2874')
        # Assigning a type to the variable 'base' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 13), 'base', tuple_var_assignment_2874_3999)
        
        # Assigning a Name to a Name (line 446):
        # Getting the type of 'tuple_var_assignment_2875' (line 446)
        tuple_var_assignment_2875_4000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 12), 'tuple_var_assignment_2875')
        # Assigning a type to the variable 'ext' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 19), 'ext', tuple_var_assignment_2875_4000)
        
        # Assigning a Subscript to a Name (line 447):
        
        # Assigning a Subscript to a Name (line 447):
        
        # Obtaining the type of the subscript
        int_4001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 44), 'int')
        
        # Call to splitdrive(...): (line 447)
        # Processing the call arguments (line 447)
        # Getting the type of 'base' (line 447)
        base_4005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 38), 'base', False)
        # Processing the call keyword arguments (line 447)
        kwargs_4006 = {}
        # Getting the type of 'os' (line 447)
        os_4002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 447)
        path_4003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 19), os_4002, 'path')
        # Obtaining the member 'splitdrive' of a type (line 447)
        splitdrive_4004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 19), path_4003, 'splitdrive')
        # Calling splitdrive(args, kwargs) (line 447)
        splitdrive_call_result_4007 = invoke(stypy.reporting.localization.Localization(__file__, 447, 19), splitdrive_4004, *[base_4005], **kwargs_4006)
        
        # Obtaining the member '__getitem__' of a type (line 447)
        getitem___4008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 19), splitdrive_call_result_4007, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 447)
        subscript_call_result_4009 = invoke(stypy.reporting.localization.Localization(__file__, 447, 19), getitem___4008, int_4001)
        
        # Assigning a type to the variable 'base' (line 447)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'base', subscript_call_result_4009)
        
        # Assigning a Subscript to a Name (line 448):
        
        # Assigning a Subscript to a Name (line 448):
        
        # Obtaining the type of the subscript
        
        # Call to isabs(...): (line 448)
        # Processing the call arguments (line 448)
        # Getting the type of 'base' (line 448)
        base_4013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 38), 'base', False)
        # Processing the call keyword arguments (line 448)
        kwargs_4014 = {}
        # Getting the type of 'os' (line 448)
        os_4010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 448)
        path_4011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 24), os_4010, 'path')
        # Obtaining the member 'isabs' of a type (line 448)
        isabs_4012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 24), path_4011, 'isabs')
        # Calling isabs(args, kwargs) (line 448)
        isabs_call_result_4015 = invoke(stypy.reporting.localization.Localization(__file__, 448, 24), isabs_4012, *[base_4013], **kwargs_4014)
        
        slice_4016 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 448, 19), isabs_call_result_4015, None, None)
        # Getting the type of 'base' (line 448)
        base_4017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 19), 'base')
        # Obtaining the member '__getitem__' of a type (line 448)
        getitem___4018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 19), base_4017, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 448)
        subscript_call_result_4019 = invoke(stypy.reporting.localization.Localization(__file__, 448, 19), getitem___4018, slice_4016)
        
        # Assigning a type to the variable 'base' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 12), 'base', subscript_call_result_4019)
        
        
        # Getting the type of 'ext' (line 449)
        ext_4020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 15), 'ext')
        # Getting the type of 'self' (line 449)
        self_4021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 26), 'self')
        # Obtaining the member 'src_extensions' of a type (line 449)
        src_extensions_4022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 26), self_4021, 'src_extensions')
        # Applying the binary operator 'notin' (line 449)
        result_contains_4023 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 15), 'notin', ext_4020, src_extensions_4022)
        
        # Testing the type of an if condition (line 449)
        if_condition_4024 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 449, 12), result_contains_4023)
        # Assigning a type to the variable 'if_condition_4024' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 12), 'if_condition_4024', if_condition_4024)
        # SSA begins for if statement (line 449)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to CompileError(...): (line 453)
        # Processing the call arguments (line 453)
        str_4026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 36), 'str', "Don't know how to compile %s")
        # Getting the type of 'src_name' (line 453)
        src_name_4027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 69), 'src_name', False)
        # Applying the binary operator '%' (line 453)
        result_mod_4028 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 36), '%', str_4026, src_name_4027)
        
        # Processing the call keyword arguments (line 453)
        kwargs_4029 = {}
        # Getting the type of 'CompileError' (line 453)
        CompileError_4025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 22), 'CompileError', False)
        # Calling CompileError(args, kwargs) (line 453)
        CompileError_call_result_4030 = invoke(stypy.reporting.localization.Localization(__file__, 453, 22), CompileError_4025, *[result_mod_4028], **kwargs_4029)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 453, 16), CompileError_call_result_4030, 'raise parameter', BaseException)
        # SSA join for if statement (line 449)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'strip_dir' (line 454)
        strip_dir_4031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 15), 'strip_dir')
        # Testing the type of an if condition (line 454)
        if_condition_4032 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 454, 12), strip_dir_4031)
        # Assigning a type to the variable 'if_condition_4032' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 12), 'if_condition_4032', if_condition_4032)
        # SSA begins for if statement (line 454)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 455):
        
        # Assigning a Call to a Name (line 455):
        
        # Call to basename(...): (line 455)
        # Processing the call arguments (line 455)
        # Getting the type of 'base' (line 455)
        base_4036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 41), 'base', False)
        # Processing the call keyword arguments (line 455)
        kwargs_4037 = {}
        # Getting the type of 'os' (line 455)
        os_4033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 455)
        path_4034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 23), os_4033, 'path')
        # Obtaining the member 'basename' of a type (line 455)
        basename_4035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 23), path_4034, 'basename')
        # Calling basename(args, kwargs) (line 455)
        basename_call_result_4038 = invoke(stypy.reporting.localization.Localization(__file__, 455, 23), basename_4035, *[base_4036], **kwargs_4037)
        
        # Assigning a type to the variable 'base' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 16), 'base', basename_call_result_4038)
        # SSA join for if statement (line 454)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'ext' (line 456)
        ext_4039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 15), 'ext')
        # Getting the type of 'self' (line 456)
        self_4040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 22), 'self')
        # Obtaining the member '_rc_extensions' of a type (line 456)
        _rc_extensions_4041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 22), self_4040, '_rc_extensions')
        # Applying the binary operator 'in' (line 456)
        result_contains_4042 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 15), 'in', ext_4039, _rc_extensions_4041)
        
        # Testing the type of an if condition (line 456)
        if_condition_4043 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 456, 12), result_contains_4042)
        # Assigning a type to the variable 'if_condition_4043' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 12), 'if_condition_4043', if_condition_4043)
        # SSA begins for if statement (line 456)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 457)
        # Processing the call arguments (line 457)
        
        # Call to join(...): (line 457)
        # Processing the call arguments (line 457)
        # Getting the type of 'output_dir' (line 457)
        output_dir_4049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 48), 'output_dir', False)
        # Getting the type of 'base' (line 458)
        base_4050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 48), 'base', False)
        # Getting the type of 'self' (line 458)
        self_4051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 55), 'self', False)
        # Obtaining the member 'res_extension' of a type (line 458)
        res_extension_4052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 55), self_4051, 'res_extension')
        # Applying the binary operator '+' (line 458)
        result_add_4053 = python_operator(stypy.reporting.localization.Localization(__file__, 458, 48), '+', base_4050, res_extension_4052)
        
        # Processing the call keyword arguments (line 457)
        kwargs_4054 = {}
        # Getting the type of 'os' (line 457)
        os_4046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 34), 'os', False)
        # Obtaining the member 'path' of a type (line 457)
        path_4047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 34), os_4046, 'path')
        # Obtaining the member 'join' of a type (line 457)
        join_4048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 34), path_4047, 'join')
        # Calling join(args, kwargs) (line 457)
        join_call_result_4055 = invoke(stypy.reporting.localization.Localization(__file__, 457, 34), join_4048, *[output_dir_4049, result_add_4053], **kwargs_4054)
        
        # Processing the call keyword arguments (line 457)
        kwargs_4056 = {}
        # Getting the type of 'obj_names' (line 457)
        obj_names_4044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 16), 'obj_names', False)
        # Obtaining the member 'append' of a type (line 457)
        append_4045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 16), obj_names_4044, 'append')
        # Calling append(args, kwargs) (line 457)
        append_call_result_4057 = invoke(stypy.reporting.localization.Localization(__file__, 457, 16), append_4045, *[join_call_result_4055], **kwargs_4056)
        
        # SSA branch for the else part of an if statement (line 456)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'ext' (line 459)
        ext_4058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 17), 'ext')
        # Getting the type of 'self' (line 459)
        self_4059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 24), 'self')
        # Obtaining the member '_mc_extensions' of a type (line 459)
        _mc_extensions_4060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 24), self_4059, '_mc_extensions')
        # Applying the binary operator 'in' (line 459)
        result_contains_4061 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 17), 'in', ext_4058, _mc_extensions_4060)
        
        # Testing the type of an if condition (line 459)
        if_condition_4062 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 459, 17), result_contains_4061)
        # Assigning a type to the variable 'if_condition_4062' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 17), 'if_condition_4062', if_condition_4062)
        # SSA begins for if statement (line 459)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 460)
        # Processing the call arguments (line 460)
        
        # Call to join(...): (line 460)
        # Processing the call arguments (line 460)
        # Getting the type of 'output_dir' (line 460)
        output_dir_4068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 48), 'output_dir', False)
        # Getting the type of 'base' (line 461)
        base_4069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 48), 'base', False)
        # Getting the type of 'self' (line 461)
        self_4070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 55), 'self', False)
        # Obtaining the member 'res_extension' of a type (line 461)
        res_extension_4071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 55), self_4070, 'res_extension')
        # Applying the binary operator '+' (line 461)
        result_add_4072 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 48), '+', base_4069, res_extension_4071)
        
        # Processing the call keyword arguments (line 460)
        kwargs_4073 = {}
        # Getting the type of 'os' (line 460)
        os_4065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 34), 'os', False)
        # Obtaining the member 'path' of a type (line 460)
        path_4066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 34), os_4065, 'path')
        # Obtaining the member 'join' of a type (line 460)
        join_4067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 34), path_4066, 'join')
        # Calling join(args, kwargs) (line 460)
        join_call_result_4074 = invoke(stypy.reporting.localization.Localization(__file__, 460, 34), join_4067, *[output_dir_4068, result_add_4072], **kwargs_4073)
        
        # Processing the call keyword arguments (line 460)
        kwargs_4075 = {}
        # Getting the type of 'obj_names' (line 460)
        obj_names_4063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 16), 'obj_names', False)
        # Obtaining the member 'append' of a type (line 460)
        append_4064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 16), obj_names_4063, 'append')
        # Calling append(args, kwargs) (line 460)
        append_call_result_4076 = invoke(stypy.reporting.localization.Localization(__file__, 460, 16), append_4064, *[join_call_result_4074], **kwargs_4075)
        
        # SSA branch for the else part of an if statement (line 459)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 463)
        # Processing the call arguments (line 463)
        
        # Call to join(...): (line 463)
        # Processing the call arguments (line 463)
        # Getting the type of 'output_dir' (line 463)
        output_dir_4082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 48), 'output_dir', False)
        # Getting the type of 'base' (line 464)
        base_4083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 48), 'base', False)
        # Getting the type of 'self' (line 464)
        self_4084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 55), 'self', False)
        # Obtaining the member 'obj_extension' of a type (line 464)
        obj_extension_4085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 55), self_4084, 'obj_extension')
        # Applying the binary operator '+' (line 464)
        result_add_4086 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 48), '+', base_4083, obj_extension_4085)
        
        # Processing the call keyword arguments (line 463)
        kwargs_4087 = {}
        # Getting the type of 'os' (line 463)
        os_4079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 34), 'os', False)
        # Obtaining the member 'path' of a type (line 463)
        path_4080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 34), os_4079, 'path')
        # Obtaining the member 'join' of a type (line 463)
        join_4081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 34), path_4080, 'join')
        # Calling join(args, kwargs) (line 463)
        join_call_result_4088 = invoke(stypy.reporting.localization.Localization(__file__, 463, 34), join_4081, *[output_dir_4082, result_add_4086], **kwargs_4087)
        
        # Processing the call keyword arguments (line 463)
        kwargs_4089 = {}
        # Getting the type of 'obj_names' (line 463)
        obj_names_4077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 16), 'obj_names', False)
        # Obtaining the member 'append' of a type (line 463)
        append_4078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 16), obj_names_4077, 'append')
        # Calling append(args, kwargs) (line 463)
        append_call_result_4090 = invoke(stypy.reporting.localization.Localization(__file__, 463, 16), append_4078, *[join_call_result_4088], **kwargs_4089)
        
        # SSA join for if statement (line 459)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 456)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'obj_names' (line 465)
        obj_names_4091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 15), 'obj_names')
        # Assigning a type to the variable 'stypy_return_type' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'stypy_return_type', obj_names_4091)
        
        # ################# End of 'object_filenames(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'object_filenames' in the type store
        # Getting the type of 'stypy_return_type' (line 437)
        stypy_return_type_4092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4092)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'object_filenames'
        return stypy_return_type_4092


    @norecursion
    def compile(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 469)
        None_4093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 27), 'None')
        # Getting the type of 'None' (line 469)
        None_4094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 40), 'None')
        # Getting the type of 'None' (line 469)
        None_4095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 59), 'None')
        int_4096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 71), 'int')
        # Getting the type of 'None' (line 470)
        None_4097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 30), 'None')
        # Getting the type of 'None' (line 470)
        None_4098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 51), 'None')
        # Getting the type of 'None' (line 470)
        None_4099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 65), 'None')
        defaults = [None_4093, None_4094, None_4095, int_4096, None_4097, None_4098, None_4099]
        # Create a new context for function 'compile'
        module_type_store = module_type_store.open_function_context('compile', 468, 4, False)
        # Assigning a type to the variable 'self' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 4), 'self', type_of_self)
        
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

        
        
        # Getting the type of 'self' (line 472)
        self_4100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 15), 'self')
        # Obtaining the member 'initialized' of a type (line 472)
        initialized_4101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 15), self_4100, 'initialized')
        # Applying the 'not' unary operator (line 472)
        result_not__4102 = python_operator(stypy.reporting.localization.Localization(__file__, 472, 11), 'not', initialized_4101)
        
        # Testing the type of an if condition (line 472)
        if_condition_4103 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 472, 8), result_not__4102)
        # Assigning a type to the variable 'if_condition_4103' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'if_condition_4103', if_condition_4103)
        # SSA begins for if statement (line 472)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to initialize(...): (line 473)
        # Processing the call keyword arguments (line 473)
        kwargs_4106 = {}
        # Getting the type of 'self' (line 473)
        self_4104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 12), 'self', False)
        # Obtaining the member 'initialize' of a type (line 473)
        initialize_4105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 12), self_4104, 'initialize')
        # Calling initialize(args, kwargs) (line 473)
        initialize_call_result_4107 = invoke(stypy.reporting.localization.Localization(__file__, 473, 12), initialize_4105, *[], **kwargs_4106)
        
        # SSA join for if statement (line 472)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 474):
        
        # Assigning a Call to a Name (line 474):
        
        # Call to _setup_compile(...): (line 474)
        # Processing the call arguments (line 474)
        # Getting the type of 'output_dir' (line 474)
        output_dir_4110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 43), 'output_dir', False)
        # Getting the type of 'macros' (line 474)
        macros_4111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 55), 'macros', False)
        # Getting the type of 'include_dirs' (line 474)
        include_dirs_4112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 63), 'include_dirs', False)
        # Getting the type of 'sources' (line 475)
        sources_4113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 43), 'sources', False)
        # Getting the type of 'depends' (line 475)
        depends_4114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 52), 'depends', False)
        # Getting the type of 'extra_postargs' (line 475)
        extra_postargs_4115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 61), 'extra_postargs', False)
        # Processing the call keyword arguments (line 474)
        kwargs_4116 = {}
        # Getting the type of 'self' (line 474)
        self_4108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 23), 'self', False)
        # Obtaining the member '_setup_compile' of a type (line 474)
        _setup_compile_4109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 23), self_4108, '_setup_compile')
        # Calling _setup_compile(args, kwargs) (line 474)
        _setup_compile_call_result_4117 = invoke(stypy.reporting.localization.Localization(__file__, 474, 23), _setup_compile_4109, *[output_dir_4110, macros_4111, include_dirs_4112, sources_4113, depends_4114, extra_postargs_4115], **kwargs_4116)
        
        # Assigning a type to the variable 'compile_info' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'compile_info', _setup_compile_call_result_4117)
        
        # Assigning a Name to a Tuple (line 476):
        
        # Assigning a Subscript to a Name (line 476):
        
        # Obtaining the type of the subscript
        int_4118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 8), 'int')
        # Getting the type of 'compile_info' (line 476)
        compile_info_4119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 58), 'compile_info')
        # Obtaining the member '__getitem__' of a type (line 476)
        getitem___4120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 8), compile_info_4119, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 476)
        subscript_call_result_4121 = invoke(stypy.reporting.localization.Localization(__file__, 476, 8), getitem___4120, int_4118)
        
        # Assigning a type to the variable 'tuple_var_assignment_2876' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'tuple_var_assignment_2876', subscript_call_result_4121)
        
        # Assigning a Subscript to a Name (line 476):
        
        # Obtaining the type of the subscript
        int_4122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 8), 'int')
        # Getting the type of 'compile_info' (line 476)
        compile_info_4123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 58), 'compile_info')
        # Obtaining the member '__getitem__' of a type (line 476)
        getitem___4124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 8), compile_info_4123, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 476)
        subscript_call_result_4125 = invoke(stypy.reporting.localization.Localization(__file__, 476, 8), getitem___4124, int_4122)
        
        # Assigning a type to the variable 'tuple_var_assignment_2877' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'tuple_var_assignment_2877', subscript_call_result_4125)
        
        # Assigning a Subscript to a Name (line 476):
        
        # Obtaining the type of the subscript
        int_4126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 8), 'int')
        # Getting the type of 'compile_info' (line 476)
        compile_info_4127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 58), 'compile_info')
        # Obtaining the member '__getitem__' of a type (line 476)
        getitem___4128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 8), compile_info_4127, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 476)
        subscript_call_result_4129 = invoke(stypy.reporting.localization.Localization(__file__, 476, 8), getitem___4128, int_4126)
        
        # Assigning a type to the variable 'tuple_var_assignment_2878' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'tuple_var_assignment_2878', subscript_call_result_4129)
        
        # Assigning a Subscript to a Name (line 476):
        
        # Obtaining the type of the subscript
        int_4130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 8), 'int')
        # Getting the type of 'compile_info' (line 476)
        compile_info_4131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 58), 'compile_info')
        # Obtaining the member '__getitem__' of a type (line 476)
        getitem___4132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 8), compile_info_4131, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 476)
        subscript_call_result_4133 = invoke(stypy.reporting.localization.Localization(__file__, 476, 8), getitem___4132, int_4130)
        
        # Assigning a type to the variable 'tuple_var_assignment_2879' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'tuple_var_assignment_2879', subscript_call_result_4133)
        
        # Assigning a Subscript to a Name (line 476):
        
        # Obtaining the type of the subscript
        int_4134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 8), 'int')
        # Getting the type of 'compile_info' (line 476)
        compile_info_4135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 58), 'compile_info')
        # Obtaining the member '__getitem__' of a type (line 476)
        getitem___4136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 8), compile_info_4135, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 476)
        subscript_call_result_4137 = invoke(stypy.reporting.localization.Localization(__file__, 476, 8), getitem___4136, int_4134)
        
        # Assigning a type to the variable 'tuple_var_assignment_2880' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'tuple_var_assignment_2880', subscript_call_result_4137)
        
        # Assigning a Name to a Name (line 476):
        # Getting the type of 'tuple_var_assignment_2876' (line 476)
        tuple_var_assignment_2876_4138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'tuple_var_assignment_2876')
        # Assigning a type to the variable 'macros' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'macros', tuple_var_assignment_2876_4138)
        
        # Assigning a Name to a Name (line 476):
        # Getting the type of 'tuple_var_assignment_2877' (line 476)
        tuple_var_assignment_2877_4139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'tuple_var_assignment_2877')
        # Assigning a type to the variable 'objects' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 16), 'objects', tuple_var_assignment_2877_4139)
        
        # Assigning a Name to a Name (line 476):
        # Getting the type of 'tuple_var_assignment_2878' (line 476)
        tuple_var_assignment_2878_4140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'tuple_var_assignment_2878')
        # Assigning a type to the variable 'extra_postargs' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 25), 'extra_postargs', tuple_var_assignment_2878_4140)
        
        # Assigning a Name to a Name (line 476):
        # Getting the type of 'tuple_var_assignment_2879' (line 476)
        tuple_var_assignment_2879_4141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'tuple_var_assignment_2879')
        # Assigning a type to the variable 'pp_opts' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 41), 'pp_opts', tuple_var_assignment_2879_4141)
        
        # Assigning a Name to a Name (line 476):
        # Getting the type of 'tuple_var_assignment_2880' (line 476)
        tuple_var_assignment_2880_4142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'tuple_var_assignment_2880')
        # Assigning a type to the variable 'build' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 50), 'build', tuple_var_assignment_2880_4142)
        
        # Assigning a BoolOp to a Name (line 478):
        
        # Assigning a BoolOp to a Name (line 478):
        
        # Evaluating a boolean operation
        # Getting the type of 'extra_preargs' (line 478)
        extra_preargs_4143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 23), 'extra_preargs')
        
        # Obtaining an instance of the builtin type 'list' (line 478)
        list_4144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 478)
        
        # Applying the binary operator 'or' (line 478)
        result_or_keyword_4145 = python_operator(stypy.reporting.localization.Localization(__file__, 478, 23), 'or', extra_preargs_4143, list_4144)
        
        # Assigning a type to the variable 'compile_opts' (line 478)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'compile_opts', result_or_keyword_4145)
        
        # Call to append(...): (line 479)
        # Processing the call arguments (line 479)
        str_4148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 29), 'str', '/c')
        # Processing the call keyword arguments (line 479)
        kwargs_4149 = {}
        # Getting the type of 'compile_opts' (line 479)
        compile_opts_4146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'compile_opts', False)
        # Obtaining the member 'append' of a type (line 479)
        append_4147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 8), compile_opts_4146, 'append')
        # Calling append(args, kwargs) (line 479)
        append_call_result_4150 = invoke(stypy.reporting.localization.Localization(__file__, 479, 8), append_4147, *[str_4148], **kwargs_4149)
        
        
        # Getting the type of 'debug' (line 480)
        debug_4151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 11), 'debug')
        # Testing the type of an if condition (line 480)
        if_condition_4152 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 480, 8), debug_4151)
        # Assigning a type to the variable 'if_condition_4152' (line 480)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'if_condition_4152', if_condition_4152)
        # SSA begins for if statement (line 480)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 481)
        # Processing the call arguments (line 481)
        # Getting the type of 'self' (line 481)
        self_4155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 32), 'self', False)
        # Obtaining the member 'compile_options_debug' of a type (line 481)
        compile_options_debug_4156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 32), self_4155, 'compile_options_debug')
        # Processing the call keyword arguments (line 481)
        kwargs_4157 = {}
        # Getting the type of 'compile_opts' (line 481)
        compile_opts_4153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 12), 'compile_opts', False)
        # Obtaining the member 'extend' of a type (line 481)
        extend_4154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 12), compile_opts_4153, 'extend')
        # Calling extend(args, kwargs) (line 481)
        extend_call_result_4158 = invoke(stypy.reporting.localization.Localization(__file__, 481, 12), extend_4154, *[compile_options_debug_4156], **kwargs_4157)
        
        # SSA branch for the else part of an if statement (line 480)
        module_type_store.open_ssa_branch('else')
        
        # Call to extend(...): (line 483)
        # Processing the call arguments (line 483)
        # Getting the type of 'self' (line 483)
        self_4161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 32), 'self', False)
        # Obtaining the member 'compile_options' of a type (line 483)
        compile_options_4162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 32), self_4161, 'compile_options')
        # Processing the call keyword arguments (line 483)
        kwargs_4163 = {}
        # Getting the type of 'compile_opts' (line 483)
        compile_opts_4159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 12), 'compile_opts', False)
        # Obtaining the member 'extend' of a type (line 483)
        extend_4160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 12), compile_opts_4159, 'extend')
        # Calling extend(args, kwargs) (line 483)
        extend_call_result_4164 = invoke(stypy.reporting.localization.Localization(__file__, 483, 12), extend_4160, *[compile_options_4162], **kwargs_4163)
        
        # SSA join for if statement (line 480)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'objects' (line 485)
        objects_4165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 19), 'objects')
        # Testing the type of a for loop iterable (line 485)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 485, 8), objects_4165)
        # Getting the type of the for loop variable (line 485)
        for_loop_var_4166 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 485, 8), objects_4165)
        # Assigning a type to the variable 'obj' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'obj', for_loop_var_4166)
        # SSA begins for a for statement (line 485)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # SSA begins for try-except statement (line 486)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Tuple (line 487):
        
        # Assigning a Subscript to a Name (line 487):
        
        # Obtaining the type of the subscript
        int_4167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 16), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'obj' (line 487)
        obj_4168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 33), 'obj')
        # Getting the type of 'build' (line 487)
        build_4169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 27), 'build')
        # Obtaining the member '__getitem__' of a type (line 487)
        getitem___4170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 27), build_4169, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 487)
        subscript_call_result_4171 = invoke(stypy.reporting.localization.Localization(__file__, 487, 27), getitem___4170, obj_4168)
        
        # Obtaining the member '__getitem__' of a type (line 487)
        getitem___4172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 16), subscript_call_result_4171, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 487)
        subscript_call_result_4173 = invoke(stypy.reporting.localization.Localization(__file__, 487, 16), getitem___4172, int_4167)
        
        # Assigning a type to the variable 'tuple_var_assignment_2881' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 16), 'tuple_var_assignment_2881', subscript_call_result_4173)
        
        # Assigning a Subscript to a Name (line 487):
        
        # Obtaining the type of the subscript
        int_4174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 16), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'obj' (line 487)
        obj_4175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 33), 'obj')
        # Getting the type of 'build' (line 487)
        build_4176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 27), 'build')
        # Obtaining the member '__getitem__' of a type (line 487)
        getitem___4177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 27), build_4176, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 487)
        subscript_call_result_4178 = invoke(stypy.reporting.localization.Localization(__file__, 487, 27), getitem___4177, obj_4175)
        
        # Obtaining the member '__getitem__' of a type (line 487)
        getitem___4179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 16), subscript_call_result_4178, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 487)
        subscript_call_result_4180 = invoke(stypy.reporting.localization.Localization(__file__, 487, 16), getitem___4179, int_4174)
        
        # Assigning a type to the variable 'tuple_var_assignment_2882' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 16), 'tuple_var_assignment_2882', subscript_call_result_4180)
        
        # Assigning a Name to a Name (line 487):
        # Getting the type of 'tuple_var_assignment_2881' (line 487)
        tuple_var_assignment_2881_4181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 16), 'tuple_var_assignment_2881')
        # Assigning a type to the variable 'src' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 16), 'src', tuple_var_assignment_2881_4181)
        
        # Assigning a Name to a Name (line 487):
        # Getting the type of 'tuple_var_assignment_2882' (line 487)
        tuple_var_assignment_2882_4182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 16), 'tuple_var_assignment_2882')
        # Assigning a type to the variable 'ext' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 21), 'ext', tuple_var_assignment_2882_4182)
        # SSA branch for the except part of a try statement (line 486)
        # SSA branch for the except 'KeyError' branch of a try statement (line 486)
        module_type_store.open_ssa_branch('except')
        # SSA join for try-except statement (line 486)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'debug' (line 490)
        debug_4183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 15), 'debug')
        # Testing the type of an if condition (line 490)
        if_condition_4184 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 490, 12), debug_4183)
        # Assigning a type to the variable 'if_condition_4184' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'if_condition_4184', if_condition_4184)
        # SSA begins for if statement (line 490)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 494):
        
        # Assigning a Call to a Name (line 494):
        
        # Call to abspath(...): (line 494)
        # Processing the call arguments (line 494)
        # Getting the type of 'src' (line 494)
        src_4188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 38), 'src', False)
        # Processing the call keyword arguments (line 494)
        kwargs_4189 = {}
        # Getting the type of 'os' (line 494)
        os_4185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 22), 'os', False)
        # Obtaining the member 'path' of a type (line 494)
        path_4186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 22), os_4185, 'path')
        # Obtaining the member 'abspath' of a type (line 494)
        abspath_4187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 22), path_4186, 'abspath')
        # Calling abspath(args, kwargs) (line 494)
        abspath_call_result_4190 = invoke(stypy.reporting.localization.Localization(__file__, 494, 22), abspath_4187, *[src_4188], **kwargs_4189)
        
        # Assigning a type to the variable 'src' (line 494)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 16), 'src', abspath_call_result_4190)
        # SSA join for if statement (line 490)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'ext' (line 496)
        ext_4191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 15), 'ext')
        # Getting the type of 'self' (line 496)
        self_4192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 22), 'self')
        # Obtaining the member '_c_extensions' of a type (line 496)
        _c_extensions_4193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 22), self_4192, '_c_extensions')
        # Applying the binary operator 'in' (line 496)
        result_contains_4194 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 15), 'in', ext_4191, _c_extensions_4193)
        
        # Testing the type of an if condition (line 496)
        if_condition_4195 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 496, 12), result_contains_4194)
        # Assigning a type to the variable 'if_condition_4195' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'if_condition_4195', if_condition_4195)
        # SSA begins for if statement (line 496)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 497):
        
        # Assigning a BinOp to a Name (line 497):
        str_4196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 28), 'str', '/Tc')
        # Getting the type of 'src' (line 497)
        src_4197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 36), 'src')
        # Applying the binary operator '+' (line 497)
        result_add_4198 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 28), '+', str_4196, src_4197)
        
        # Assigning a type to the variable 'input_opt' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 16), 'input_opt', result_add_4198)
        # SSA branch for the else part of an if statement (line 496)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'ext' (line 498)
        ext_4199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 17), 'ext')
        # Getting the type of 'self' (line 498)
        self_4200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 24), 'self')
        # Obtaining the member '_cpp_extensions' of a type (line 498)
        _cpp_extensions_4201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 24), self_4200, '_cpp_extensions')
        # Applying the binary operator 'in' (line 498)
        result_contains_4202 = python_operator(stypy.reporting.localization.Localization(__file__, 498, 17), 'in', ext_4199, _cpp_extensions_4201)
        
        # Testing the type of an if condition (line 498)
        if_condition_4203 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 498, 17), result_contains_4202)
        # Assigning a type to the variable 'if_condition_4203' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 17), 'if_condition_4203', if_condition_4203)
        # SSA begins for if statement (line 498)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 499):
        
        # Assigning a BinOp to a Name (line 499):
        str_4204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 28), 'str', '/Tp')
        # Getting the type of 'src' (line 499)
        src_4205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 36), 'src')
        # Applying the binary operator '+' (line 499)
        result_add_4206 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 28), '+', str_4204, src_4205)
        
        # Assigning a type to the variable 'input_opt' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 16), 'input_opt', result_add_4206)
        # SSA branch for the else part of an if statement (line 498)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'ext' (line 500)
        ext_4207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 17), 'ext')
        # Getting the type of 'self' (line 500)
        self_4208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 24), 'self')
        # Obtaining the member '_rc_extensions' of a type (line 500)
        _rc_extensions_4209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 24), self_4208, '_rc_extensions')
        # Applying the binary operator 'in' (line 500)
        result_contains_4210 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 17), 'in', ext_4207, _rc_extensions_4209)
        
        # Testing the type of an if condition (line 500)
        if_condition_4211 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 500, 17), result_contains_4210)
        # Assigning a type to the variable 'if_condition_4211' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 17), 'if_condition_4211', if_condition_4211)
        # SSA begins for if statement (line 500)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 502):
        
        # Assigning a Name to a Name (line 502):
        # Getting the type of 'src' (line 502)
        src_4212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 28), 'src')
        # Assigning a type to the variable 'input_opt' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 16), 'input_opt', src_4212)
        
        # Assigning a BinOp to a Name (line 503):
        
        # Assigning a BinOp to a Name (line 503):
        str_4213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 29), 'str', '/fo')
        # Getting the type of 'obj' (line 503)
        obj_4214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 37), 'obj')
        # Applying the binary operator '+' (line 503)
        result_add_4215 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 29), '+', str_4213, obj_4214)
        
        # Assigning a type to the variable 'output_opt' (line 503)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 16), 'output_opt', result_add_4215)
        
        
        # SSA begins for try-except statement (line 504)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to spawn(...): (line 505)
        # Processing the call arguments (line 505)
        
        # Obtaining an instance of the builtin type 'list' (line 505)
        list_4218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 505)
        # Adding element type (line 505)
        # Getting the type of 'self' (line 505)
        self_4219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 32), 'self', False)
        # Obtaining the member 'rc' of a type (line 505)
        rc_4220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 32), self_4219, 'rc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 31), list_4218, rc_4220)
        
        # Getting the type of 'pp_opts' (line 505)
        pp_opts_4221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 43), 'pp_opts', False)
        # Applying the binary operator '+' (line 505)
        result_add_4222 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 31), '+', list_4218, pp_opts_4221)
        
        
        # Obtaining an instance of the builtin type 'list' (line 506)
        list_4223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 506)
        # Adding element type (line 506)
        # Getting the type of 'output_opt' (line 506)
        output_opt_4224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 32), 'output_opt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 506, 31), list_4223, output_opt_4224)
        
        # Applying the binary operator '+' (line 505)
        result_add_4225 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 51), '+', result_add_4222, list_4223)
        
        
        # Obtaining an instance of the builtin type 'list' (line 506)
        list_4226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 506)
        # Adding element type (line 506)
        # Getting the type of 'input_opt' (line 506)
        input_opt_4227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 47), 'input_opt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 506, 46), list_4226, input_opt_4227)
        
        # Applying the binary operator '+' (line 506)
        result_add_4228 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 44), '+', result_add_4225, list_4226)
        
        # Processing the call keyword arguments (line 505)
        kwargs_4229 = {}
        # Getting the type of 'self' (line 505)
        self_4216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 20), 'self', False)
        # Obtaining the member 'spawn' of a type (line 505)
        spawn_4217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 20), self_4216, 'spawn')
        # Calling spawn(args, kwargs) (line 505)
        spawn_call_result_4230 = invoke(stypy.reporting.localization.Localization(__file__, 505, 20), spawn_4217, *[result_add_4228], **kwargs_4229)
        
        # SSA branch for the except part of a try statement (line 504)
        # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 504)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsExecError' (line 507)
        DistutilsExecError_4231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 23), 'DistutilsExecError')
        # Assigning a type to the variable 'msg' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 16), 'msg', DistutilsExecError_4231)
        
        # Call to CompileError(...): (line 508)
        # Processing the call arguments (line 508)
        # Getting the type of 'msg' (line 508)
        msg_4233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 39), 'msg', False)
        # Processing the call keyword arguments (line 508)
        kwargs_4234 = {}
        # Getting the type of 'CompileError' (line 508)
        CompileError_4232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 26), 'CompileError', False)
        # Calling CompileError(args, kwargs) (line 508)
        CompileError_call_result_4235 = invoke(stypy.reporting.localization.Localization(__file__, 508, 26), CompileError_4232, *[msg_4233], **kwargs_4234)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 508, 20), CompileError_call_result_4235, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 504)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 500)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'ext' (line 510)
        ext_4236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 17), 'ext')
        # Getting the type of 'self' (line 510)
        self_4237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 24), 'self')
        # Obtaining the member '_mc_extensions' of a type (line 510)
        _mc_extensions_4238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 24), self_4237, '_mc_extensions')
        # Applying the binary operator 'in' (line 510)
        result_contains_4239 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 17), 'in', ext_4236, _mc_extensions_4238)
        
        # Testing the type of an if condition (line 510)
        if_condition_4240 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 510, 17), result_contains_4239)
        # Assigning a type to the variable 'if_condition_4240' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 17), 'if_condition_4240', if_condition_4240)
        # SSA begins for if statement (line 510)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 522):
        
        # Assigning a Call to a Name (line 522):
        
        # Call to dirname(...): (line 522)
        # Processing the call arguments (line 522)
        # Getting the type of 'src' (line 522)
        src_4244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 40), 'src', False)
        # Processing the call keyword arguments (line 522)
        kwargs_4245 = {}
        # Getting the type of 'os' (line 522)
        os_4241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 522)
        path_4242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 24), os_4241, 'path')
        # Obtaining the member 'dirname' of a type (line 522)
        dirname_4243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 24), path_4242, 'dirname')
        # Calling dirname(args, kwargs) (line 522)
        dirname_call_result_4246 = invoke(stypy.reporting.localization.Localization(__file__, 522, 24), dirname_4243, *[src_4244], **kwargs_4245)
        
        # Assigning a type to the variable 'h_dir' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 16), 'h_dir', dirname_call_result_4246)
        
        # Assigning a Call to a Name (line 523):
        
        # Assigning a Call to a Name (line 523):
        
        # Call to dirname(...): (line 523)
        # Processing the call arguments (line 523)
        # Getting the type of 'obj' (line 523)
        obj_4250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 41), 'obj', False)
        # Processing the call keyword arguments (line 523)
        kwargs_4251 = {}
        # Getting the type of 'os' (line 523)
        os_4247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 523)
        path_4248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 25), os_4247, 'path')
        # Obtaining the member 'dirname' of a type (line 523)
        dirname_4249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 25), path_4248, 'dirname')
        # Calling dirname(args, kwargs) (line 523)
        dirname_call_result_4252 = invoke(stypy.reporting.localization.Localization(__file__, 523, 25), dirname_4249, *[obj_4250], **kwargs_4251)
        
        # Assigning a type to the variable 'rc_dir' (line 523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 16), 'rc_dir', dirname_call_result_4252)
        
        
        # SSA begins for try-except statement (line 524)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to spawn(...): (line 526)
        # Processing the call arguments (line 526)
        
        # Obtaining an instance of the builtin type 'list' (line 526)
        list_4255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 526)
        # Adding element type (line 526)
        # Getting the type of 'self' (line 526)
        self_4256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 32), 'self', False)
        # Obtaining the member 'mc' of a type (line 526)
        mc_4257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 32), self_4256, 'mc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 526, 31), list_4255, mc_4257)
        
        
        # Obtaining an instance of the builtin type 'list' (line 527)
        list_4258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 527)
        # Adding element type (line 527)
        str_4259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 32), 'str', '-h')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 31), list_4258, str_4259)
        # Adding element type (line 527)
        # Getting the type of 'h_dir' (line 527)
        h_dir_4260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 38), 'h_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 31), list_4258, h_dir_4260)
        # Adding element type (line 527)
        str_4261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 45), 'str', '-r')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 31), list_4258, str_4261)
        # Adding element type (line 527)
        # Getting the type of 'rc_dir' (line 527)
        rc_dir_4262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 51), 'rc_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 31), list_4258, rc_dir_4262)
        
        # Applying the binary operator '+' (line 526)
        result_add_4263 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 31), '+', list_4255, list_4258)
        
        
        # Obtaining an instance of the builtin type 'list' (line 527)
        list_4264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 61), 'list')
        # Adding type elements to the builtin type 'list' instance (line 527)
        # Adding element type (line 527)
        # Getting the type of 'src' (line 527)
        src_4265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 62), 'src', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 61), list_4264, src_4265)
        
        # Applying the binary operator '+' (line 527)
        result_add_4266 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 59), '+', result_add_4263, list_4264)
        
        # Processing the call keyword arguments (line 526)
        kwargs_4267 = {}
        # Getting the type of 'self' (line 526)
        self_4253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 20), 'self', False)
        # Obtaining the member 'spawn' of a type (line 526)
        spawn_4254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 20), self_4253, 'spawn')
        # Calling spawn(args, kwargs) (line 526)
        spawn_call_result_4268 = invoke(stypy.reporting.localization.Localization(__file__, 526, 20), spawn_4254, *[result_add_4266], **kwargs_4267)
        
        
        # Assigning a Call to a Tuple (line 528):
        
        # Assigning a Subscript to a Name (line 528):
        
        # Obtaining the type of the subscript
        int_4269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 20), 'int')
        
        # Call to splitext(...): (line 528)
        # Processing the call arguments (line 528)
        
        # Call to basename(...): (line 528)
        # Processing the call arguments (line 528)
        # Getting the type of 'src' (line 528)
        src_4276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 66), 'src', False)
        # Processing the call keyword arguments (line 528)
        kwargs_4277 = {}
        # Getting the type of 'os' (line 528)
        os_4273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 48), 'os', False)
        # Obtaining the member 'path' of a type (line 528)
        path_4274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 48), os_4273, 'path')
        # Obtaining the member 'basename' of a type (line 528)
        basename_4275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 48), path_4274, 'basename')
        # Calling basename(args, kwargs) (line 528)
        basename_call_result_4278 = invoke(stypy.reporting.localization.Localization(__file__, 528, 48), basename_4275, *[src_4276], **kwargs_4277)
        
        # Processing the call keyword arguments (line 528)
        kwargs_4279 = {}
        # Getting the type of 'os' (line 528)
        os_4270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 30), 'os', False)
        # Obtaining the member 'path' of a type (line 528)
        path_4271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 30), os_4270, 'path')
        # Obtaining the member 'splitext' of a type (line 528)
        splitext_4272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 30), path_4271, 'splitext')
        # Calling splitext(args, kwargs) (line 528)
        splitext_call_result_4280 = invoke(stypy.reporting.localization.Localization(__file__, 528, 30), splitext_4272, *[basename_call_result_4278], **kwargs_4279)
        
        # Obtaining the member '__getitem__' of a type (line 528)
        getitem___4281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 20), splitext_call_result_4280, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 528)
        subscript_call_result_4282 = invoke(stypy.reporting.localization.Localization(__file__, 528, 20), getitem___4281, int_4269)
        
        # Assigning a type to the variable 'tuple_var_assignment_2883' (line 528)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 20), 'tuple_var_assignment_2883', subscript_call_result_4282)
        
        # Assigning a Subscript to a Name (line 528):
        
        # Obtaining the type of the subscript
        int_4283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 20), 'int')
        
        # Call to splitext(...): (line 528)
        # Processing the call arguments (line 528)
        
        # Call to basename(...): (line 528)
        # Processing the call arguments (line 528)
        # Getting the type of 'src' (line 528)
        src_4290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 66), 'src', False)
        # Processing the call keyword arguments (line 528)
        kwargs_4291 = {}
        # Getting the type of 'os' (line 528)
        os_4287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 48), 'os', False)
        # Obtaining the member 'path' of a type (line 528)
        path_4288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 48), os_4287, 'path')
        # Obtaining the member 'basename' of a type (line 528)
        basename_4289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 48), path_4288, 'basename')
        # Calling basename(args, kwargs) (line 528)
        basename_call_result_4292 = invoke(stypy.reporting.localization.Localization(__file__, 528, 48), basename_4289, *[src_4290], **kwargs_4291)
        
        # Processing the call keyword arguments (line 528)
        kwargs_4293 = {}
        # Getting the type of 'os' (line 528)
        os_4284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 30), 'os', False)
        # Obtaining the member 'path' of a type (line 528)
        path_4285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 30), os_4284, 'path')
        # Obtaining the member 'splitext' of a type (line 528)
        splitext_4286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 30), path_4285, 'splitext')
        # Calling splitext(args, kwargs) (line 528)
        splitext_call_result_4294 = invoke(stypy.reporting.localization.Localization(__file__, 528, 30), splitext_4286, *[basename_call_result_4292], **kwargs_4293)
        
        # Obtaining the member '__getitem__' of a type (line 528)
        getitem___4295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 20), splitext_call_result_4294, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 528)
        subscript_call_result_4296 = invoke(stypy.reporting.localization.Localization(__file__, 528, 20), getitem___4295, int_4283)
        
        # Assigning a type to the variable 'tuple_var_assignment_2884' (line 528)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 20), 'tuple_var_assignment_2884', subscript_call_result_4296)
        
        # Assigning a Name to a Name (line 528):
        # Getting the type of 'tuple_var_assignment_2883' (line 528)
        tuple_var_assignment_2883_4297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 20), 'tuple_var_assignment_2883')
        # Assigning a type to the variable 'base' (line 528)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 20), 'base', tuple_var_assignment_2883_4297)
        
        # Assigning a Name to a Name (line 528):
        # Getting the type of 'tuple_var_assignment_2884' (line 528)
        tuple_var_assignment_2884_4298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 20), 'tuple_var_assignment_2884')
        # Assigning a type to the variable '_' (line 528)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 26), '_', tuple_var_assignment_2884_4298)
        
        # Assigning a Call to a Name (line 529):
        
        # Assigning a Call to a Name (line 529):
        
        # Call to join(...): (line 529)
        # Processing the call arguments (line 529)
        # Getting the type of 'rc_dir' (line 529)
        rc_dir_4302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 44), 'rc_dir', False)
        # Getting the type of 'base' (line 529)
        base_4303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 52), 'base', False)
        str_4304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 59), 'str', '.rc')
        # Applying the binary operator '+' (line 529)
        result_add_4305 = python_operator(stypy.reporting.localization.Localization(__file__, 529, 52), '+', base_4303, str_4304)
        
        # Processing the call keyword arguments (line 529)
        kwargs_4306 = {}
        # Getting the type of 'os' (line 529)
        os_4299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 30), 'os', False)
        # Obtaining the member 'path' of a type (line 529)
        path_4300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 30), os_4299, 'path')
        # Obtaining the member 'join' of a type (line 529)
        join_4301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 30), path_4300, 'join')
        # Calling join(args, kwargs) (line 529)
        join_call_result_4307 = invoke(stypy.reporting.localization.Localization(__file__, 529, 30), join_4301, *[rc_dir_4302, result_add_4305], **kwargs_4306)
        
        # Assigning a type to the variable 'rc_file' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 20), 'rc_file', join_call_result_4307)
        
        # Call to spawn(...): (line 531)
        # Processing the call arguments (line 531)
        
        # Obtaining an instance of the builtin type 'list' (line 531)
        list_4310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 531)
        # Adding element type (line 531)
        # Getting the type of 'self' (line 531)
        self_4311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 32), 'self', False)
        # Obtaining the member 'rc' of a type (line 531)
        rc_4312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 32), self_4311, 'rc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 31), list_4310, rc_4312)
        
        
        # Obtaining an instance of the builtin type 'list' (line 532)
        list_4313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 532)
        # Adding element type (line 532)
        str_4314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 32), 'str', '/fo')
        # Getting the type of 'obj' (line 532)
        obj_4315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 40), 'obj', False)
        # Applying the binary operator '+' (line 532)
        result_add_4316 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 32), '+', str_4314, obj_4315)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 31), list_4313, result_add_4316)
        
        # Applying the binary operator '+' (line 531)
        result_add_4317 = python_operator(stypy.reporting.localization.Localization(__file__, 531, 31), '+', list_4310, list_4313)
        
        
        # Obtaining an instance of the builtin type 'list' (line 532)
        list_4318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 532)
        # Adding element type (line 532)
        # Getting the type of 'rc_file' (line 532)
        rc_file_4319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 48), 'rc_file', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 47), list_4318, rc_file_4319)
        
        # Applying the binary operator '+' (line 532)
        result_add_4320 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 45), '+', result_add_4317, list_4318)
        
        # Processing the call keyword arguments (line 531)
        kwargs_4321 = {}
        # Getting the type of 'self' (line 531)
        self_4308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 20), 'self', False)
        # Obtaining the member 'spawn' of a type (line 531)
        spawn_4309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 20), self_4308, 'spawn')
        # Calling spawn(args, kwargs) (line 531)
        spawn_call_result_4322 = invoke(stypy.reporting.localization.Localization(__file__, 531, 20), spawn_4309, *[result_add_4320], **kwargs_4321)
        
        # SSA branch for the except part of a try statement (line 524)
        # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 524)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsExecError' (line 534)
        DistutilsExecError_4323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 23), 'DistutilsExecError')
        # Assigning a type to the variable 'msg' (line 534)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 16), 'msg', DistutilsExecError_4323)
        
        # Call to CompileError(...): (line 535)
        # Processing the call arguments (line 535)
        # Getting the type of 'msg' (line 535)
        msg_4325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 39), 'msg', False)
        # Processing the call keyword arguments (line 535)
        kwargs_4326 = {}
        # Getting the type of 'CompileError' (line 535)
        CompileError_4324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 26), 'CompileError', False)
        # Calling CompileError(args, kwargs) (line 535)
        CompileError_call_result_4327 = invoke(stypy.reporting.localization.Localization(__file__, 535, 26), CompileError_4324, *[msg_4325], **kwargs_4326)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 535, 20), CompileError_call_result_4327, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 524)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 510)
        module_type_store.open_ssa_branch('else')
        
        # Call to CompileError(...): (line 539)
        # Processing the call arguments (line 539)
        str_4329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 35), 'str', "Don't know how to compile %s to %s")
        
        # Obtaining an instance of the builtin type 'tuple' (line 540)
        tuple_4330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 540)
        # Adding element type (line 540)
        # Getting the type of 'src' (line 540)
        src_4331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 38), 'src', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 540, 38), tuple_4330, src_4331)
        # Adding element type (line 540)
        # Getting the type of 'obj' (line 540)
        obj_4332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 43), 'obj', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 540, 38), tuple_4330, obj_4332)
        
        # Applying the binary operator '%' (line 539)
        result_mod_4333 = python_operator(stypy.reporting.localization.Localization(__file__, 539, 35), '%', str_4329, tuple_4330)
        
        # Processing the call keyword arguments (line 539)
        kwargs_4334 = {}
        # Getting the type of 'CompileError' (line 539)
        CompileError_4328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 22), 'CompileError', False)
        # Calling CompileError(args, kwargs) (line 539)
        CompileError_call_result_4335 = invoke(stypy.reporting.localization.Localization(__file__, 539, 22), CompileError_4328, *[result_mod_4333], **kwargs_4334)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 539, 16), CompileError_call_result_4335, 'raise parameter', BaseException)
        # SSA join for if statement (line 510)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 500)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 498)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 496)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 542):
        
        # Assigning a BinOp to a Name (line 542):
        str_4336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 25), 'str', '/Fo')
        # Getting the type of 'obj' (line 542)
        obj_4337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 33), 'obj')
        # Applying the binary operator '+' (line 542)
        result_add_4338 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 25), '+', str_4336, obj_4337)
        
        # Assigning a type to the variable 'output_opt' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 12), 'output_opt', result_add_4338)
        
        
        # SSA begins for try-except statement (line 543)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to spawn(...): (line 544)
        # Processing the call arguments (line 544)
        
        # Obtaining an instance of the builtin type 'list' (line 544)
        list_4341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 544)
        # Adding element type (line 544)
        # Getting the type of 'self' (line 544)
        self_4342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 28), 'self', False)
        # Obtaining the member 'cc' of a type (line 544)
        cc_4343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 28), self_4342, 'cc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 544, 27), list_4341, cc_4343)
        
        # Getting the type of 'compile_opts' (line 544)
        compile_opts_4344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 39), 'compile_opts', False)
        # Applying the binary operator '+' (line 544)
        result_add_4345 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 27), '+', list_4341, compile_opts_4344)
        
        # Getting the type of 'pp_opts' (line 544)
        pp_opts_4346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 54), 'pp_opts', False)
        # Applying the binary operator '+' (line 544)
        result_add_4347 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 52), '+', result_add_4345, pp_opts_4346)
        
        
        # Obtaining an instance of the builtin type 'list' (line 545)
        list_4348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 545)
        # Adding element type (line 545)
        # Getting the type of 'input_opt' (line 545)
        input_opt_4349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 28), 'input_opt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 545, 27), list_4348, input_opt_4349)
        # Adding element type (line 545)
        # Getting the type of 'output_opt' (line 545)
        output_opt_4350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 39), 'output_opt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 545, 27), list_4348, output_opt_4350)
        
        # Applying the binary operator '+' (line 544)
        result_add_4351 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 62), '+', result_add_4347, list_4348)
        
        # Getting the type of 'extra_postargs' (line 546)
        extra_postargs_4352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 27), 'extra_postargs', False)
        # Applying the binary operator '+' (line 545)
        result_add_4353 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 51), '+', result_add_4351, extra_postargs_4352)
        
        # Processing the call keyword arguments (line 544)
        kwargs_4354 = {}
        # Getting the type of 'self' (line 544)
        self_4339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 16), 'self', False)
        # Obtaining the member 'spawn' of a type (line 544)
        spawn_4340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 16), self_4339, 'spawn')
        # Calling spawn(args, kwargs) (line 544)
        spawn_call_result_4355 = invoke(stypy.reporting.localization.Localization(__file__, 544, 16), spawn_4340, *[result_add_4353], **kwargs_4354)
        
        # SSA branch for the except part of a try statement (line 543)
        # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 543)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsExecError' (line 547)
        DistutilsExecError_4356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 19), 'DistutilsExecError')
        # Assigning a type to the variable 'msg' (line 547)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 12), 'msg', DistutilsExecError_4356)
        
        # Call to CompileError(...): (line 548)
        # Processing the call arguments (line 548)
        # Getting the type of 'msg' (line 548)
        msg_4358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 35), 'msg', False)
        # Processing the call keyword arguments (line 548)
        kwargs_4359 = {}
        # Getting the type of 'CompileError' (line 548)
        CompileError_4357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 22), 'CompileError', False)
        # Calling CompileError(args, kwargs) (line 548)
        CompileError_call_result_4360 = invoke(stypy.reporting.localization.Localization(__file__, 548, 22), CompileError_4357, *[msg_4358], **kwargs_4359)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 548, 16), CompileError_call_result_4360, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 543)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'objects' (line 550)
        objects_4361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 15), 'objects')
        # Assigning a type to the variable 'stypy_return_type' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'stypy_return_type', objects_4361)
        
        # ################# End of 'compile(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'compile' in the type store
        # Getting the type of 'stypy_return_type' (line 468)
        stypy_return_type_4362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4362)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'compile'
        return stypy_return_type_4362


    @norecursion
    def create_static_lib(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 556)
        None_4363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 37), 'None')
        int_4364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 32), 'int')
        # Getting the type of 'None' (line 558)
        None_4365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 38), 'None')
        defaults = [None_4363, int_4364, None_4365]
        # Create a new context for function 'create_static_lib'
        module_type_store = module_type_store.open_function_context('create_static_lib', 553, 4, False)
        # Assigning a type to the variable 'self' (line 554)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'self', type_of_self)
        
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

        
        
        # Getting the type of 'self' (line 560)
        self_4366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 15), 'self')
        # Obtaining the member 'initialized' of a type (line 560)
        initialized_4367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 15), self_4366, 'initialized')
        # Applying the 'not' unary operator (line 560)
        result_not__4368 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 11), 'not', initialized_4367)
        
        # Testing the type of an if condition (line 560)
        if_condition_4369 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 560, 8), result_not__4368)
        # Assigning a type to the variable 'if_condition_4369' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'if_condition_4369', if_condition_4369)
        # SSA begins for if statement (line 560)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to initialize(...): (line 561)
        # Processing the call keyword arguments (line 561)
        kwargs_4372 = {}
        # Getting the type of 'self' (line 561)
        self_4370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 12), 'self', False)
        # Obtaining the member 'initialize' of a type (line 561)
        initialize_4371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 12), self_4370, 'initialize')
        # Calling initialize(args, kwargs) (line 561)
        initialize_call_result_4373 = invoke(stypy.reporting.localization.Localization(__file__, 561, 12), initialize_4371, *[], **kwargs_4372)
        
        # SSA join for if statement (line 560)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 562):
        
        # Assigning a Subscript to a Name (line 562):
        
        # Obtaining the type of the subscript
        int_4374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 8), 'int')
        
        # Call to _fix_object_args(...): (line 562)
        # Processing the call arguments (line 562)
        # Getting the type of 'objects' (line 562)
        objects_4377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 54), 'objects', False)
        # Getting the type of 'output_dir' (line 562)
        output_dir_4378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 63), 'output_dir', False)
        # Processing the call keyword arguments (line 562)
        kwargs_4379 = {}
        # Getting the type of 'self' (line 562)
        self_4375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 32), 'self', False)
        # Obtaining the member '_fix_object_args' of a type (line 562)
        _fix_object_args_4376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 32), self_4375, '_fix_object_args')
        # Calling _fix_object_args(args, kwargs) (line 562)
        _fix_object_args_call_result_4380 = invoke(stypy.reporting.localization.Localization(__file__, 562, 32), _fix_object_args_4376, *[objects_4377, output_dir_4378], **kwargs_4379)
        
        # Obtaining the member '__getitem__' of a type (line 562)
        getitem___4381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 8), _fix_object_args_call_result_4380, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 562)
        subscript_call_result_4382 = invoke(stypy.reporting.localization.Localization(__file__, 562, 8), getitem___4381, int_4374)
        
        # Assigning a type to the variable 'tuple_var_assignment_2885' (line 562)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'tuple_var_assignment_2885', subscript_call_result_4382)
        
        # Assigning a Subscript to a Name (line 562):
        
        # Obtaining the type of the subscript
        int_4383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 8), 'int')
        
        # Call to _fix_object_args(...): (line 562)
        # Processing the call arguments (line 562)
        # Getting the type of 'objects' (line 562)
        objects_4386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 54), 'objects', False)
        # Getting the type of 'output_dir' (line 562)
        output_dir_4387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 63), 'output_dir', False)
        # Processing the call keyword arguments (line 562)
        kwargs_4388 = {}
        # Getting the type of 'self' (line 562)
        self_4384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 32), 'self', False)
        # Obtaining the member '_fix_object_args' of a type (line 562)
        _fix_object_args_4385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 32), self_4384, '_fix_object_args')
        # Calling _fix_object_args(args, kwargs) (line 562)
        _fix_object_args_call_result_4389 = invoke(stypy.reporting.localization.Localization(__file__, 562, 32), _fix_object_args_4385, *[objects_4386, output_dir_4387], **kwargs_4388)
        
        # Obtaining the member '__getitem__' of a type (line 562)
        getitem___4390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 8), _fix_object_args_call_result_4389, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 562)
        subscript_call_result_4391 = invoke(stypy.reporting.localization.Localization(__file__, 562, 8), getitem___4390, int_4383)
        
        # Assigning a type to the variable 'tuple_var_assignment_2886' (line 562)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'tuple_var_assignment_2886', subscript_call_result_4391)
        
        # Assigning a Name to a Name (line 562):
        # Getting the type of 'tuple_var_assignment_2885' (line 562)
        tuple_var_assignment_2885_4392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'tuple_var_assignment_2885')
        # Assigning a type to the variable 'objects' (line 562)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 9), 'objects', tuple_var_assignment_2885_4392)
        
        # Assigning a Name to a Name (line 562):
        # Getting the type of 'tuple_var_assignment_2886' (line 562)
        tuple_var_assignment_2886_4393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'tuple_var_assignment_2886')
        # Assigning a type to the variable 'output_dir' (line 562)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 18), 'output_dir', tuple_var_assignment_2886_4393)
        
        # Assigning a Call to a Name (line 563):
        
        # Assigning a Call to a Name (line 563):
        
        # Call to library_filename(...): (line 563)
        # Processing the call arguments (line 563)
        # Getting the type of 'output_libname' (line 563)
        output_libname_4396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 48), 'output_libname', False)
        # Processing the call keyword arguments (line 563)
        # Getting the type of 'output_dir' (line 564)
        output_dir_4397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 59), 'output_dir', False)
        keyword_4398 = output_dir_4397
        kwargs_4399 = {'output_dir': keyword_4398}
        # Getting the type of 'self' (line 563)
        self_4394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 26), 'self', False)
        # Obtaining the member 'library_filename' of a type (line 563)
        library_filename_4395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 26), self_4394, 'library_filename')
        # Calling library_filename(args, kwargs) (line 563)
        library_filename_call_result_4400 = invoke(stypy.reporting.localization.Localization(__file__, 563, 26), library_filename_4395, *[output_libname_4396], **kwargs_4399)
        
        # Assigning a type to the variable 'output_filename' (line 563)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 8), 'output_filename', library_filename_call_result_4400)
        
        
        # Call to _need_link(...): (line 566)
        # Processing the call arguments (line 566)
        # Getting the type of 'objects' (line 566)
        objects_4403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 27), 'objects', False)
        # Getting the type of 'output_filename' (line 566)
        output_filename_4404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 36), 'output_filename', False)
        # Processing the call keyword arguments (line 566)
        kwargs_4405 = {}
        # Getting the type of 'self' (line 566)
        self_4401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 11), 'self', False)
        # Obtaining the member '_need_link' of a type (line 566)
        _need_link_4402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 11), self_4401, '_need_link')
        # Calling _need_link(args, kwargs) (line 566)
        _need_link_call_result_4406 = invoke(stypy.reporting.localization.Localization(__file__, 566, 11), _need_link_4402, *[objects_4403, output_filename_4404], **kwargs_4405)
        
        # Testing the type of an if condition (line 566)
        if_condition_4407 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 566, 8), _need_link_call_result_4406)
        # Assigning a type to the variable 'if_condition_4407' (line 566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'if_condition_4407', if_condition_4407)
        # SSA begins for if statement (line 566)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 567):
        
        # Assigning a BinOp to a Name (line 567):
        # Getting the type of 'objects' (line 567)
        objects_4408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 23), 'objects')
        
        # Obtaining an instance of the builtin type 'list' (line 567)
        list_4409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 567)
        # Adding element type (line 567)
        str_4410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 34), 'str', '/OUT:')
        # Getting the type of 'output_filename' (line 567)
        output_filename_4411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 44), 'output_filename')
        # Applying the binary operator '+' (line 567)
        result_add_4412 = python_operator(stypy.reporting.localization.Localization(__file__, 567, 34), '+', str_4410, output_filename_4411)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 567, 33), list_4409, result_add_4412)
        
        # Applying the binary operator '+' (line 567)
        result_add_4413 = python_operator(stypy.reporting.localization.Localization(__file__, 567, 23), '+', objects_4408, list_4409)
        
        # Assigning a type to the variable 'lib_args' (line 567)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 12), 'lib_args', result_add_4413)
        
        # Getting the type of 'debug' (line 568)
        debug_4414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 15), 'debug')
        # Testing the type of an if condition (line 568)
        if_condition_4415 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 568, 12), debug_4414)
        # Assigning a type to the variable 'if_condition_4415' (line 568)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 12), 'if_condition_4415', if_condition_4415)
        # SSA begins for if statement (line 568)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA join for if statement (line 568)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 570)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to spawn(...): (line 571)
        # Processing the call arguments (line 571)
        
        # Obtaining an instance of the builtin type 'list' (line 571)
        list_4418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 571)
        # Adding element type (line 571)
        # Getting the type of 'self' (line 571)
        self_4419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 28), 'self', False)
        # Obtaining the member 'lib' of a type (line 571)
        lib_4420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 28), self_4419, 'lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 571, 27), list_4418, lib_4420)
        
        # Getting the type of 'lib_args' (line 571)
        lib_args_4421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 40), 'lib_args', False)
        # Applying the binary operator '+' (line 571)
        result_add_4422 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 27), '+', list_4418, lib_args_4421)
        
        # Processing the call keyword arguments (line 571)
        kwargs_4423 = {}
        # Getting the type of 'self' (line 571)
        self_4416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 16), 'self', False)
        # Obtaining the member 'spawn' of a type (line 571)
        spawn_4417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 16), self_4416, 'spawn')
        # Calling spawn(args, kwargs) (line 571)
        spawn_call_result_4424 = invoke(stypy.reporting.localization.Localization(__file__, 571, 16), spawn_4417, *[result_add_4422], **kwargs_4423)
        
        # SSA branch for the except part of a try statement (line 570)
        # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 570)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsExecError' (line 572)
        DistutilsExecError_4425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 19), 'DistutilsExecError')
        # Assigning a type to the variable 'msg' (line 572)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 12), 'msg', DistutilsExecError_4425)
        
        # Call to LibError(...): (line 573)
        # Processing the call arguments (line 573)
        # Getting the type of 'msg' (line 573)
        msg_4427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 31), 'msg', False)
        # Processing the call keyword arguments (line 573)
        kwargs_4428 = {}
        # Getting the type of 'LibError' (line 573)
        LibError_4426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 22), 'LibError', False)
        # Calling LibError(args, kwargs) (line 573)
        LibError_call_result_4429 = invoke(stypy.reporting.localization.Localization(__file__, 573, 22), LibError_4426, *[msg_4427], **kwargs_4428)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 573, 16), LibError_call_result_4429, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 570)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 566)
        module_type_store.open_ssa_branch('else')
        
        # Call to debug(...): (line 575)
        # Processing the call arguments (line 575)
        str_4432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 22), 'str', 'skipping %s (up-to-date)')
        # Getting the type of 'output_filename' (line 575)
        output_filename_4433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 50), 'output_filename', False)
        # Processing the call keyword arguments (line 575)
        kwargs_4434 = {}
        # Getting the type of 'log' (line 575)
        log_4430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 12), 'log', False)
        # Obtaining the member 'debug' of a type (line 575)
        debug_4431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 12), log_4430, 'debug')
        # Calling debug(args, kwargs) (line 575)
        debug_call_result_4435 = invoke(stypy.reporting.localization.Localization(__file__, 575, 12), debug_4431, *[str_4432, output_filename_4433], **kwargs_4434)
        
        # SSA join for if statement (line 566)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'create_static_lib(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_static_lib' in the type store
        # Getting the type of 'stypy_return_type' (line 553)
        stypy_return_type_4436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4436)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_static_lib'
        return stypy_return_type_4436


    @norecursion
    def link(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 582)
        None_4437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 24), 'None')
        # Getting the type of 'None' (line 583)
        None_4438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 23), 'None')
        # Getting the type of 'None' (line 584)
        None_4439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 26), 'None')
        # Getting the type of 'None' (line 585)
        None_4440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 34), 'None')
        # Getting the type of 'None' (line 586)
        None_4441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 28), 'None')
        int_4442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 19), 'int')
        # Getting the type of 'None' (line 588)
        None_4443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 27), 'None')
        # Getting the type of 'None' (line 589)
        None_4444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 28), 'None')
        # Getting the type of 'None' (line 590)
        None_4445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 24), 'None')
        # Getting the type of 'None' (line 591)
        None_4446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 25), 'None')
        defaults = [None_4437, None_4438, None_4439, None_4440, None_4441, int_4442, None_4443, None_4444, None_4445, None_4446]
        # Create a new context for function 'link'
        module_type_store = module_type_store.open_function_context('link', 578, 4, False)
        # Assigning a type to the variable 'self' (line 579)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'self', type_of_self)
        
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

        
        
        # Getting the type of 'self' (line 593)
        self_4447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 15), 'self')
        # Obtaining the member 'initialized' of a type (line 593)
        initialized_4448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 15), self_4447, 'initialized')
        # Applying the 'not' unary operator (line 593)
        result_not__4449 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 11), 'not', initialized_4448)
        
        # Testing the type of an if condition (line 593)
        if_condition_4450 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 593, 8), result_not__4449)
        # Assigning a type to the variable 'if_condition_4450' (line 593)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'if_condition_4450', if_condition_4450)
        # SSA begins for if statement (line 593)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to initialize(...): (line 594)
        # Processing the call keyword arguments (line 594)
        kwargs_4453 = {}
        # Getting the type of 'self' (line 594)
        self_4451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 12), 'self', False)
        # Obtaining the member 'initialize' of a type (line 594)
        initialize_4452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 12), self_4451, 'initialize')
        # Calling initialize(args, kwargs) (line 594)
        initialize_call_result_4454 = invoke(stypy.reporting.localization.Localization(__file__, 594, 12), initialize_4452, *[], **kwargs_4453)
        
        # SSA join for if statement (line 593)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 595):
        
        # Assigning a Subscript to a Name (line 595):
        
        # Obtaining the type of the subscript
        int_4455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 8), 'int')
        
        # Call to _fix_object_args(...): (line 595)
        # Processing the call arguments (line 595)
        # Getting the type of 'objects' (line 595)
        objects_4458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 54), 'objects', False)
        # Getting the type of 'output_dir' (line 595)
        output_dir_4459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 63), 'output_dir', False)
        # Processing the call keyword arguments (line 595)
        kwargs_4460 = {}
        # Getting the type of 'self' (line 595)
        self_4456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 32), 'self', False)
        # Obtaining the member '_fix_object_args' of a type (line 595)
        _fix_object_args_4457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 32), self_4456, '_fix_object_args')
        # Calling _fix_object_args(args, kwargs) (line 595)
        _fix_object_args_call_result_4461 = invoke(stypy.reporting.localization.Localization(__file__, 595, 32), _fix_object_args_4457, *[objects_4458, output_dir_4459], **kwargs_4460)
        
        # Obtaining the member '__getitem__' of a type (line 595)
        getitem___4462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 8), _fix_object_args_call_result_4461, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 595)
        subscript_call_result_4463 = invoke(stypy.reporting.localization.Localization(__file__, 595, 8), getitem___4462, int_4455)
        
        # Assigning a type to the variable 'tuple_var_assignment_2887' (line 595)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 8), 'tuple_var_assignment_2887', subscript_call_result_4463)
        
        # Assigning a Subscript to a Name (line 595):
        
        # Obtaining the type of the subscript
        int_4464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 8), 'int')
        
        # Call to _fix_object_args(...): (line 595)
        # Processing the call arguments (line 595)
        # Getting the type of 'objects' (line 595)
        objects_4467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 54), 'objects', False)
        # Getting the type of 'output_dir' (line 595)
        output_dir_4468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 63), 'output_dir', False)
        # Processing the call keyword arguments (line 595)
        kwargs_4469 = {}
        # Getting the type of 'self' (line 595)
        self_4465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 32), 'self', False)
        # Obtaining the member '_fix_object_args' of a type (line 595)
        _fix_object_args_4466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 32), self_4465, '_fix_object_args')
        # Calling _fix_object_args(args, kwargs) (line 595)
        _fix_object_args_call_result_4470 = invoke(stypy.reporting.localization.Localization(__file__, 595, 32), _fix_object_args_4466, *[objects_4467, output_dir_4468], **kwargs_4469)
        
        # Obtaining the member '__getitem__' of a type (line 595)
        getitem___4471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 8), _fix_object_args_call_result_4470, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 595)
        subscript_call_result_4472 = invoke(stypy.reporting.localization.Localization(__file__, 595, 8), getitem___4471, int_4464)
        
        # Assigning a type to the variable 'tuple_var_assignment_2888' (line 595)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 8), 'tuple_var_assignment_2888', subscript_call_result_4472)
        
        # Assigning a Name to a Name (line 595):
        # Getting the type of 'tuple_var_assignment_2887' (line 595)
        tuple_var_assignment_2887_4473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 8), 'tuple_var_assignment_2887')
        # Assigning a type to the variable 'objects' (line 595)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 9), 'objects', tuple_var_assignment_2887_4473)
        
        # Assigning a Name to a Name (line 595):
        # Getting the type of 'tuple_var_assignment_2888' (line 595)
        tuple_var_assignment_2888_4474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 8), 'tuple_var_assignment_2888')
        # Assigning a type to the variable 'output_dir' (line 595)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 18), 'output_dir', tuple_var_assignment_2888_4474)
        
        # Assigning a Call to a Name (line 596):
        
        # Assigning a Call to a Name (line 596):
        
        # Call to _fix_lib_args(...): (line 596)
        # Processing the call arguments (line 596)
        # Getting the type of 'libraries' (line 596)
        libraries_4477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 40), 'libraries', False)
        # Getting the type of 'library_dirs' (line 596)
        library_dirs_4478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 51), 'library_dirs', False)
        # Getting the type of 'runtime_library_dirs' (line 597)
        runtime_library_dirs_4479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 40), 'runtime_library_dirs', False)
        # Processing the call keyword arguments (line 596)
        kwargs_4480 = {}
        # Getting the type of 'self' (line 596)
        self_4475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 21), 'self', False)
        # Obtaining the member '_fix_lib_args' of a type (line 596)
        _fix_lib_args_4476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 21), self_4475, '_fix_lib_args')
        # Calling _fix_lib_args(args, kwargs) (line 596)
        _fix_lib_args_call_result_4481 = invoke(stypy.reporting.localization.Localization(__file__, 596, 21), _fix_lib_args_4476, *[libraries_4477, library_dirs_4478, runtime_library_dirs_4479], **kwargs_4480)
        
        # Assigning a type to the variable 'fixed_args' (line 596)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 8), 'fixed_args', _fix_lib_args_call_result_4481)
        
        # Assigning a Name to a Tuple (line 598):
        
        # Assigning a Subscript to a Name (line 598):
        
        # Obtaining the type of the subscript
        int_4482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 8), 'int')
        # Getting the type of 'fixed_args' (line 598)
        fixed_args_4483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 58), 'fixed_args')
        # Obtaining the member '__getitem__' of a type (line 598)
        getitem___4484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 8), fixed_args_4483, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 598)
        subscript_call_result_4485 = invoke(stypy.reporting.localization.Localization(__file__, 598, 8), getitem___4484, int_4482)
        
        # Assigning a type to the variable 'tuple_var_assignment_2889' (line 598)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'tuple_var_assignment_2889', subscript_call_result_4485)
        
        # Assigning a Subscript to a Name (line 598):
        
        # Obtaining the type of the subscript
        int_4486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 8), 'int')
        # Getting the type of 'fixed_args' (line 598)
        fixed_args_4487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 58), 'fixed_args')
        # Obtaining the member '__getitem__' of a type (line 598)
        getitem___4488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 8), fixed_args_4487, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 598)
        subscript_call_result_4489 = invoke(stypy.reporting.localization.Localization(__file__, 598, 8), getitem___4488, int_4486)
        
        # Assigning a type to the variable 'tuple_var_assignment_2890' (line 598)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'tuple_var_assignment_2890', subscript_call_result_4489)
        
        # Assigning a Subscript to a Name (line 598):
        
        # Obtaining the type of the subscript
        int_4490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 8), 'int')
        # Getting the type of 'fixed_args' (line 598)
        fixed_args_4491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 58), 'fixed_args')
        # Obtaining the member '__getitem__' of a type (line 598)
        getitem___4492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 8), fixed_args_4491, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 598)
        subscript_call_result_4493 = invoke(stypy.reporting.localization.Localization(__file__, 598, 8), getitem___4492, int_4490)
        
        # Assigning a type to the variable 'tuple_var_assignment_2891' (line 598)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'tuple_var_assignment_2891', subscript_call_result_4493)
        
        # Assigning a Name to a Name (line 598):
        # Getting the type of 'tuple_var_assignment_2889' (line 598)
        tuple_var_assignment_2889_4494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'tuple_var_assignment_2889')
        # Assigning a type to the variable 'libraries' (line 598)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 9), 'libraries', tuple_var_assignment_2889_4494)
        
        # Assigning a Name to a Name (line 598):
        # Getting the type of 'tuple_var_assignment_2890' (line 598)
        tuple_var_assignment_2890_4495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'tuple_var_assignment_2890')
        # Assigning a type to the variable 'library_dirs' (line 598)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 20), 'library_dirs', tuple_var_assignment_2890_4495)
        
        # Assigning a Name to a Name (line 598):
        # Getting the type of 'tuple_var_assignment_2891' (line 598)
        tuple_var_assignment_2891_4496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'tuple_var_assignment_2891')
        # Assigning a type to the variable 'runtime_library_dirs' (line 598)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 34), 'runtime_library_dirs', tuple_var_assignment_2891_4496)
        
        # Getting the type of 'runtime_library_dirs' (line 600)
        runtime_library_dirs_4497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 11), 'runtime_library_dirs')
        # Testing the type of an if condition (line 600)
        if_condition_4498 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 600, 8), runtime_library_dirs_4497)
        # Assigning a type to the variable 'if_condition_4498' (line 600)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 8), 'if_condition_4498', if_condition_4498)
        # SSA begins for if statement (line 600)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 601)
        # Processing the call arguments (line 601)
        str_4501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 23), 'str', "I don't know what to do with 'runtime_library_dirs': ")
        
        # Call to str(...): (line 602)
        # Processing the call arguments (line 602)
        # Getting the type of 'runtime_library_dirs' (line 602)
        runtime_library_dirs_4503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 30), 'runtime_library_dirs', False)
        # Processing the call keyword arguments (line 602)
        kwargs_4504 = {}
        # Getting the type of 'str' (line 602)
        str_4502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 25), 'str', False)
        # Calling str(args, kwargs) (line 602)
        str_call_result_4505 = invoke(stypy.reporting.localization.Localization(__file__, 602, 25), str_4502, *[runtime_library_dirs_4503], **kwargs_4504)
        
        # Applying the binary operator '+' (line 601)
        result_add_4506 = python_operator(stypy.reporting.localization.Localization(__file__, 601, 23), '+', str_4501, str_call_result_4505)
        
        # Processing the call keyword arguments (line 601)
        kwargs_4507 = {}
        # Getting the type of 'self' (line 601)
        self_4499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 12), 'self', False)
        # Obtaining the member 'warn' of a type (line 601)
        warn_4500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 12), self_4499, 'warn')
        # Calling warn(args, kwargs) (line 601)
        warn_call_result_4508 = invoke(stypy.reporting.localization.Localization(__file__, 601, 12), warn_4500, *[result_add_4506], **kwargs_4507)
        
        # SSA join for if statement (line 600)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 604):
        
        # Assigning a Call to a Name (line 604):
        
        # Call to gen_lib_options(...): (line 604)
        # Processing the call arguments (line 604)
        # Getting the type of 'self' (line 604)
        self_4510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 35), 'self', False)
        # Getting the type of 'library_dirs' (line 605)
        library_dirs_4511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 35), 'library_dirs', False)
        # Getting the type of 'runtime_library_dirs' (line 605)
        runtime_library_dirs_4512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 49), 'runtime_library_dirs', False)
        # Getting the type of 'libraries' (line 606)
        libraries_4513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 35), 'libraries', False)
        # Processing the call keyword arguments (line 604)
        kwargs_4514 = {}
        # Getting the type of 'gen_lib_options' (line 604)
        gen_lib_options_4509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 19), 'gen_lib_options', False)
        # Calling gen_lib_options(args, kwargs) (line 604)
        gen_lib_options_call_result_4515 = invoke(stypy.reporting.localization.Localization(__file__, 604, 19), gen_lib_options_4509, *[self_4510, library_dirs_4511, runtime_library_dirs_4512, libraries_4513], **kwargs_4514)
        
        # Assigning a type to the variable 'lib_opts' (line 604)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 8), 'lib_opts', gen_lib_options_call_result_4515)
        
        # Type idiom detected: calculating its left and rigth part (line 607)
        # Getting the type of 'output_dir' (line 607)
        output_dir_4516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 8), 'output_dir')
        # Getting the type of 'None' (line 607)
        None_4517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 29), 'None')
        
        (may_be_4518, more_types_in_union_4519) = may_not_be_none(output_dir_4516, None_4517)

        if may_be_4518:

            if more_types_in_union_4519:
                # Runtime conditional SSA (line 607)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 608):
            
            # Assigning a Call to a Name (line 608):
            
            # Call to join(...): (line 608)
            # Processing the call arguments (line 608)
            # Getting the type of 'output_dir' (line 608)
            output_dir_4523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 43), 'output_dir', False)
            # Getting the type of 'output_filename' (line 608)
            output_filename_4524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 55), 'output_filename', False)
            # Processing the call keyword arguments (line 608)
            kwargs_4525 = {}
            # Getting the type of 'os' (line 608)
            os_4520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 30), 'os', False)
            # Obtaining the member 'path' of a type (line 608)
            path_4521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 30), os_4520, 'path')
            # Obtaining the member 'join' of a type (line 608)
            join_4522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 30), path_4521, 'join')
            # Calling join(args, kwargs) (line 608)
            join_call_result_4526 = invoke(stypy.reporting.localization.Localization(__file__, 608, 30), join_4522, *[output_dir_4523, output_filename_4524], **kwargs_4525)
            
            # Assigning a type to the variable 'output_filename' (line 608)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 12), 'output_filename', join_call_result_4526)

            if more_types_in_union_4519:
                # SSA join for if statement (line 607)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Call to _need_link(...): (line 610)
        # Processing the call arguments (line 610)
        # Getting the type of 'objects' (line 610)
        objects_4529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 27), 'objects', False)
        # Getting the type of 'output_filename' (line 610)
        output_filename_4530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 36), 'output_filename', False)
        # Processing the call keyword arguments (line 610)
        kwargs_4531 = {}
        # Getting the type of 'self' (line 610)
        self_4527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 11), 'self', False)
        # Obtaining the member '_need_link' of a type (line 610)
        _need_link_4528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 11), self_4527, '_need_link')
        # Calling _need_link(args, kwargs) (line 610)
        _need_link_call_result_4532 = invoke(stypy.reporting.localization.Localization(__file__, 610, 11), _need_link_4528, *[objects_4529, output_filename_4530], **kwargs_4531)
        
        # Testing the type of an if condition (line 610)
        if_condition_4533 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 610, 8), _need_link_call_result_4532)
        # Assigning a type to the variable 'if_condition_4533' (line 610)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 8), 'if_condition_4533', if_condition_4533)
        # SSA begins for if statement (line 610)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'target_desc' (line 611)
        target_desc_4534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 15), 'target_desc')
        # Getting the type of 'CCompiler' (line 611)
        CCompiler_4535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 30), 'CCompiler')
        # Obtaining the member 'EXECUTABLE' of a type (line 611)
        EXECUTABLE_4536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 30), CCompiler_4535, 'EXECUTABLE')
        # Applying the binary operator '==' (line 611)
        result_eq_4537 = python_operator(stypy.reporting.localization.Localization(__file__, 611, 15), '==', target_desc_4534, EXECUTABLE_4536)
        
        # Testing the type of an if condition (line 611)
        if_condition_4538 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 611, 12), result_eq_4537)
        # Assigning a type to the variable 'if_condition_4538' (line 611)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 12), 'if_condition_4538', if_condition_4538)
        # SSA begins for if statement (line 611)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'debug' (line 612)
        debug_4539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 19), 'debug')
        # Testing the type of an if condition (line 612)
        if_condition_4540 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 612, 16), debug_4539)
        # Assigning a type to the variable 'if_condition_4540' (line 612)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 16), 'if_condition_4540', if_condition_4540)
        # SSA begins for if statement (line 612)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 613):
        
        # Assigning a Subscript to a Name (line 613):
        
        # Obtaining the type of the subscript
        int_4541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 56), 'int')
        slice_4542 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 613, 30), int_4541, None, None)
        # Getting the type of 'self' (line 613)
        self_4543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 30), 'self')
        # Obtaining the member 'ldflags_shared_debug' of a type (line 613)
        ldflags_shared_debug_4544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 30), self_4543, 'ldflags_shared_debug')
        # Obtaining the member '__getitem__' of a type (line 613)
        getitem___4545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 30), ldflags_shared_debug_4544, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 613)
        subscript_call_result_4546 = invoke(stypy.reporting.localization.Localization(__file__, 613, 30), getitem___4545, slice_4542)
        
        # Assigning a type to the variable 'ldflags' (line 613)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 20), 'ldflags', subscript_call_result_4546)
        # SSA branch for the else part of an if statement (line 612)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Name (line 615):
        
        # Assigning a Subscript to a Name (line 615):
        
        # Obtaining the type of the subscript
        int_4547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 50), 'int')
        slice_4548 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 615, 30), int_4547, None, None)
        # Getting the type of 'self' (line 615)
        self_4549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 30), 'self')
        # Obtaining the member 'ldflags_shared' of a type (line 615)
        ldflags_shared_4550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 30), self_4549, 'ldflags_shared')
        # Obtaining the member '__getitem__' of a type (line 615)
        getitem___4551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 30), ldflags_shared_4550, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 615)
        subscript_call_result_4552 = invoke(stypy.reporting.localization.Localization(__file__, 615, 30), getitem___4551, slice_4548)
        
        # Assigning a type to the variable 'ldflags' (line 615)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 20), 'ldflags', subscript_call_result_4552)
        # SSA join for if statement (line 612)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 611)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'debug' (line 617)
        debug_4553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 19), 'debug')
        # Testing the type of an if condition (line 617)
        if_condition_4554 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 617, 16), debug_4553)
        # Assigning a type to the variable 'if_condition_4554' (line 617)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 16), 'if_condition_4554', if_condition_4554)
        # SSA begins for if statement (line 617)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 618):
        
        # Assigning a Attribute to a Name (line 618):
        # Getting the type of 'self' (line 618)
        self_4555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 30), 'self')
        # Obtaining the member 'ldflags_shared_debug' of a type (line 618)
        ldflags_shared_debug_4556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 30), self_4555, 'ldflags_shared_debug')
        # Assigning a type to the variable 'ldflags' (line 618)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 20), 'ldflags', ldflags_shared_debug_4556)
        # SSA branch for the else part of an if statement (line 617)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 620):
        
        # Assigning a Attribute to a Name (line 620):
        # Getting the type of 'self' (line 620)
        self_4557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 30), 'self')
        # Obtaining the member 'ldflags_shared' of a type (line 620)
        ldflags_shared_4558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 30), self_4557, 'ldflags_shared')
        # Assigning a type to the variable 'ldflags' (line 620)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 20), 'ldflags', ldflags_shared_4558)
        # SSA join for if statement (line 617)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 611)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 622):
        
        # Assigning a List to a Name (line 622):
        
        # Obtaining an instance of the builtin type 'list' (line 622)
        list_4559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 622)
        
        # Assigning a type to the variable 'export_opts' (line 622)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 12), 'export_opts', list_4559)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'export_symbols' (line 623)
        export_symbols_4560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 24), 'export_symbols')
        
        # Obtaining an instance of the builtin type 'list' (line 623)
        list_4561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 623)
        
        # Applying the binary operator 'or' (line 623)
        result_or_keyword_4562 = python_operator(stypy.reporting.localization.Localization(__file__, 623, 24), 'or', export_symbols_4560, list_4561)
        
        # Testing the type of a for loop iterable (line 623)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 623, 12), result_or_keyword_4562)
        # Getting the type of the for loop variable (line 623)
        for_loop_var_4563 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 623, 12), result_or_keyword_4562)
        # Assigning a type to the variable 'sym' (line 623)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 12), 'sym', for_loop_var_4563)
        # SSA begins for a for statement (line 623)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 624)
        # Processing the call arguments (line 624)
        str_4566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 35), 'str', '/EXPORT:')
        # Getting the type of 'sym' (line 624)
        sym_4567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 48), 'sym', False)
        # Applying the binary operator '+' (line 624)
        result_add_4568 = python_operator(stypy.reporting.localization.Localization(__file__, 624, 35), '+', str_4566, sym_4567)
        
        # Processing the call keyword arguments (line 624)
        kwargs_4569 = {}
        # Getting the type of 'export_opts' (line 624)
        export_opts_4564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 16), 'export_opts', False)
        # Obtaining the member 'append' of a type (line 624)
        append_4565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 16), export_opts_4564, 'append')
        # Calling append(args, kwargs) (line 624)
        append_call_result_4570 = invoke(stypy.reporting.localization.Localization(__file__, 624, 16), append_4565, *[result_add_4568], **kwargs_4569)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 626):
        
        # Assigning a BinOp to a Name (line 626):
        # Getting the type of 'ldflags' (line 626)
        ldflags_4571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 23), 'ldflags')
        # Getting the type of 'lib_opts' (line 626)
        lib_opts_4572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 33), 'lib_opts')
        # Applying the binary operator '+' (line 626)
        result_add_4573 = python_operator(stypy.reporting.localization.Localization(__file__, 626, 23), '+', ldflags_4571, lib_opts_4572)
        
        # Getting the type of 'export_opts' (line 626)
        export_opts_4574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 44), 'export_opts')
        # Applying the binary operator '+' (line 626)
        result_add_4575 = python_operator(stypy.reporting.localization.Localization(__file__, 626, 42), '+', result_add_4573, export_opts_4574)
        
        # Getting the type of 'objects' (line 627)
        objects_4576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 23), 'objects')
        # Applying the binary operator '+' (line 626)
        result_add_4577 = python_operator(stypy.reporting.localization.Localization(__file__, 626, 56), '+', result_add_4575, objects_4576)
        
        
        # Obtaining an instance of the builtin type 'list' (line 627)
        list_4578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 627)
        # Adding element type (line 627)
        str_4579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 34), 'str', '/OUT:')
        # Getting the type of 'output_filename' (line 627)
        output_filename_4580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 44), 'output_filename')
        # Applying the binary operator '+' (line 627)
        result_add_4581 = python_operator(stypy.reporting.localization.Localization(__file__, 627, 34), '+', str_4579, output_filename_4580)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 627, 33), list_4578, result_add_4581)
        
        # Applying the binary operator '+' (line 627)
        result_add_4582 = python_operator(stypy.reporting.localization.Localization(__file__, 627, 31), '+', result_add_4577, list_4578)
        
        # Assigning a type to the variable 'ld_args' (line 626)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 12), 'ld_args', result_add_4582)
        
        # Assigning a Call to a Name (line 634):
        
        # Assigning a Call to a Name (line 634):
        
        # Call to dirname(...): (line 634)
        # Processing the call arguments (line 634)
        
        # Obtaining the type of the subscript
        int_4586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 49), 'int')
        # Getting the type of 'objects' (line 634)
        objects_4587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 41), 'objects', False)
        # Obtaining the member '__getitem__' of a type (line 634)
        getitem___4588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 41), objects_4587, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 634)
        subscript_call_result_4589 = invoke(stypy.reporting.localization.Localization(__file__, 634, 41), getitem___4588, int_4586)
        
        # Processing the call keyword arguments (line 634)
        kwargs_4590 = {}
        # Getting the type of 'os' (line 634)
        os_4583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 634)
        path_4584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 25), os_4583, 'path')
        # Obtaining the member 'dirname' of a type (line 634)
        dirname_4585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 25), path_4584, 'dirname')
        # Calling dirname(args, kwargs) (line 634)
        dirname_call_result_4591 = invoke(stypy.reporting.localization.Localization(__file__, 634, 25), dirname_4585, *[subscript_call_result_4589], **kwargs_4590)
        
        # Assigning a type to the variable 'build_temp' (line 634)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 12), 'build_temp', dirname_call_result_4591)
        
        # Type idiom detected: calculating its left and rigth part (line 635)
        # Getting the type of 'export_symbols' (line 635)
        export_symbols_4592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 12), 'export_symbols')
        # Getting the type of 'None' (line 635)
        None_4593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 37), 'None')
        
        (may_be_4594, more_types_in_union_4595) = may_not_be_none(export_symbols_4592, None_4593)

        if may_be_4594:

            if more_types_in_union_4595:
                # Runtime conditional SSA (line 635)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Tuple (line 636):
            
            # Assigning a Subscript to a Name (line 636):
            
            # Obtaining the type of the subscript
            int_4596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 16), 'int')
            
            # Call to splitext(...): (line 636)
            # Processing the call arguments (line 636)
            
            # Call to basename(...): (line 637)
            # Processing the call arguments (line 637)
            # Getting the type of 'output_filename' (line 637)
            output_filename_4603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 37), 'output_filename', False)
            # Processing the call keyword arguments (line 637)
            kwargs_4604 = {}
            # Getting the type of 'os' (line 637)
            os_4600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 20), 'os', False)
            # Obtaining the member 'path' of a type (line 637)
            path_4601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 20), os_4600, 'path')
            # Obtaining the member 'basename' of a type (line 637)
            basename_4602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 20), path_4601, 'basename')
            # Calling basename(args, kwargs) (line 637)
            basename_call_result_4605 = invoke(stypy.reporting.localization.Localization(__file__, 637, 20), basename_4602, *[output_filename_4603], **kwargs_4604)
            
            # Processing the call keyword arguments (line 636)
            kwargs_4606 = {}
            # Getting the type of 'os' (line 636)
            os_4597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 38), 'os', False)
            # Obtaining the member 'path' of a type (line 636)
            path_4598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 38), os_4597, 'path')
            # Obtaining the member 'splitext' of a type (line 636)
            splitext_4599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 38), path_4598, 'splitext')
            # Calling splitext(args, kwargs) (line 636)
            splitext_call_result_4607 = invoke(stypy.reporting.localization.Localization(__file__, 636, 38), splitext_4599, *[basename_call_result_4605], **kwargs_4606)
            
            # Obtaining the member '__getitem__' of a type (line 636)
            getitem___4608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 16), splitext_call_result_4607, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 636)
            subscript_call_result_4609 = invoke(stypy.reporting.localization.Localization(__file__, 636, 16), getitem___4608, int_4596)
            
            # Assigning a type to the variable 'tuple_var_assignment_2892' (line 636)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 16), 'tuple_var_assignment_2892', subscript_call_result_4609)
            
            # Assigning a Subscript to a Name (line 636):
            
            # Obtaining the type of the subscript
            int_4610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 16), 'int')
            
            # Call to splitext(...): (line 636)
            # Processing the call arguments (line 636)
            
            # Call to basename(...): (line 637)
            # Processing the call arguments (line 637)
            # Getting the type of 'output_filename' (line 637)
            output_filename_4617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 37), 'output_filename', False)
            # Processing the call keyword arguments (line 637)
            kwargs_4618 = {}
            # Getting the type of 'os' (line 637)
            os_4614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 20), 'os', False)
            # Obtaining the member 'path' of a type (line 637)
            path_4615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 20), os_4614, 'path')
            # Obtaining the member 'basename' of a type (line 637)
            basename_4616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 20), path_4615, 'basename')
            # Calling basename(args, kwargs) (line 637)
            basename_call_result_4619 = invoke(stypy.reporting.localization.Localization(__file__, 637, 20), basename_4616, *[output_filename_4617], **kwargs_4618)
            
            # Processing the call keyword arguments (line 636)
            kwargs_4620 = {}
            # Getting the type of 'os' (line 636)
            os_4611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 38), 'os', False)
            # Obtaining the member 'path' of a type (line 636)
            path_4612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 38), os_4611, 'path')
            # Obtaining the member 'splitext' of a type (line 636)
            splitext_4613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 38), path_4612, 'splitext')
            # Calling splitext(args, kwargs) (line 636)
            splitext_call_result_4621 = invoke(stypy.reporting.localization.Localization(__file__, 636, 38), splitext_4613, *[basename_call_result_4619], **kwargs_4620)
            
            # Obtaining the member '__getitem__' of a type (line 636)
            getitem___4622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 16), splitext_call_result_4621, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 636)
            subscript_call_result_4623 = invoke(stypy.reporting.localization.Localization(__file__, 636, 16), getitem___4622, int_4610)
            
            # Assigning a type to the variable 'tuple_var_assignment_2893' (line 636)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 16), 'tuple_var_assignment_2893', subscript_call_result_4623)
            
            # Assigning a Name to a Name (line 636):
            # Getting the type of 'tuple_var_assignment_2892' (line 636)
            tuple_var_assignment_2892_4624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 16), 'tuple_var_assignment_2892')
            # Assigning a type to the variable 'dll_name' (line 636)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 17), 'dll_name', tuple_var_assignment_2892_4624)
            
            # Assigning a Name to a Name (line 636):
            # Getting the type of 'tuple_var_assignment_2893' (line 636)
            tuple_var_assignment_2893_4625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 16), 'tuple_var_assignment_2893')
            # Assigning a type to the variable 'dll_ext' (line 636)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 27), 'dll_ext', tuple_var_assignment_2893_4625)
            
            # Assigning a Call to a Name (line 638):
            
            # Assigning a Call to a Name (line 638):
            
            # Call to join(...): (line 638)
            # Processing the call arguments (line 638)
            # Getting the type of 'build_temp' (line 639)
            build_temp_4629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 20), 'build_temp', False)
            
            # Call to library_filename(...): (line 640)
            # Processing the call arguments (line 640)
            # Getting the type of 'dll_name' (line 640)
            dll_name_4632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 42), 'dll_name', False)
            # Processing the call keyword arguments (line 640)
            kwargs_4633 = {}
            # Getting the type of 'self' (line 640)
            self_4630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 20), 'self', False)
            # Obtaining the member 'library_filename' of a type (line 640)
            library_filename_4631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 20), self_4630, 'library_filename')
            # Calling library_filename(args, kwargs) (line 640)
            library_filename_call_result_4634 = invoke(stypy.reporting.localization.Localization(__file__, 640, 20), library_filename_4631, *[dll_name_4632], **kwargs_4633)
            
            # Processing the call keyword arguments (line 638)
            kwargs_4635 = {}
            # Getting the type of 'os' (line 638)
            os_4626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 30), 'os', False)
            # Obtaining the member 'path' of a type (line 638)
            path_4627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 30), os_4626, 'path')
            # Obtaining the member 'join' of a type (line 638)
            join_4628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 30), path_4627, 'join')
            # Calling join(args, kwargs) (line 638)
            join_call_result_4636 = invoke(stypy.reporting.localization.Localization(__file__, 638, 30), join_4628, *[build_temp_4629, library_filename_call_result_4634], **kwargs_4635)
            
            # Assigning a type to the variable 'implib_file' (line 638)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 16), 'implib_file', join_call_result_4636)
            
            # Call to append(...): (line 641)
            # Processing the call arguments (line 641)
            str_4639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 32), 'str', '/IMPLIB:')
            # Getting the type of 'implib_file' (line 641)
            implib_file_4640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 45), 'implib_file', False)
            # Applying the binary operator '+' (line 641)
            result_add_4641 = python_operator(stypy.reporting.localization.Localization(__file__, 641, 32), '+', str_4639, implib_file_4640)
            
            # Processing the call keyword arguments (line 641)
            kwargs_4642 = {}
            # Getting the type of 'ld_args' (line 641)
            ld_args_4637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 16), 'ld_args', False)
            # Obtaining the member 'append' of a type (line 641)
            append_4638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 16), ld_args_4637, 'append')
            # Calling append(args, kwargs) (line 641)
            append_call_result_4643 = invoke(stypy.reporting.localization.Localization(__file__, 641, 16), append_4638, *[result_add_4641], **kwargs_4642)
            

            if more_types_in_union_4595:
                # SSA join for if statement (line 635)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to manifest_setup_ldargs(...): (line 643)
        # Processing the call arguments (line 643)
        # Getting the type of 'output_filename' (line 643)
        output_filename_4646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 39), 'output_filename', False)
        # Getting the type of 'build_temp' (line 643)
        build_temp_4647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 56), 'build_temp', False)
        # Getting the type of 'ld_args' (line 643)
        ld_args_4648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 68), 'ld_args', False)
        # Processing the call keyword arguments (line 643)
        kwargs_4649 = {}
        # Getting the type of 'self' (line 643)
        self_4644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 12), 'self', False)
        # Obtaining the member 'manifest_setup_ldargs' of a type (line 643)
        manifest_setup_ldargs_4645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 12), self_4644, 'manifest_setup_ldargs')
        # Calling manifest_setup_ldargs(args, kwargs) (line 643)
        manifest_setup_ldargs_call_result_4650 = invoke(stypy.reporting.localization.Localization(__file__, 643, 12), manifest_setup_ldargs_4645, *[output_filename_4646, build_temp_4647, ld_args_4648], **kwargs_4649)
        
        
        # Getting the type of 'extra_preargs' (line 645)
        extra_preargs_4651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 15), 'extra_preargs')
        # Testing the type of an if condition (line 645)
        if_condition_4652 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 645, 12), extra_preargs_4651)
        # Assigning a type to the variable 'if_condition_4652' (line 645)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 12), 'if_condition_4652', if_condition_4652)
        # SSA begins for if statement (line 645)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 646):
        
        # Assigning a Name to a Subscript (line 646):
        # Getting the type of 'extra_preargs' (line 646)
        extra_preargs_4653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 30), 'extra_preargs')
        # Getting the type of 'ld_args' (line 646)
        ld_args_4654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 16), 'ld_args')
        int_4655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 25), 'int')
        slice_4656 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 646, 16), None, int_4655, None)
        # Storing an element on a container (line 646)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 646, 16), ld_args_4654, (slice_4656, extra_preargs_4653))
        # SSA join for if statement (line 645)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'extra_postargs' (line 647)
        extra_postargs_4657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 15), 'extra_postargs')
        # Testing the type of an if condition (line 647)
        if_condition_4658 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 647, 12), extra_postargs_4657)
        # Assigning a type to the variable 'if_condition_4658' (line 647)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 12), 'if_condition_4658', if_condition_4658)
        # SSA begins for if statement (line 647)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 648)
        # Processing the call arguments (line 648)
        # Getting the type of 'extra_postargs' (line 648)
        extra_postargs_4661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 31), 'extra_postargs', False)
        # Processing the call keyword arguments (line 648)
        kwargs_4662 = {}
        # Getting the type of 'ld_args' (line 648)
        ld_args_4659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 16), 'ld_args', False)
        # Obtaining the member 'extend' of a type (line 648)
        extend_4660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 16), ld_args_4659, 'extend')
        # Calling extend(args, kwargs) (line 648)
        extend_call_result_4663 = invoke(stypy.reporting.localization.Localization(__file__, 648, 16), extend_4660, *[extra_postargs_4661], **kwargs_4662)
        
        # SSA join for if statement (line 647)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to mkpath(...): (line 650)
        # Processing the call arguments (line 650)
        
        # Call to dirname(...): (line 650)
        # Processing the call arguments (line 650)
        # Getting the type of 'output_filename' (line 650)
        output_filename_4669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 40), 'output_filename', False)
        # Processing the call keyword arguments (line 650)
        kwargs_4670 = {}
        # Getting the type of 'os' (line 650)
        os_4666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 650)
        path_4667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 24), os_4666, 'path')
        # Obtaining the member 'dirname' of a type (line 650)
        dirname_4668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 24), path_4667, 'dirname')
        # Calling dirname(args, kwargs) (line 650)
        dirname_call_result_4671 = invoke(stypy.reporting.localization.Localization(__file__, 650, 24), dirname_4668, *[output_filename_4669], **kwargs_4670)
        
        # Processing the call keyword arguments (line 650)
        kwargs_4672 = {}
        # Getting the type of 'self' (line 650)
        self_4664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 12), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 650)
        mkpath_4665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 12), self_4664, 'mkpath')
        # Calling mkpath(args, kwargs) (line 650)
        mkpath_call_result_4673 = invoke(stypy.reporting.localization.Localization(__file__, 650, 12), mkpath_4665, *[dirname_call_result_4671], **kwargs_4672)
        
        
        
        # SSA begins for try-except statement (line 651)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to spawn(...): (line 652)
        # Processing the call arguments (line 652)
        
        # Obtaining an instance of the builtin type 'list' (line 652)
        list_4676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 652)
        # Adding element type (line 652)
        # Getting the type of 'self' (line 652)
        self_4677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 28), 'self', False)
        # Obtaining the member 'linker' of a type (line 652)
        linker_4678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 28), self_4677, 'linker')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 652, 27), list_4676, linker_4678)
        
        # Getting the type of 'ld_args' (line 652)
        ld_args_4679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 43), 'ld_args', False)
        # Applying the binary operator '+' (line 652)
        result_add_4680 = python_operator(stypy.reporting.localization.Localization(__file__, 652, 27), '+', list_4676, ld_args_4679)
        
        # Processing the call keyword arguments (line 652)
        kwargs_4681 = {}
        # Getting the type of 'self' (line 652)
        self_4674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 16), 'self', False)
        # Obtaining the member 'spawn' of a type (line 652)
        spawn_4675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 16), self_4674, 'spawn')
        # Calling spawn(args, kwargs) (line 652)
        spawn_call_result_4682 = invoke(stypy.reporting.localization.Localization(__file__, 652, 16), spawn_4675, *[result_add_4680], **kwargs_4681)
        
        # SSA branch for the except part of a try statement (line 651)
        # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 651)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsExecError' (line 653)
        DistutilsExecError_4683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 19), 'DistutilsExecError')
        # Assigning a type to the variable 'msg' (line 653)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 12), 'msg', DistutilsExecError_4683)
        
        # Call to LinkError(...): (line 654)
        # Processing the call arguments (line 654)
        # Getting the type of 'msg' (line 654)
        msg_4685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 32), 'msg', False)
        # Processing the call keyword arguments (line 654)
        kwargs_4686 = {}
        # Getting the type of 'LinkError' (line 654)
        LinkError_4684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 22), 'LinkError', False)
        # Calling LinkError(args, kwargs) (line 654)
        LinkError_call_result_4687 = invoke(stypy.reporting.localization.Localization(__file__, 654, 22), LinkError_4684, *[msg_4685], **kwargs_4686)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 654, 16), LinkError_call_result_4687, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 651)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 661):
        
        # Assigning a Call to a Name (line 661):
        
        # Call to manifest_get_embed_info(...): (line 661)
        # Processing the call arguments (line 661)
        # Getting the type of 'target_desc' (line 661)
        target_desc_4690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 50), 'target_desc', False)
        # Getting the type of 'ld_args' (line 661)
        ld_args_4691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 63), 'ld_args', False)
        # Processing the call keyword arguments (line 661)
        kwargs_4692 = {}
        # Getting the type of 'self' (line 661)
        self_4688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 21), 'self', False)
        # Obtaining the member 'manifest_get_embed_info' of a type (line 661)
        manifest_get_embed_info_4689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 21), self_4688, 'manifest_get_embed_info')
        # Calling manifest_get_embed_info(args, kwargs) (line 661)
        manifest_get_embed_info_call_result_4693 = invoke(stypy.reporting.localization.Localization(__file__, 661, 21), manifest_get_embed_info_4689, *[target_desc_4690, ld_args_4691], **kwargs_4692)
        
        # Assigning a type to the variable 'mfinfo' (line 661)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 12), 'mfinfo', manifest_get_embed_info_call_result_4693)
        
        # Type idiom detected: calculating its left and rigth part (line 662)
        # Getting the type of 'mfinfo' (line 662)
        mfinfo_4694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 12), 'mfinfo')
        # Getting the type of 'None' (line 662)
        None_4695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 29), 'None')
        
        (may_be_4696, more_types_in_union_4697) = may_not_be_none(mfinfo_4694, None_4695)

        if may_be_4696:

            if more_types_in_union_4697:
                # Runtime conditional SSA (line 662)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Tuple (line 663):
            
            # Assigning a Subscript to a Name (line 663):
            
            # Obtaining the type of the subscript
            int_4698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 16), 'int')
            # Getting the type of 'mfinfo' (line 663)
            mfinfo_4699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 35), 'mfinfo')
            # Obtaining the member '__getitem__' of a type (line 663)
            getitem___4700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 16), mfinfo_4699, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 663)
            subscript_call_result_4701 = invoke(stypy.reporting.localization.Localization(__file__, 663, 16), getitem___4700, int_4698)
            
            # Assigning a type to the variable 'tuple_var_assignment_2894' (line 663)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 16), 'tuple_var_assignment_2894', subscript_call_result_4701)
            
            # Assigning a Subscript to a Name (line 663):
            
            # Obtaining the type of the subscript
            int_4702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 16), 'int')
            # Getting the type of 'mfinfo' (line 663)
            mfinfo_4703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 35), 'mfinfo')
            # Obtaining the member '__getitem__' of a type (line 663)
            getitem___4704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 16), mfinfo_4703, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 663)
            subscript_call_result_4705 = invoke(stypy.reporting.localization.Localization(__file__, 663, 16), getitem___4704, int_4702)
            
            # Assigning a type to the variable 'tuple_var_assignment_2895' (line 663)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 16), 'tuple_var_assignment_2895', subscript_call_result_4705)
            
            # Assigning a Name to a Name (line 663):
            # Getting the type of 'tuple_var_assignment_2894' (line 663)
            tuple_var_assignment_2894_4706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 16), 'tuple_var_assignment_2894')
            # Assigning a type to the variable 'mffilename' (line 663)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 16), 'mffilename', tuple_var_assignment_2894_4706)
            
            # Assigning a Name to a Name (line 663):
            # Getting the type of 'tuple_var_assignment_2895' (line 663)
            tuple_var_assignment_2895_4707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 16), 'tuple_var_assignment_2895')
            # Assigning a type to the variable 'mfid' (line 663)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 28), 'mfid', tuple_var_assignment_2895_4707)
            
            # Assigning a BinOp to a Name (line 664):
            
            # Assigning a BinOp to a Name (line 664):
            str_4708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 26), 'str', '-outputresource:%s;%s')
            
            # Obtaining an instance of the builtin type 'tuple' (line 664)
            tuple_4709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 53), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 664)
            # Adding element type (line 664)
            # Getting the type of 'output_filename' (line 664)
            output_filename_4710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 53), 'output_filename')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 664, 53), tuple_4709, output_filename_4710)
            # Adding element type (line 664)
            # Getting the type of 'mfid' (line 664)
            mfid_4711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 70), 'mfid')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 664, 53), tuple_4709, mfid_4711)
            
            # Applying the binary operator '%' (line 664)
            result_mod_4712 = python_operator(stypy.reporting.localization.Localization(__file__, 664, 26), '%', str_4708, tuple_4709)
            
            # Assigning a type to the variable 'out_arg' (line 664)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 16), 'out_arg', result_mod_4712)
            
            
            # SSA begins for try-except statement (line 665)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Call to spawn(...): (line 666)
            # Processing the call arguments (line 666)
            
            # Obtaining an instance of the builtin type 'list' (line 666)
            list_4715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 31), 'list')
            # Adding type elements to the builtin type 'list' instance (line 666)
            # Adding element type (line 666)
            str_4716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 32), 'str', 'mt.exe')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 666, 31), list_4715, str_4716)
            # Adding element type (line 666)
            str_4717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 42), 'str', '-nologo')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 666, 31), list_4715, str_4717)
            # Adding element type (line 666)
            str_4718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 53), 'str', '-manifest')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 666, 31), list_4715, str_4718)
            # Adding element type (line 666)
            # Getting the type of 'mffilename' (line 667)
            mffilename_4719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 32), 'mffilename', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 666, 31), list_4715, mffilename_4719)
            # Adding element type (line 666)
            # Getting the type of 'out_arg' (line 667)
            out_arg_4720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 44), 'out_arg', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 666, 31), list_4715, out_arg_4720)
            
            # Processing the call keyword arguments (line 666)
            kwargs_4721 = {}
            # Getting the type of 'self' (line 666)
            self_4713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 20), 'self', False)
            # Obtaining the member 'spawn' of a type (line 666)
            spawn_4714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 20), self_4713, 'spawn')
            # Calling spawn(args, kwargs) (line 666)
            spawn_call_result_4722 = invoke(stypy.reporting.localization.Localization(__file__, 666, 20), spawn_4714, *[list_4715], **kwargs_4721)
            
            # SSA branch for the except part of a try statement (line 665)
            # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 665)
            # Storing handler type
            module_type_store.open_ssa_branch('except')
            # Getting the type of 'DistutilsExecError' (line 668)
            DistutilsExecError_4723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 23), 'DistutilsExecError')
            # Assigning a type to the variable 'msg' (line 668)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 16), 'msg', DistutilsExecError_4723)
            
            # Call to LinkError(...): (line 669)
            # Processing the call arguments (line 669)
            # Getting the type of 'msg' (line 669)
            msg_4725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 36), 'msg', False)
            # Processing the call keyword arguments (line 669)
            kwargs_4726 = {}
            # Getting the type of 'LinkError' (line 669)
            LinkError_4724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 26), 'LinkError', False)
            # Calling LinkError(args, kwargs) (line 669)
            LinkError_call_result_4727 = invoke(stypy.reporting.localization.Localization(__file__, 669, 26), LinkError_4724, *[msg_4725], **kwargs_4726)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 669, 20), LinkError_call_result_4727, 'raise parameter', BaseException)
            # SSA join for try-except statement (line 665)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_4697:
                # SSA join for if statement (line 662)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA branch for the else part of an if statement (line 610)
        module_type_store.open_ssa_branch('else')
        
        # Call to debug(...): (line 671)
        # Processing the call arguments (line 671)
        str_4730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 22), 'str', 'skipping %s (up-to-date)')
        # Getting the type of 'output_filename' (line 671)
        output_filename_4731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 50), 'output_filename', False)
        # Processing the call keyword arguments (line 671)
        kwargs_4732 = {}
        # Getting the type of 'log' (line 671)
        log_4728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 12), 'log', False)
        # Obtaining the member 'debug' of a type (line 671)
        debug_4729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 12), log_4728, 'debug')
        # Calling debug(args, kwargs) (line 671)
        debug_call_result_4733 = invoke(stypy.reporting.localization.Localization(__file__, 671, 12), debug_4729, *[str_4730, output_filename_4731], **kwargs_4732)
        
        # SSA join for if statement (line 610)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'link(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'link' in the type store
        # Getting the type of 'stypy_return_type' (line 578)
        stypy_return_type_4734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4734)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'link'
        return stypy_return_type_4734


    @norecursion
    def manifest_setup_ldargs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'manifest_setup_ldargs'
        module_type_store = module_type_store.open_function_context('manifest_setup_ldargs', 673, 4, False)
        # Assigning a type to the variable 'self' (line 674)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MSVCCompiler.manifest_setup_ldargs.__dict__.__setitem__('stypy_localization', localization)
        MSVCCompiler.manifest_setup_ldargs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MSVCCompiler.manifest_setup_ldargs.__dict__.__setitem__('stypy_type_store', module_type_store)
        MSVCCompiler.manifest_setup_ldargs.__dict__.__setitem__('stypy_function_name', 'MSVCCompiler.manifest_setup_ldargs')
        MSVCCompiler.manifest_setup_ldargs.__dict__.__setitem__('stypy_param_names_list', ['output_filename', 'build_temp', 'ld_args'])
        MSVCCompiler.manifest_setup_ldargs.__dict__.__setitem__('stypy_varargs_param_name', None)
        MSVCCompiler.manifest_setup_ldargs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MSVCCompiler.manifest_setup_ldargs.__dict__.__setitem__('stypy_call_defaults', defaults)
        MSVCCompiler.manifest_setup_ldargs.__dict__.__setitem__('stypy_call_varargs', varargs)
        MSVCCompiler.manifest_setup_ldargs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MSVCCompiler.manifest_setup_ldargs.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MSVCCompiler.manifest_setup_ldargs', ['output_filename', 'build_temp', 'ld_args'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'manifest_setup_ldargs', localization, ['output_filename', 'build_temp', 'ld_args'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'manifest_setup_ldargs(...)' code ##################

        
        # Assigning a Call to a Name (line 680):
        
        # Assigning a Call to a Name (line 680):
        
        # Call to join(...): (line 680)
        # Processing the call arguments (line 680)
        # Getting the type of 'build_temp' (line 681)
        build_temp_4738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 16), 'build_temp', False)
        
        # Call to basename(...): (line 682)
        # Processing the call arguments (line 682)
        # Getting the type of 'output_filename' (line 682)
        output_filename_4742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 33), 'output_filename', False)
        # Processing the call keyword arguments (line 682)
        kwargs_4743 = {}
        # Getting the type of 'os' (line 682)
        os_4739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 16), 'os', False)
        # Obtaining the member 'path' of a type (line 682)
        path_4740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 16), os_4739, 'path')
        # Obtaining the member 'basename' of a type (line 682)
        basename_4741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 16), path_4740, 'basename')
        # Calling basename(args, kwargs) (line 682)
        basename_call_result_4744 = invoke(stypy.reporting.localization.Localization(__file__, 682, 16), basename_4741, *[output_filename_4742], **kwargs_4743)
        
        str_4745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 52), 'str', '.manifest')
        # Applying the binary operator '+' (line 682)
        result_add_4746 = python_operator(stypy.reporting.localization.Localization(__file__, 682, 16), '+', basename_call_result_4744, str_4745)
        
        # Processing the call keyword arguments (line 680)
        kwargs_4747 = {}
        # Getting the type of 'os' (line 680)
        os_4735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 680)
        path_4736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 24), os_4735, 'path')
        # Obtaining the member 'join' of a type (line 680)
        join_4737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 24), path_4736, 'join')
        # Calling join(args, kwargs) (line 680)
        join_call_result_4748 = invoke(stypy.reporting.localization.Localization(__file__, 680, 24), join_4737, *[build_temp_4738, result_add_4746], **kwargs_4747)
        
        # Assigning a type to the variable 'temp_manifest' (line 680)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 8), 'temp_manifest', join_call_result_4748)
        
        # Call to append(...): (line 683)
        # Processing the call arguments (line 683)
        str_4751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 23), 'str', '/MANIFESTFILE:')
        # Getting the type of 'temp_manifest' (line 683)
        temp_manifest_4752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 42), 'temp_manifest', False)
        # Applying the binary operator '+' (line 683)
        result_add_4753 = python_operator(stypy.reporting.localization.Localization(__file__, 683, 23), '+', str_4751, temp_manifest_4752)
        
        # Processing the call keyword arguments (line 683)
        kwargs_4754 = {}
        # Getting the type of 'ld_args' (line 683)
        ld_args_4749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 8), 'ld_args', False)
        # Obtaining the member 'append' of a type (line 683)
        append_4750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 683, 8), ld_args_4749, 'append')
        # Calling append(args, kwargs) (line 683)
        append_call_result_4755 = invoke(stypy.reporting.localization.Localization(__file__, 683, 8), append_4750, *[result_add_4753], **kwargs_4754)
        
        
        # ################# End of 'manifest_setup_ldargs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'manifest_setup_ldargs' in the type store
        # Getting the type of 'stypy_return_type' (line 673)
        stypy_return_type_4756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4756)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'manifest_setup_ldargs'
        return stypy_return_type_4756


    @norecursion
    def manifest_get_embed_info(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'manifest_get_embed_info'
        module_type_store = module_type_store.open_function_context('manifest_get_embed_info', 685, 4, False)
        # Assigning a type to the variable 'self' (line 686)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MSVCCompiler.manifest_get_embed_info.__dict__.__setitem__('stypy_localization', localization)
        MSVCCompiler.manifest_get_embed_info.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MSVCCompiler.manifest_get_embed_info.__dict__.__setitem__('stypy_type_store', module_type_store)
        MSVCCompiler.manifest_get_embed_info.__dict__.__setitem__('stypy_function_name', 'MSVCCompiler.manifest_get_embed_info')
        MSVCCompiler.manifest_get_embed_info.__dict__.__setitem__('stypy_param_names_list', ['target_desc', 'ld_args'])
        MSVCCompiler.manifest_get_embed_info.__dict__.__setitem__('stypy_varargs_param_name', None)
        MSVCCompiler.manifest_get_embed_info.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MSVCCompiler.manifest_get_embed_info.__dict__.__setitem__('stypy_call_defaults', defaults)
        MSVCCompiler.manifest_get_embed_info.__dict__.__setitem__('stypy_call_varargs', varargs)
        MSVCCompiler.manifest_get_embed_info.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MSVCCompiler.manifest_get_embed_info.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MSVCCompiler.manifest_get_embed_info', ['target_desc', 'ld_args'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'manifest_get_embed_info', localization, ['target_desc', 'ld_args'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'manifest_get_embed_info(...)' code ##################

        
        # Getting the type of 'ld_args' (line 690)
        ld_args_4757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 19), 'ld_args')
        # Testing the type of a for loop iterable (line 690)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 690, 8), ld_args_4757)
        # Getting the type of the for loop variable (line 690)
        for_loop_var_4758 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 690, 8), ld_args_4757)
        # Assigning a type to the variable 'arg' (line 690)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 8), 'arg', for_loop_var_4758)
        # SSA begins for a for statement (line 690)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to startswith(...): (line 691)
        # Processing the call arguments (line 691)
        str_4761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 30), 'str', '/MANIFESTFILE:')
        # Processing the call keyword arguments (line 691)
        kwargs_4762 = {}
        # Getting the type of 'arg' (line 691)
        arg_4759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 15), 'arg', False)
        # Obtaining the member 'startswith' of a type (line 691)
        startswith_4760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 15), arg_4759, 'startswith')
        # Calling startswith(args, kwargs) (line 691)
        startswith_call_result_4763 = invoke(stypy.reporting.localization.Localization(__file__, 691, 15), startswith_4760, *[str_4761], **kwargs_4762)
        
        # Testing the type of an if condition (line 691)
        if_condition_4764 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 691, 12), startswith_call_result_4763)
        # Assigning a type to the variable 'if_condition_4764' (line 691)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 12), 'if_condition_4764', if_condition_4764)
        # SSA begins for if statement (line 691)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 692):
        
        # Assigning a Subscript to a Name (line 692):
        
        # Obtaining the type of the subscript
        int_4765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 50), 'int')
        
        # Call to split(...): (line 692)
        # Processing the call arguments (line 692)
        str_4768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 42), 'str', ':')
        int_4769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 47), 'int')
        # Processing the call keyword arguments (line 692)
        kwargs_4770 = {}
        # Getting the type of 'arg' (line 692)
        arg_4766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 32), 'arg', False)
        # Obtaining the member 'split' of a type (line 692)
        split_4767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 32), arg_4766, 'split')
        # Calling split(args, kwargs) (line 692)
        split_call_result_4771 = invoke(stypy.reporting.localization.Localization(__file__, 692, 32), split_4767, *[str_4768, int_4769], **kwargs_4770)
        
        # Obtaining the member '__getitem__' of a type (line 692)
        getitem___4772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 32), split_call_result_4771, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 692)
        subscript_call_result_4773 = invoke(stypy.reporting.localization.Localization(__file__, 692, 32), getitem___4772, int_4765)
        
        # Assigning a type to the variable 'temp_manifest' (line 692)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 16), 'temp_manifest', subscript_call_result_4773)
        # SSA join for if statement (line 691)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of a for statement (line 690)
        module_type_store.open_ssa_branch('for loop else')
        # Getting the type of 'None' (line 696)
        None_4774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 696)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 12), 'stypy_return_type', None_4774)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'target_desc' (line 697)
        target_desc_4775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 11), 'target_desc')
        # Getting the type of 'CCompiler' (line 697)
        CCompiler_4776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 26), 'CCompiler')
        # Obtaining the member 'EXECUTABLE' of a type (line 697)
        EXECUTABLE_4777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 26), CCompiler_4776, 'EXECUTABLE')
        # Applying the binary operator '==' (line 697)
        result_eq_4778 = python_operator(stypy.reporting.localization.Localization(__file__, 697, 11), '==', target_desc_4775, EXECUTABLE_4777)
        
        # Testing the type of an if condition (line 697)
        if_condition_4779 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 697, 8), result_eq_4778)
        # Assigning a type to the variable 'if_condition_4779' (line 697)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 697, 8), 'if_condition_4779', if_condition_4779)
        # SSA begins for if statement (line 697)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 700):
        
        # Assigning a Num to a Name (line 700):
        int_4780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 19), 'int')
        # Assigning a type to the variable 'mfid' (line 700)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 12), 'mfid', int_4780)
        # SSA branch for the else part of an if statement (line 697)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 703):
        
        # Assigning a Num to a Name (line 703):
        int_4781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 703, 19), 'int')
        # Assigning a type to the variable 'mfid' (line 703)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 12), 'mfid', int_4781)
        
        # Assigning a Call to a Name (line 704):
        
        # Assigning a Call to a Name (line 704):
        
        # Call to _remove_visual_c_ref(...): (line 704)
        # Processing the call arguments (line 704)
        # Getting the type of 'temp_manifest' (line 704)
        temp_manifest_4784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 54), 'temp_manifest', False)
        # Processing the call keyword arguments (line 704)
        kwargs_4785 = {}
        # Getting the type of 'self' (line 704)
        self_4782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 28), 'self', False)
        # Obtaining the member '_remove_visual_c_ref' of a type (line 704)
        _remove_visual_c_ref_4783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 28), self_4782, '_remove_visual_c_ref')
        # Calling _remove_visual_c_ref(args, kwargs) (line 704)
        _remove_visual_c_ref_call_result_4786 = invoke(stypy.reporting.localization.Localization(__file__, 704, 28), _remove_visual_c_ref_4783, *[temp_manifest_4784], **kwargs_4785)
        
        # Assigning a type to the variable 'temp_manifest' (line 704)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 12), 'temp_manifest', _remove_visual_c_ref_call_result_4786)
        # SSA join for if statement (line 697)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 705)
        # Getting the type of 'temp_manifest' (line 705)
        temp_manifest_4787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 11), 'temp_manifest')
        # Getting the type of 'None' (line 705)
        None_4788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 28), 'None')
        
        (may_be_4789, more_types_in_union_4790) = may_be_none(temp_manifest_4787, None_4788)

        if may_be_4789:

            if more_types_in_union_4790:
                # Runtime conditional SSA (line 705)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'None' (line 706)
            None_4791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 19), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 706)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 12), 'stypy_return_type', None_4791)

            if more_types_in_union_4790:
                # SSA join for if statement (line 705)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Obtaining an instance of the builtin type 'tuple' (line 707)
        tuple_4792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 707)
        # Adding element type (line 707)
        # Getting the type of 'temp_manifest' (line 707)
        temp_manifest_4793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 15), 'temp_manifest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 707, 15), tuple_4792, temp_manifest_4793)
        # Adding element type (line 707)
        # Getting the type of 'mfid' (line 707)
        mfid_4794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 30), 'mfid')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 707, 15), tuple_4792, mfid_4794)
        
        # Assigning a type to the variable 'stypy_return_type' (line 707)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 8), 'stypy_return_type', tuple_4792)
        
        # ################# End of 'manifest_get_embed_info(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'manifest_get_embed_info' in the type store
        # Getting the type of 'stypy_return_type' (line 685)
        stypy_return_type_4795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4795)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'manifest_get_embed_info'
        return stypy_return_type_4795


    @norecursion
    def _remove_visual_c_ref(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_remove_visual_c_ref'
        module_type_store = module_type_store.open_function_context('_remove_visual_c_ref', 709, 4, False)
        # Assigning a type to the variable 'self' (line 710)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 710, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MSVCCompiler._remove_visual_c_ref.__dict__.__setitem__('stypy_localization', localization)
        MSVCCompiler._remove_visual_c_ref.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MSVCCompiler._remove_visual_c_ref.__dict__.__setitem__('stypy_type_store', module_type_store)
        MSVCCompiler._remove_visual_c_ref.__dict__.__setitem__('stypy_function_name', 'MSVCCompiler._remove_visual_c_ref')
        MSVCCompiler._remove_visual_c_ref.__dict__.__setitem__('stypy_param_names_list', ['manifest_file'])
        MSVCCompiler._remove_visual_c_ref.__dict__.__setitem__('stypy_varargs_param_name', None)
        MSVCCompiler._remove_visual_c_ref.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MSVCCompiler._remove_visual_c_ref.__dict__.__setitem__('stypy_call_defaults', defaults)
        MSVCCompiler._remove_visual_c_ref.__dict__.__setitem__('stypy_call_varargs', varargs)
        MSVCCompiler._remove_visual_c_ref.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MSVCCompiler._remove_visual_c_ref.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MSVCCompiler._remove_visual_c_ref', ['manifest_file'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_remove_visual_c_ref', localization, ['manifest_file'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_remove_visual_c_ref(...)' code ##################

        
        
        # SSA begins for try-except statement (line 710)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 719):
        
        # Assigning a Call to a Name (line 719):
        
        # Call to open(...): (line 719)
        # Processing the call arguments (line 719)
        # Getting the type of 'manifest_file' (line 719)
        manifest_file_4797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 30), 'manifest_file', False)
        # Processing the call keyword arguments (line 719)
        kwargs_4798 = {}
        # Getting the type of 'open' (line 719)
        open_4796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 25), 'open', False)
        # Calling open(args, kwargs) (line 719)
        open_call_result_4799 = invoke(stypy.reporting.localization.Localization(__file__, 719, 25), open_4796, *[manifest_file_4797], **kwargs_4798)
        
        # Assigning a type to the variable 'manifest_f' (line 719)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 12), 'manifest_f', open_call_result_4799)
        
        # Try-finally block (line 720)
        
        # Assigning a Call to a Name (line 721):
        
        # Assigning a Call to a Name (line 721):
        
        # Call to read(...): (line 721)
        # Processing the call keyword arguments (line 721)
        kwargs_4802 = {}
        # Getting the type of 'manifest_f' (line 721)
        manifest_f_4800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 31), 'manifest_f', False)
        # Obtaining the member 'read' of a type (line 721)
        read_4801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 721, 31), manifest_f_4800, 'read')
        # Calling read(args, kwargs) (line 721)
        read_call_result_4803 = invoke(stypy.reporting.localization.Localization(__file__, 721, 31), read_4801, *[], **kwargs_4802)
        
        # Assigning a type to the variable 'manifest_buf' (line 721)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 721, 16), 'manifest_buf', read_call_result_4803)
        
        # finally branch of the try-finally block (line 720)
        
        # Call to close(...): (line 723)
        # Processing the call keyword arguments (line 723)
        kwargs_4806 = {}
        # Getting the type of 'manifest_f' (line 723)
        manifest_f_4804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 16), 'manifest_f', False)
        # Obtaining the member 'close' of a type (line 723)
        close_4805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 16), manifest_f_4804, 'close')
        # Calling close(args, kwargs) (line 723)
        close_call_result_4807 = invoke(stypy.reporting.localization.Localization(__file__, 723, 16), close_4805, *[], **kwargs_4806)
        
        
        
        # Assigning a Call to a Name (line 724):
        
        # Assigning a Call to a Name (line 724):
        
        # Call to compile(...): (line 724)
        # Processing the call arguments (line 724)
        str_4810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 16), 'str', '<assemblyIdentity.*?name=("|\')Microsoft\\.VC\\d{2}\\.CRT("|\').*?(/>|</assemblyIdentity>)')
        # Getting the type of 're' (line 727)
        re_4811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 16), 're', False)
        # Obtaining the member 'DOTALL' of a type (line 727)
        DOTALL_4812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 727, 16), re_4811, 'DOTALL')
        # Processing the call keyword arguments (line 724)
        kwargs_4813 = {}
        # Getting the type of 're' (line 724)
        re_4808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 22), 're', False)
        # Obtaining the member 'compile' of a type (line 724)
        compile_4809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 724, 22), re_4808, 'compile')
        # Calling compile(args, kwargs) (line 724)
        compile_call_result_4814 = invoke(stypy.reporting.localization.Localization(__file__, 724, 22), compile_4809, *[str_4810, DOTALL_4812], **kwargs_4813)
        
        # Assigning a type to the variable 'pattern' (line 724)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 12), 'pattern', compile_call_result_4814)
        
        # Assigning a Call to a Name (line 728):
        
        # Assigning a Call to a Name (line 728):
        
        # Call to sub(...): (line 728)
        # Processing the call arguments (line 728)
        # Getting the type of 'pattern' (line 728)
        pattern_4817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 34), 'pattern', False)
        str_4818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 43), 'str', '')
        # Getting the type of 'manifest_buf' (line 728)
        manifest_buf_4819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 47), 'manifest_buf', False)
        # Processing the call keyword arguments (line 728)
        kwargs_4820 = {}
        # Getting the type of 're' (line 728)
        re_4815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 27), 're', False)
        # Obtaining the member 'sub' of a type (line 728)
        sub_4816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 728, 27), re_4815, 'sub')
        # Calling sub(args, kwargs) (line 728)
        sub_call_result_4821 = invoke(stypy.reporting.localization.Localization(__file__, 728, 27), sub_4816, *[pattern_4817, str_4818, manifest_buf_4819], **kwargs_4820)
        
        # Assigning a type to the variable 'manifest_buf' (line 728)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 12), 'manifest_buf', sub_call_result_4821)
        
        # Assigning a Str to a Name (line 729):
        
        # Assigning a Str to a Name (line 729):
        str_4822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 22), 'str', '<dependentAssembly>\\s*</dependentAssembly>')
        # Assigning a type to the variable 'pattern' (line 729)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 12), 'pattern', str_4822)
        
        # Assigning a Call to a Name (line 730):
        
        # Assigning a Call to a Name (line 730):
        
        # Call to sub(...): (line 730)
        # Processing the call arguments (line 730)
        # Getting the type of 'pattern' (line 730)
        pattern_4825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 34), 'pattern', False)
        str_4826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 730, 43), 'str', '')
        # Getting the type of 'manifest_buf' (line 730)
        manifest_buf_4827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 47), 'manifest_buf', False)
        # Processing the call keyword arguments (line 730)
        kwargs_4828 = {}
        # Getting the type of 're' (line 730)
        re_4823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 27), 're', False)
        # Obtaining the member 'sub' of a type (line 730)
        sub_4824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 730, 27), re_4823, 'sub')
        # Calling sub(args, kwargs) (line 730)
        sub_call_result_4829 = invoke(stypy.reporting.localization.Localization(__file__, 730, 27), sub_4824, *[pattern_4825, str_4826, manifest_buf_4827], **kwargs_4828)
        
        # Assigning a type to the variable 'manifest_buf' (line 730)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 12), 'manifest_buf', sub_call_result_4829)
        
        # Assigning a Call to a Name (line 733):
        
        # Assigning a Call to a Name (line 733):
        
        # Call to compile(...): (line 733)
        # Processing the call arguments (line 733)
        str_4832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 16), 'str', '<assemblyIdentity.*?name=(?:"|\')(.+?)(?:"|\').*?(?:/>|</assemblyIdentity>)')
        # Getting the type of 're' (line 735)
        re_4833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 54), 're', False)
        # Obtaining the member 'DOTALL' of a type (line 735)
        DOTALL_4834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 54), re_4833, 'DOTALL')
        # Processing the call keyword arguments (line 733)
        kwargs_4835 = {}
        # Getting the type of 're' (line 733)
        re_4830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 22), 're', False)
        # Obtaining the member 'compile' of a type (line 733)
        compile_4831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 733, 22), re_4830, 'compile')
        # Calling compile(args, kwargs) (line 733)
        compile_call_result_4836 = invoke(stypy.reporting.localization.Localization(__file__, 733, 22), compile_4831, *[str_4832, DOTALL_4834], **kwargs_4835)
        
        # Assigning a type to the variable 'pattern' (line 733)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 12), 'pattern', compile_call_result_4836)
        
        # Type idiom detected: calculating its left and rigth part (line 736)
        
        # Call to search(...): (line 736)
        # Processing the call arguments (line 736)
        # Getting the type of 'pattern' (line 736)
        pattern_4839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 25), 'pattern', False)
        # Getting the type of 'manifest_buf' (line 736)
        manifest_buf_4840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 34), 'manifest_buf', False)
        # Processing the call keyword arguments (line 736)
        kwargs_4841 = {}
        # Getting the type of 're' (line 736)
        re_4837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 15), 're', False)
        # Obtaining the member 'search' of a type (line 736)
        search_4838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 15), re_4837, 'search')
        # Calling search(args, kwargs) (line 736)
        search_call_result_4842 = invoke(stypy.reporting.localization.Localization(__file__, 736, 15), search_4838, *[pattern_4839, manifest_buf_4840], **kwargs_4841)
        
        # Getting the type of 'None' (line 736)
        None_4843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 51), 'None')
        
        (may_be_4844, more_types_in_union_4845) = may_be_none(search_call_result_4842, None_4843)

        if may_be_4844:

            if more_types_in_union_4845:
                # Runtime conditional SSA (line 736)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'None' (line 737)
            None_4846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 23), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 737)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 737, 16), 'stypy_return_type', None_4846)

            if more_types_in_union_4845:
                # SSA join for if statement (line 736)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 739):
        
        # Assigning a Call to a Name (line 739):
        
        # Call to open(...): (line 739)
        # Processing the call arguments (line 739)
        # Getting the type of 'manifest_file' (line 739)
        manifest_file_4848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 30), 'manifest_file', False)
        str_4849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 45), 'str', 'w')
        # Processing the call keyword arguments (line 739)
        kwargs_4850 = {}
        # Getting the type of 'open' (line 739)
        open_4847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 25), 'open', False)
        # Calling open(args, kwargs) (line 739)
        open_call_result_4851 = invoke(stypy.reporting.localization.Localization(__file__, 739, 25), open_4847, *[manifest_file_4848, str_4849], **kwargs_4850)
        
        # Assigning a type to the variable 'manifest_f' (line 739)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 12), 'manifest_f', open_call_result_4851)
        
        # Try-finally block (line 740)
        
        # Call to write(...): (line 741)
        # Processing the call arguments (line 741)
        # Getting the type of 'manifest_buf' (line 741)
        manifest_buf_4854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 33), 'manifest_buf', False)
        # Processing the call keyword arguments (line 741)
        kwargs_4855 = {}
        # Getting the type of 'manifest_f' (line 741)
        manifest_f_4852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 16), 'manifest_f', False)
        # Obtaining the member 'write' of a type (line 741)
        write_4853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 16), manifest_f_4852, 'write')
        # Calling write(args, kwargs) (line 741)
        write_call_result_4856 = invoke(stypy.reporting.localization.Localization(__file__, 741, 16), write_4853, *[manifest_buf_4854], **kwargs_4855)
        
        # Getting the type of 'manifest_file' (line 742)
        manifest_file_4857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 23), 'manifest_file')
        # Assigning a type to the variable 'stypy_return_type' (line 742)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 16), 'stypy_return_type', manifest_file_4857)
        
        # finally branch of the try-finally block (line 740)
        
        # Call to close(...): (line 744)
        # Processing the call keyword arguments (line 744)
        kwargs_4860 = {}
        # Getting the type of 'manifest_f' (line 744)
        manifest_f_4858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 16), 'manifest_f', False)
        # Obtaining the member 'close' of a type (line 744)
        close_4859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 16), manifest_f_4858, 'close')
        # Calling close(args, kwargs) (line 744)
        close_call_result_4861 = invoke(stypy.reporting.localization.Localization(__file__, 744, 16), close_4859, *[], **kwargs_4860)
        
        
        # SSA branch for the except part of a try statement (line 710)
        # SSA branch for the except 'IOError' branch of a try statement (line 710)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 710)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_remove_visual_c_ref(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_remove_visual_c_ref' in the type store
        # Getting the type of 'stypy_return_type' (line 709)
        stypy_return_type_4862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4862)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_remove_visual_c_ref'
        return stypy_return_type_4862


    @norecursion
    def library_dir_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'library_dir_option'
        module_type_store = module_type_store.open_function_context('library_dir_option', 752, 4, False)
        # Assigning a type to the variable 'self' (line 753)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 753, 4), 'self', type_of_self)
        
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

        str_4863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, 15), 'str', '/LIBPATH:')
        # Getting the type of 'dir' (line 753)
        dir_4864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 29), 'dir')
        # Applying the binary operator '+' (line 753)
        result_add_4865 = python_operator(stypy.reporting.localization.Localization(__file__, 753, 15), '+', str_4863, dir_4864)
        
        # Assigning a type to the variable 'stypy_return_type' (line 753)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 753, 8), 'stypy_return_type', result_add_4865)
        
        # ################# End of 'library_dir_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'library_dir_option' in the type store
        # Getting the type of 'stypy_return_type' (line 752)
        stypy_return_type_4866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4866)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'library_dir_option'
        return stypy_return_type_4866


    @norecursion
    def runtime_library_dir_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'runtime_library_dir_option'
        module_type_store = module_type_store.open_function_context('runtime_library_dir_option', 755, 4, False)
        # Assigning a type to the variable 'self' (line 756)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 4), 'self', type_of_self)
        
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

        
        # Call to DistutilsPlatformError(...): (line 756)
        # Processing the call arguments (line 756)
        str_4868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 14), 'str', "don't know how to set runtime library search path for MSVC++")
        # Processing the call keyword arguments (line 756)
        kwargs_4869 = {}
        # Getting the type of 'DistutilsPlatformError' (line 756)
        DistutilsPlatformError_4867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 14), 'DistutilsPlatformError', False)
        # Calling DistutilsPlatformError(args, kwargs) (line 756)
        DistutilsPlatformError_call_result_4870 = invoke(stypy.reporting.localization.Localization(__file__, 756, 14), DistutilsPlatformError_4867, *[str_4868], **kwargs_4869)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 756, 8), DistutilsPlatformError_call_result_4870, 'raise parameter', BaseException)
        
        # ################# End of 'runtime_library_dir_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'runtime_library_dir_option' in the type store
        # Getting the type of 'stypy_return_type' (line 755)
        stypy_return_type_4871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4871)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'runtime_library_dir_option'
        return stypy_return_type_4871


    @norecursion
    def library_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'library_option'
        module_type_store = module_type_store.open_function_context('library_option', 759, 4, False)
        # Assigning a type to the variable 'self' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 4), 'self', type_of_self)
        
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

        
        # Call to library_filename(...): (line 760)
        # Processing the call arguments (line 760)
        # Getting the type of 'lib' (line 760)
        lib_4874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 37), 'lib', False)
        # Processing the call keyword arguments (line 760)
        kwargs_4875 = {}
        # Getting the type of 'self' (line 760)
        self_4872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 15), 'self', False)
        # Obtaining the member 'library_filename' of a type (line 760)
        library_filename_4873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 15), self_4872, 'library_filename')
        # Calling library_filename(args, kwargs) (line 760)
        library_filename_call_result_4876 = invoke(stypy.reporting.localization.Localization(__file__, 760, 15), library_filename_4873, *[lib_4874], **kwargs_4875)
        
        # Assigning a type to the variable 'stypy_return_type' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 8), 'stypy_return_type', library_filename_call_result_4876)
        
        # ################# End of 'library_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'library_option' in the type store
        # Getting the type of 'stypy_return_type' (line 759)
        stypy_return_type_4877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4877)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'library_option'
        return stypy_return_type_4877


    @norecursion
    def find_library_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_4878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 49), 'int')
        defaults = [int_4878]
        # Create a new context for function 'find_library_file'
        module_type_store = module_type_store.open_function_context('find_library_file', 763, 4, False)
        # Assigning a type to the variable 'self' (line 764)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 4), 'self', type_of_self)
        
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

        
        # Getting the type of 'debug' (line 766)
        debug_4879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 11), 'debug')
        # Testing the type of an if condition (line 766)
        if_condition_4880 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 766, 8), debug_4879)
        # Assigning a type to the variable 'if_condition_4880' (line 766)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 8), 'if_condition_4880', if_condition_4880)
        # SSA begins for if statement (line 766)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 767):
        
        # Assigning a List to a Name (line 767):
        
        # Obtaining an instance of the builtin type 'list' (line 767)
        list_4881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 767)
        # Adding element type (line 767)
        # Getting the type of 'lib' (line 767)
        lib_4882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 25), 'lib')
        str_4883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 31), 'str', '_d')
        # Applying the binary operator '+' (line 767)
        result_add_4884 = python_operator(stypy.reporting.localization.Localization(__file__, 767, 25), '+', lib_4882, str_4883)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 767, 24), list_4881, result_add_4884)
        # Adding element type (line 767)
        # Getting the type of 'lib' (line 767)
        lib_4885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 37), 'lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 767, 24), list_4881, lib_4885)
        
        # Assigning a type to the variable 'try_names' (line 767)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 12), 'try_names', list_4881)
        # SSA branch for the else part of an if statement (line 766)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Name (line 769):
        
        # Assigning a List to a Name (line 769):
        
        # Obtaining an instance of the builtin type 'list' (line 769)
        list_4886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 769)
        # Adding element type (line 769)
        # Getting the type of 'lib' (line 769)
        lib_4887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 25), 'lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 24), list_4886, lib_4887)
        
        # Assigning a type to the variable 'try_names' (line 769)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 769, 12), 'try_names', list_4886)
        # SSA join for if statement (line 766)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'dirs' (line 770)
        dirs_4888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 19), 'dirs')
        # Testing the type of a for loop iterable (line 770)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 770, 8), dirs_4888)
        # Getting the type of the for loop variable (line 770)
        for_loop_var_4889 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 770, 8), dirs_4888)
        # Assigning a type to the variable 'dir' (line 770)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 770, 8), 'dir', for_loop_var_4889)
        # SSA begins for a for statement (line 770)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'try_names' (line 771)
        try_names_4890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 24), 'try_names')
        # Testing the type of a for loop iterable (line 771)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 771, 12), try_names_4890)
        # Getting the type of the for loop variable (line 771)
        for_loop_var_4891 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 771, 12), try_names_4890)
        # Assigning a type to the variable 'name' (line 771)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 12), 'name', for_loop_var_4891)
        # SSA begins for a for statement (line 771)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 772):
        
        # Assigning a Call to a Name (line 772):
        
        # Call to join(...): (line 772)
        # Processing the call arguments (line 772)
        # Getting the type of 'dir' (line 772)
        dir_4895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 39), 'dir', False)
        
        # Call to library_filename(...): (line 772)
        # Processing the call arguments (line 772)
        # Getting the type of 'name' (line 772)
        name_4898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 67), 'name', False)
        # Processing the call keyword arguments (line 772)
        kwargs_4899 = {}
        # Getting the type of 'self' (line 772)
        self_4896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 44), 'self', False)
        # Obtaining the member 'library_filename' of a type (line 772)
        library_filename_4897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 772, 44), self_4896, 'library_filename')
        # Calling library_filename(args, kwargs) (line 772)
        library_filename_call_result_4900 = invoke(stypy.reporting.localization.Localization(__file__, 772, 44), library_filename_4897, *[name_4898], **kwargs_4899)
        
        # Processing the call keyword arguments (line 772)
        kwargs_4901 = {}
        # Getting the type of 'os' (line 772)
        os_4892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 772)
        path_4893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 772, 26), os_4892, 'path')
        # Obtaining the member 'join' of a type (line 772)
        join_4894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 772, 26), path_4893, 'join')
        # Calling join(args, kwargs) (line 772)
        join_call_result_4902 = invoke(stypy.reporting.localization.Localization(__file__, 772, 26), join_4894, *[dir_4895, library_filename_call_result_4900], **kwargs_4901)
        
        # Assigning a type to the variable 'libfile' (line 772)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 772, 16), 'libfile', join_call_result_4902)
        
        
        # Call to exists(...): (line 773)
        # Processing the call arguments (line 773)
        # Getting the type of 'libfile' (line 773)
        libfile_4906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 34), 'libfile', False)
        # Processing the call keyword arguments (line 773)
        kwargs_4907 = {}
        # Getting the type of 'os' (line 773)
        os_4903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 773)
        path_4904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 19), os_4903, 'path')
        # Obtaining the member 'exists' of a type (line 773)
        exists_4905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 19), path_4904, 'exists')
        # Calling exists(args, kwargs) (line 773)
        exists_call_result_4908 = invoke(stypy.reporting.localization.Localization(__file__, 773, 19), exists_4905, *[libfile_4906], **kwargs_4907)
        
        # Testing the type of an if condition (line 773)
        if_condition_4909 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 773, 16), exists_call_result_4908)
        # Assigning a type to the variable 'if_condition_4909' (line 773)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 16), 'if_condition_4909', if_condition_4909)
        # SSA begins for if statement (line 773)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'libfile' (line 774)
        libfile_4910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 27), 'libfile')
        # Assigning a type to the variable 'stypy_return_type' (line 774)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 774, 20), 'stypy_return_type', libfile_4910)
        # SSA join for if statement (line 773)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of a for statement (line 770)
        module_type_store.open_ssa_branch('for loop else')
        # Getting the type of 'None' (line 777)
        None_4911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 777)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 777, 12), 'stypy_return_type', None_4911)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'find_library_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find_library_file' in the type store
        # Getting the type of 'stypy_return_type' (line 763)
        stypy_return_type_4912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4912)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find_library_file'
        return stypy_return_type_4912


    @norecursion
    def find_exe(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'find_exe'
        module_type_store = module_type_store.open_function_context('find_exe', 781, 4, False)
        # Assigning a type to the variable 'self' (line 782)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 782, 4), 'self', type_of_self)
        
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

        str_4913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 789, (-1)), 'str', "Return path to an MSVC executable program.\n\n        Tries to find the program in several places: first, one of the\n        MSVC program search paths from the registry; next, the directories\n        in the PATH environment variable.  If any of those work, return an\n        absolute path that is known to exist.  If none of them work, just\n        return the original program name, 'exe'.\n        ")
        
        # Getting the type of 'self' (line 790)
        self_4914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 17), 'self')
        # Obtaining the member '__paths' of a type (line 790)
        paths_4915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 790, 17), self_4914, '__paths')
        # Testing the type of a for loop iterable (line 790)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 790, 8), paths_4915)
        # Getting the type of the for loop variable (line 790)
        for_loop_var_4916 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 790, 8), paths_4915)
        # Assigning a type to the variable 'p' (line 790)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 790, 8), 'p', for_loop_var_4916)
        # SSA begins for a for statement (line 790)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 791):
        
        # Assigning a Call to a Name (line 791):
        
        # Call to join(...): (line 791)
        # Processing the call arguments (line 791)
        
        # Call to abspath(...): (line 791)
        # Processing the call arguments (line 791)
        # Getting the type of 'p' (line 791)
        p_4923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 46), 'p', False)
        # Processing the call keyword arguments (line 791)
        kwargs_4924 = {}
        # Getting the type of 'os' (line 791)
        os_4920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 30), 'os', False)
        # Obtaining the member 'path' of a type (line 791)
        path_4921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 30), os_4920, 'path')
        # Obtaining the member 'abspath' of a type (line 791)
        abspath_4922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 30), path_4921, 'abspath')
        # Calling abspath(args, kwargs) (line 791)
        abspath_call_result_4925 = invoke(stypy.reporting.localization.Localization(__file__, 791, 30), abspath_4922, *[p_4923], **kwargs_4924)
        
        # Getting the type of 'exe' (line 791)
        exe_4926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 50), 'exe', False)
        # Processing the call keyword arguments (line 791)
        kwargs_4927 = {}
        # Getting the type of 'os' (line 791)
        os_4917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 791)
        path_4918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 17), os_4917, 'path')
        # Obtaining the member 'join' of a type (line 791)
        join_4919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 17), path_4918, 'join')
        # Calling join(args, kwargs) (line 791)
        join_call_result_4928 = invoke(stypy.reporting.localization.Localization(__file__, 791, 17), join_4919, *[abspath_call_result_4925, exe_4926], **kwargs_4927)
        
        # Assigning a type to the variable 'fn' (line 791)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 791, 12), 'fn', join_call_result_4928)
        
        
        # Call to isfile(...): (line 792)
        # Processing the call arguments (line 792)
        # Getting the type of 'fn' (line 792)
        fn_4932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 30), 'fn', False)
        # Processing the call keyword arguments (line 792)
        kwargs_4933 = {}
        # Getting the type of 'os' (line 792)
        os_4929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 792)
        path_4930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 792, 15), os_4929, 'path')
        # Obtaining the member 'isfile' of a type (line 792)
        isfile_4931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 792, 15), path_4930, 'isfile')
        # Calling isfile(args, kwargs) (line 792)
        isfile_call_result_4934 = invoke(stypy.reporting.localization.Localization(__file__, 792, 15), isfile_4931, *[fn_4932], **kwargs_4933)
        
        # Testing the type of an if condition (line 792)
        if_condition_4935 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 792, 12), isfile_call_result_4934)
        # Assigning a type to the variable 'if_condition_4935' (line 792)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 792, 12), 'if_condition_4935', if_condition_4935)
        # SSA begins for if statement (line 792)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'fn' (line 793)
        fn_4936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 23), 'fn')
        # Assigning a type to the variable 'stypy_return_type' (line 793)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 793, 16), 'stypy_return_type', fn_4936)
        # SSA join for if statement (line 792)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to split(...): (line 796)
        # Processing the call arguments (line 796)
        str_4943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 796, 42), 'str', ';')
        # Processing the call keyword arguments (line 796)
        kwargs_4944 = {}
        
        # Obtaining the type of the subscript
        str_4937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 796, 28), 'str', 'Path')
        # Getting the type of 'os' (line 796)
        os_4938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 17), 'os', False)
        # Obtaining the member 'environ' of a type (line 796)
        environ_4939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 796, 17), os_4938, 'environ')
        # Obtaining the member '__getitem__' of a type (line 796)
        getitem___4940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 796, 17), environ_4939, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 796)
        subscript_call_result_4941 = invoke(stypy.reporting.localization.Localization(__file__, 796, 17), getitem___4940, str_4937)
        
        # Obtaining the member 'split' of a type (line 796)
        split_4942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 796, 17), subscript_call_result_4941, 'split')
        # Calling split(args, kwargs) (line 796)
        split_call_result_4945 = invoke(stypy.reporting.localization.Localization(__file__, 796, 17), split_4942, *[str_4943], **kwargs_4944)
        
        # Testing the type of a for loop iterable (line 796)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 796, 8), split_call_result_4945)
        # Getting the type of the for loop variable (line 796)
        for_loop_var_4946 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 796, 8), split_call_result_4945)
        # Assigning a type to the variable 'p' (line 796)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 796, 8), 'p', for_loop_var_4946)
        # SSA begins for a for statement (line 796)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 797):
        
        # Assigning a Call to a Name (line 797):
        
        # Call to join(...): (line 797)
        # Processing the call arguments (line 797)
        
        # Call to abspath(...): (line 797)
        # Processing the call arguments (line 797)
        # Getting the type of 'p' (line 797)
        p_4953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 46), 'p', False)
        # Processing the call keyword arguments (line 797)
        kwargs_4954 = {}
        # Getting the type of 'os' (line 797)
        os_4950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 30), 'os', False)
        # Obtaining the member 'path' of a type (line 797)
        path_4951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 797, 30), os_4950, 'path')
        # Obtaining the member 'abspath' of a type (line 797)
        abspath_4952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 797, 30), path_4951, 'abspath')
        # Calling abspath(args, kwargs) (line 797)
        abspath_call_result_4955 = invoke(stypy.reporting.localization.Localization(__file__, 797, 30), abspath_4952, *[p_4953], **kwargs_4954)
        
        # Getting the type of 'exe' (line 797)
        exe_4956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 49), 'exe', False)
        # Processing the call keyword arguments (line 797)
        kwargs_4957 = {}
        # Getting the type of 'os' (line 797)
        os_4947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 797)
        path_4948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 797, 17), os_4947, 'path')
        # Obtaining the member 'join' of a type (line 797)
        join_4949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 797, 17), path_4948, 'join')
        # Calling join(args, kwargs) (line 797)
        join_call_result_4958 = invoke(stypy.reporting.localization.Localization(__file__, 797, 17), join_4949, *[abspath_call_result_4955, exe_4956], **kwargs_4957)
        
        # Assigning a type to the variable 'fn' (line 797)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 12), 'fn', join_call_result_4958)
        
        
        # Call to isfile(...): (line 798)
        # Processing the call arguments (line 798)
        # Getting the type of 'fn' (line 798)
        fn_4962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 30), 'fn', False)
        # Processing the call keyword arguments (line 798)
        kwargs_4963 = {}
        # Getting the type of 'os' (line 798)
        os_4959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 798)
        path_4960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 798, 15), os_4959, 'path')
        # Obtaining the member 'isfile' of a type (line 798)
        isfile_4961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 798, 15), path_4960, 'isfile')
        # Calling isfile(args, kwargs) (line 798)
        isfile_call_result_4964 = invoke(stypy.reporting.localization.Localization(__file__, 798, 15), isfile_4961, *[fn_4962], **kwargs_4963)
        
        # Testing the type of an if condition (line 798)
        if_condition_4965 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 798, 12), isfile_call_result_4964)
        # Assigning a type to the variable 'if_condition_4965' (line 798)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 798, 12), 'if_condition_4965', if_condition_4965)
        # SSA begins for if statement (line 798)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'fn' (line 799)
        fn_4966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 23), 'fn')
        # Assigning a type to the variable 'stypy_return_type' (line 799)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 799, 16), 'stypy_return_type', fn_4966)
        # SSA join for if statement (line 798)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'exe' (line 801)
        exe_4967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 15), 'exe')
        # Assigning a type to the variable 'stypy_return_type' (line 801)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 801, 8), 'stypy_return_type', exe_4967)
        
        # ################# End of 'find_exe(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find_exe' in the type store
        # Getting the type of 'stypy_return_type' (line 781)
        stypy_return_type_4968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4968)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find_exe'
        return stypy_return_type_4968


# Assigning a type to the variable 'MSVCCompiler' (line 309)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 0), 'MSVCCompiler', MSVCCompiler)

# Assigning a Str to a Name (line 313):
str_4969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 20), 'str', 'msvc')
# Getting the type of 'MSVCCompiler'
MSVCCompiler_4970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_4970, 'compiler_type', str_4969)

# Assigning a Dict to a Name (line 320):

# Obtaining an instance of the builtin type 'dict' (line 320)
dict_4971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 320)

# Getting the type of 'MSVCCompiler'
MSVCCompiler_4972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member 'executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_4972, 'executables', dict_4971)

# Assigning a List to a Name (line 323):

# Obtaining an instance of the builtin type 'list' (line 323)
list_4973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 323)
# Adding element type (line 323)
str_4974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 21), 'str', '.c')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 20), list_4973, str_4974)

# Getting the type of 'MSVCCompiler'
MSVCCompiler_4975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member '_c_extensions' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_4975, '_c_extensions', list_4973)

# Assigning a List to a Name (line 324):

# Obtaining an instance of the builtin type 'list' (line 324)
list_4976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 324)
# Adding element type (line 324)
str_4977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 23), 'str', '.cc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 22), list_4976, str_4977)
# Adding element type (line 324)
str_4978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 30), 'str', '.cpp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 22), list_4976, str_4978)
# Adding element type (line 324)
str_4979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 38), 'str', '.cxx')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 22), list_4976, str_4979)

# Getting the type of 'MSVCCompiler'
MSVCCompiler_4980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member '_cpp_extensions' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_4980, '_cpp_extensions', list_4976)

# Assigning a List to a Name (line 325):

# Obtaining an instance of the builtin type 'list' (line 325)
list_4981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 325)
# Adding element type (line 325)
str_4982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 22), 'str', '.rc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 21), list_4981, str_4982)

# Getting the type of 'MSVCCompiler'
MSVCCompiler_4983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member '_rc_extensions' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_4983, '_rc_extensions', list_4981)

# Assigning a List to a Name (line 326):

# Obtaining an instance of the builtin type 'list' (line 326)
list_4984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 326)
# Adding element type (line 326)
str_4985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 22), 'str', '.mc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 21), list_4984, str_4985)

# Getting the type of 'MSVCCompiler'
MSVCCompiler_4986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member '_mc_extensions' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_4986, '_mc_extensions', list_4984)

# Assigning a BinOp to a Name (line 330):
# Getting the type of 'MSVCCompiler'
MSVCCompiler_4987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Obtaining the member '_c_extensions' of a type
_c_extensions_4988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_4987, '_c_extensions')
# Getting the type of 'MSVCCompiler'
MSVCCompiler_4989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Obtaining the member '_cpp_extensions' of a type
_cpp_extensions_4990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_4989, '_cpp_extensions')
# Applying the binary operator '+' (line 330)
result_add_4991 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 22), '+', _c_extensions_4988, _cpp_extensions_4990)

# Getting the type of 'MSVCCompiler'
MSVCCompiler_4992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Obtaining the member '_rc_extensions' of a type
_rc_extensions_4993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_4992, '_rc_extensions')
# Applying the binary operator '+' (line 330)
result_add_4994 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 54), '+', result_add_4991, _rc_extensions_4993)

# Getting the type of 'MSVCCompiler'
MSVCCompiler_4995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Obtaining the member '_mc_extensions' of a type
_mc_extensions_4996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_4995, '_mc_extensions')
# Applying the binary operator '+' (line 331)
result_add_4997 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 37), '+', result_add_4994, _mc_extensions_4996)

# Getting the type of 'MSVCCompiler'
MSVCCompiler_4998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member 'src_extensions' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_4998, 'src_extensions', result_add_4997)

# Assigning a Str to a Name (line 332):
str_4999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 20), 'str', '.res')
# Getting the type of 'MSVCCompiler'
MSVCCompiler_5000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member 'res_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_5000, 'res_extension', str_4999)

# Assigning a Str to a Name (line 333):
str_5001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 20), 'str', '.obj')
# Getting the type of 'MSVCCompiler'
MSVCCompiler_5002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member 'obj_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_5002, 'obj_extension', str_5001)

# Assigning a Str to a Name (line 334):
str_5003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 27), 'str', '.lib')
# Getting the type of 'MSVCCompiler'
MSVCCompiler_5004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member 'static_lib_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_5004, 'static_lib_extension', str_5003)

# Assigning a Str to a Name (line 335):
str_5005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 27), 'str', '.dll')
# Getting the type of 'MSVCCompiler'
MSVCCompiler_5006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member 'shared_lib_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_5006, 'shared_lib_extension', str_5005)

# Assigning a Str to a Name (line 336):
str_5007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 44), 'str', '%s%s')
# Getting the type of 'MSVCCompiler'
MSVCCompiler_5008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member 'shared_lib_format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_5008, 'shared_lib_format', str_5007)

# Assigning a Name to a Name (line 336):
# Getting the type of 'MSVCCompiler'
MSVCCompiler_5009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Obtaining the member 'shared_lib_format' of a type
shared_lib_format_5010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_5009, 'shared_lib_format')
# Getting the type of 'MSVCCompiler'
MSVCCompiler_5011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member 'static_lib_format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_5011, 'static_lib_format', shared_lib_format_5010)

# Assigning a Str to a Name (line 337):
str_5012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 20), 'str', '.exe')
# Getting the type of 'MSVCCompiler'
MSVCCompiler_5013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MSVCCompiler')
# Setting the type of the member 'exe_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MSVCCompiler_5013, 'exe_extension', str_5012)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Support code for building Python extensions on Windows.
3: 
4:     # NT stuff
5:     # 1. Make sure libpython<version>.a exists for gcc.  If not, build it.
6:     # 2. Force windows to use gcc (we're struggling with MSVC and g77 support)
7:     # 3. Force windows to use g77
8: 
9: '''
10: from __future__ import division, absolute_import, print_function
11: 
12: import os
13: import sys
14: import subprocess
15: import re
16: 
17: # Overwrite certain distutils.ccompiler functions:
18: import numpy.distutils.ccompiler
19: 
20: if sys.version_info[0] < 3:
21:     from . import log
22: else:
23:     from numpy.distutils import log
24: # NT stuff
25: # 1. Make sure libpython<version>.a exists for gcc.  If not, build it.
26: # 2. Force windows to use gcc (we're struggling with MSVC and g77 support)
27: #    --> this is done in numpy/distutils/ccompiler.py
28: # 3. Force windows to use g77
29: 
30: import distutils.cygwinccompiler
31: from distutils.version import StrictVersion
32: from numpy.distutils.ccompiler import gen_preprocess_options, gen_lib_options
33: from distutils.unixccompiler import UnixCCompiler
34: from distutils.msvccompiler import get_build_version as get_build_msvc_version
35: from distutils.errors import (DistutilsExecError, CompileError,
36:                               UnknownFileError)
37: from numpy.distutils.misc_util import (msvc_runtime_library,
38:                                        get_build_architecture)
39: 
40: # Useful to generate table of symbols from a dll
41: _START = re.compile(r'\[Ordinal/Name Pointer\] Table')
42: _TABLE = re.compile(r'^\s+\[([\s*[0-9]*)\] ([a-zA-Z0-9_]*)')
43: 
44: # the same as cygwin plus some additional parameters
45: class Mingw32CCompiler(distutils.cygwinccompiler.CygwinCCompiler):
46:     ''' A modified MingW32 compiler compatible with an MSVC built Python.
47: 
48:     '''
49: 
50:     compiler_type = 'mingw32'
51: 
52:     def __init__ (self,
53:                   verbose=0,
54:                   dry_run=0,
55:                   force=0):
56: 
57:         distutils.cygwinccompiler.CygwinCCompiler.__init__ (self, verbose,
58:                                                             dry_run, force)
59: 
60:         # we need to support 3.2 which doesn't match the standard
61:         # get_versions methods regex
62:         if self.gcc_version is None:
63:             import re
64:             p = subprocess.Popen(['gcc', '-dumpversion'], shell=True,
65:                                  stdout=subprocess.PIPE)
66:             out_string = p.stdout.read()
67:             p.stdout.close()
68:             result = re.search('(\d+\.\d+)', out_string)
69:             if result:
70:                 self.gcc_version = StrictVersion(result.group(1))
71: 
72:         # A real mingw32 doesn't need to specify a different entry point,
73:         # but cygwin 2.91.57 in no-cygwin-mode needs it.
74:         if self.gcc_version <= "2.91.57":
75:             entry_point = '--entry _DllMain@12'
76:         else:
77:             entry_point = ''
78: 
79:         if self.linker_dll == 'dllwrap':
80:             # Commented out '--driver-name g++' part that fixes weird
81:             #   g++.exe: g++: No such file or directory
82:             # error (mingw 1.0 in Enthon24 tree, gcc-3.4.5).
83:             # If the --driver-name part is required for some environment
84:             # then make the inclusion of this part specific to that
85:             # environment.
86:             self.linker = 'dllwrap' #  --driver-name g++'
87:         elif self.linker_dll == 'gcc':
88:             self.linker = 'g++'
89: 
90:         # **changes: eric jones 4/11/01
91:         # 1. Check for import library on Windows.  Build if it doesn't exist.
92: 
93:         build_import_library()
94: 
95:         # Check for custom msvc runtime library on Windows. Build if it doesn't exist.
96:         msvcr_success = build_msvcr_library()
97:         msvcr_dbg_success = build_msvcr_library(debug=True)
98:         if msvcr_success or msvcr_dbg_success:
99:             # add preprocessor statement for using customized msvcr lib
100:             self.define_macro('NPY_MINGW_USE_CUSTOM_MSVCR')
101: 
102:         # Define the MSVC version as hint for MinGW
103:         msvcr_version = '0x%03i0' % int(msvc_runtime_library().lstrip('msvcr'))
104:         self.define_macro('__MSVCRT_VERSION__', msvcr_version)
105: 
106:         # MS_WIN64 should be defined when building for amd64 on windows,
107:         # but python headers define it only for MS compilers, which has all
108:         # kind of bad consequences, like using Py_ModuleInit4 instead of
109:         # Py_ModuleInit4_64, etc... So we add it here
110:         if get_build_architecture() == 'AMD64':
111:             if self.gcc_version < "4.0":
112:                 self.set_executables(
113:                     compiler='gcc -g -DDEBUG -DMS_WIN64 -mno-cygwin -O0 -Wall',
114:                     compiler_so='gcc -g -DDEBUG -DMS_WIN64 -mno-cygwin -O0'
115:                                 ' -Wall -Wstrict-prototypes',
116:                     linker_exe='gcc -g -mno-cygwin',
117:                     linker_so='gcc -g -mno-cygwin -shared')
118:             else:
119:                 # gcc-4 series releases do not support -mno-cygwin option
120:                 self.set_executables(
121:                     compiler='gcc -g -DDEBUG -DMS_WIN64 -O0 -Wall',
122:                     compiler_so='gcc -g -DDEBUG -DMS_WIN64 -O0 -Wall -Wstrict-prototypes',
123:                     linker_exe='gcc -g',
124:                     linker_so='gcc -g -shared')
125:         else:
126:             if self.gcc_version <= "3.0.0":
127:                 self.set_executables(
128:                     compiler='gcc -mno-cygwin -O2 -w',
129:                     compiler_so='gcc -mno-cygwin -mdll -O2 -w'
130:                                 ' -Wstrict-prototypes',
131:                     linker_exe='g++ -mno-cygwin',
132:                     linker_so='%s -mno-cygwin -mdll -static %s' %
133:                               (self.linker, entry_point))
134:             elif self.gcc_version < "4.0":
135:                 self.set_executables(
136:                     compiler='gcc -mno-cygwin -O2 -Wall',
137:                     compiler_so='gcc -mno-cygwin -O2 -Wall'
138:                                 ' -Wstrict-prototypes',
139:                     linker_exe='g++ -mno-cygwin',
140:                     linker_so='g++ -mno-cygwin -shared')
141:             else:
142:                 # gcc-4 series releases do not support -mno-cygwin option
143:                 self.set_executables(compiler='gcc -O2 -Wall',
144:                                      compiler_so='gcc -O2 -Wall -Wstrict-prototypes',
145:                                      linker_exe='g++ ',
146:                                      linker_so='g++ -shared')
147:         # added for python2.3 support
148:         # we can't pass it through set_executables because pre 2.2 would fail
149:         self.compiler_cxx = ['g++']
150: 
151:         # Maybe we should also append -mthreads, but then the finished dlls
152:         # need another dll (mingwm10.dll see Mingw32 docs) (-mthreads: Support
153:         # thread-safe exception handling on `Mingw32')
154: 
155:         # no additional libraries needed
156:         #self.dll_libraries=[]
157:         return
158: 
159:     # __init__ ()
160: 
161:     def link(self,
162:              target_desc,
163:              objects,
164:              output_filename,
165:              output_dir,
166:              libraries,
167:              library_dirs,
168:              runtime_library_dirs,
169:              export_symbols = None,
170:              debug=0,
171:              extra_preargs=None,
172:              extra_postargs=None,
173:              build_temp=None,
174:              target_lang=None):
175:         # Include the appropiate MSVC runtime library if Python was built
176:         # with MSVC >= 7.0 (MinGW standard is msvcrt)
177:         runtime_library = msvc_runtime_library()
178:         if runtime_library:
179:             if not libraries:
180:                 libraries = []
181:             libraries.append(runtime_library)
182:         args = (self,
183:                 target_desc,
184:                 objects,
185:                 output_filename,
186:                 output_dir,
187:                 libraries,
188:                 library_dirs,
189:                 runtime_library_dirs,
190:                 None, #export_symbols, we do this in our def-file
191:                 debug,
192:                 extra_preargs,
193:                 extra_postargs,
194:                 build_temp,
195:                 target_lang)
196:         if self.gcc_version < "3.0.0":
197:             func = distutils.cygwinccompiler.CygwinCCompiler.link
198:         else:
199:             func = UnixCCompiler.link
200:         func(*args[:func.__code__.co_argcount])
201:         return
202: 
203:     def object_filenames (self,
204:                           source_filenames,
205:                           strip_dir=0,
206:                           output_dir=''):
207:         if output_dir is None: output_dir = ''
208:         obj_names = []
209:         for src_name in source_filenames:
210:             # use normcase to make sure '.rc' is really '.rc' and not '.RC'
211:             (base, ext) = os.path.splitext (os.path.normcase(src_name))
212: 
213:             # added these lines to strip off windows drive letters
214:             # without it, .o files are placed next to .c files
215:             # instead of the build directory
216:             drv, base = os.path.splitdrive(base)
217:             if drv:
218:                 base = base[1:]
219: 
220:             if ext not in (self.src_extensions + ['.rc', '.res']):
221:                 raise UnknownFileError(
222:                       "unknown file type '%s' (from '%s')" % \
223:                       (ext, src_name))
224:             if strip_dir:
225:                 base = os.path.basename (base)
226:             if ext == '.res' or ext == '.rc':
227:                 # these need to be compiled to object files
228:                 obj_names.append (os.path.join (output_dir,
229:                                                 base + ext + self.obj_extension))
230:             else:
231:                 obj_names.append (os.path.join (output_dir,
232:                                                 base + self.obj_extension))
233:         return obj_names
234: 
235:     # object_filenames ()
236: 
237: 
238: def find_python_dll():
239:     maj, min, micro = [int(i) for i in sys.version_info[:3]]
240:     dllname = 'python%d%d.dll' % (maj, min)
241:     print("Looking for %s" % dllname)
242: 
243:     # We can't do much here:
244:     # - find it in python main dir
245:     # - in system32,
246:     # - ortherwise (Sxs), I don't know how to get it.
247:     lib_dirs = [sys.prefix, os.path.join(sys.prefix, 'lib')]
248:     try:
249:         lib_dirs.append(os.path.join(os.environ['SYSTEMROOT'], 'system32'))
250:     except KeyError:
251:         pass
252: 
253:     for d in lib_dirs:
254:         dll = os.path.join(d, dllname)
255:         if os.path.exists(dll):
256:             return dll
257: 
258:     raise ValueError("%s not found in %s" % (dllname, lib_dirs))
259: 
260: def dump_table(dll):
261:     st = subprocess.Popen(["objdump.exe", "-p", dll], stdout=subprocess.PIPE)
262:     return st.stdout.readlines()
263: 
264: def generate_def(dll, dfile):
265:     '''Given a dll file location,  get all its exported symbols and dump them
266:     into the given def file.
267: 
268:     The .def file will be overwritten'''
269:     dump = dump_table(dll)
270:     for i in range(len(dump)):
271:         if _START.match(dump[i].decode()):
272:             break
273:     else:
274:         raise ValueError("Symbol table not found")
275: 
276:     syms = []
277:     for j in range(i+1, len(dump)):
278:         m = _TABLE.match(dump[j].decode())
279:         if m:
280:             syms.append((int(m.group(1).strip()), m.group(2)))
281:         else:
282:             break
283: 
284:     if len(syms) == 0:
285:         log.warn('No symbols found in %s' % dll)
286: 
287:     d = open(dfile, 'w')
288:     d.write('LIBRARY        %s\n' % os.path.basename(dll))
289:     d.write(';CODE          PRELOAD MOVEABLE DISCARDABLE\n')
290:     d.write(';DATA          PRELOAD SINGLE\n')
291:     d.write('\nEXPORTS\n')
292:     for s in syms:
293:         #d.write('@%d    %s\n' % (s[0], s[1]))
294:         d.write('%s\n' % s[1])
295:     d.close()
296: 
297: def find_dll(dll_name):
298: 
299:     arch = {'AMD64' : 'amd64',
300:             'Intel' : 'x86'}[get_build_architecture()]
301: 
302:     def _find_dll_in_winsxs(dll_name):
303:         # Walk through the WinSxS directory to find the dll.
304:         winsxs_path = os.path.join(os.environ['WINDIR'], 'winsxs')
305:         if not os.path.exists(winsxs_path):
306:             return None
307:         for root, dirs, files in os.walk(winsxs_path):
308:             if dll_name in files and arch in root:
309:                 return os.path.join(root, dll_name)
310:         return None
311: 
312:     def _find_dll_in_path(dll_name):
313:         # First, look in the Python directory, then scan PATH for
314:         # the given dll name.
315:         for path in [sys.prefix] + os.environ['PATH'].split(';'):
316:             filepath = os.path.join(path, dll_name)
317:             if os.path.exists(filepath):
318:                 return os.path.abspath(filepath)
319: 
320:     return _find_dll_in_winsxs(dll_name) or _find_dll_in_path(dll_name)
321: 
322: def build_msvcr_library(debug=False):
323:     if os.name != 'nt':
324:         return False
325: 
326:     msvcr_name = msvc_runtime_library()
327: 
328:     # Skip using a custom library for versions < MSVC 8.0
329:     if int(msvcr_name.lstrip('msvcr')) < 80:
330:         log.debug('Skip building msvcr library:'
331:                   ' custom functionality not present')
332:         return False
333: 
334:     if debug:
335:         msvcr_name += 'd'
336: 
337:     # Skip if custom library already exists
338:     out_name = "lib%s.a" % msvcr_name
339:     out_file = os.path.join(sys.prefix, 'libs', out_name)
340:     if os.path.isfile(out_file):
341:         log.debug('Skip building msvcr library: "%s" exists' %
342:                   (out_file,))
343:         return True
344: 
345:     # Find the msvcr dll
346:     msvcr_dll_name = msvcr_name + '.dll'
347:     dll_file = find_dll(msvcr_dll_name)
348:     if not dll_file:
349:         log.warn('Cannot build msvcr library: "%s" not found' %
350:                  msvcr_dll_name)
351:         return False
352: 
353:     def_name = "lib%s.def" % msvcr_name
354:     def_file = os.path.join(sys.prefix, 'libs', def_name)
355: 
356:     log.info('Building msvcr library: "%s" (from %s)' \
357:              % (out_file, dll_file))
358: 
359:     # Generate a symbol definition file from the msvcr dll
360:     generate_def(dll_file, def_file)
361: 
362:     # Create a custom mingw library for the given symbol definitions
363:     cmd = ['dlltool', '-d', def_file, '-l', out_file]
364:     retcode = subprocess.call(cmd)
365: 
366:     # Clean up symbol definitions
367:     os.remove(def_file)
368: 
369:     return (not retcode)
370: 
371: def build_import_library():
372:     if os.name != 'nt':
373:         return
374: 
375:     arch = get_build_architecture()
376:     if arch == 'AMD64':
377:         return _build_import_library_amd64()
378:     elif arch == 'Intel':
379:         return _build_import_library_x86()
380:     else:
381:         raise ValueError("Unhandled arch %s" % arch)
382: 
383: def _build_import_library_amd64():
384:     dll_file = find_python_dll()
385: 
386:     out_name = "libpython%d%d.a" % tuple(sys.version_info[:2])
387:     out_file = os.path.join(sys.prefix, 'libs', out_name)
388:     if os.path.isfile(out_file):
389:         log.debug('Skip building import library: "%s" exists' %
390:                   (out_file))
391:         return
392: 
393:     def_name = "python%d%d.def" % tuple(sys.version_info[:2])
394:     def_file = os.path.join(sys.prefix, 'libs', def_name)
395: 
396:     log.info('Building import library (arch=AMD64): "%s" (from %s)' %
397:              (out_file, dll_file))
398: 
399:     generate_def(dll_file, def_file)
400: 
401:     cmd = ['dlltool', '-d', def_file, '-l', out_file]
402:     subprocess.Popen(cmd)
403: 
404: def _build_import_library_x86():
405:     ''' Build the import libraries for Mingw32-gcc on Windows
406:     '''
407:     lib_name = "python%d%d.lib" % tuple(sys.version_info[:2])
408:     lib_file = os.path.join(sys.prefix, 'libs', lib_name)
409:     out_name = "libpython%d%d.a" % tuple(sys.version_info[:2])
410:     out_file = os.path.join(sys.prefix, 'libs', out_name)
411:     if not os.path.isfile(lib_file):
412:         log.warn('Cannot build import library: "%s" not found' % (lib_file))
413:         return
414:     if os.path.isfile(out_file):
415:         log.debug('Skip building import library: "%s" exists' % (out_file))
416:         return
417:     log.info('Building import library (ARCH=x86): "%s"' % (out_file))
418: 
419:     from numpy.distutils import lib2def
420: 
421:     def_name = "python%d%d.def" % tuple(sys.version_info[:2])
422:     def_file = os.path.join(sys.prefix, 'libs', def_name)
423:     nm_cmd = '%s %s' % (lib2def.DEFAULT_NM, lib_file)
424:     nm_output = lib2def.getnm(nm_cmd)
425:     dlist, flist = lib2def.parse_nm(nm_output)
426:     lib2def.output_def(dlist, flist, lib2def.DEF_HEADER, open(def_file, 'w'))
427: 
428:     dll_name = "python%d%d.dll" % tuple(sys.version_info[:2])
429:     args = (dll_name, def_file, out_file)
430:     cmd = 'dlltool --dllname %s --def %s --output-lib %s' % args
431:     status = os.system(cmd)
432:     # for now, fail silently
433:     if status:
434:         log.warn('Failed to build import library for gcc. Linking will fail.')
435:     return
436: 
437: #=====================================
438: # Dealing with Visual Studio MANIFESTS
439: #=====================================
440: 
441: # Functions to deal with visual studio manifests. Manifest are a mechanism to
442: # enforce strong DLL versioning on windows, and has nothing to do with
443: # distutils MANIFEST. manifests are XML files with version info, and used by
444: # the OS loader; they are necessary when linking against a DLL not in the
445: # system path; in particular, official python 2.6 binary is built against the
446: # MS runtime 9 (the one from VS 2008), which is not available on most windows
447: # systems; python 2.6 installer does install it in the Win SxS (Side by side)
448: # directory, but this requires the manifest for this to work. This is a big
449: # mess, thanks MS for a wonderful system.
450: 
451: # XXX: ideally, we should use exactly the same version as used by python. I
452: # submitted a patch to get this version, but it was only included for python
453: # 2.6.1 and above. So for versions below, we use a "best guess".
454: _MSVCRVER_TO_FULLVER = {}
455: if sys.platform == 'win32':
456:     try:
457:         import msvcrt
458:         # I took one version in my SxS directory: no idea if it is the good
459:         # one, and we can't retrieve it from python
460:         _MSVCRVER_TO_FULLVER['80'] = "8.0.50727.42"
461:         _MSVCRVER_TO_FULLVER['90'] = "9.0.21022.8"
462:         # Value from msvcrt.CRT_ASSEMBLY_VERSION under Python 3.3.0
463:         # on Windows XP:
464:         _MSVCRVER_TO_FULLVER['100'] = "10.0.30319.460"
465:         if hasattr(msvcrt, "CRT_ASSEMBLY_VERSION"):
466:             major, minor, rest = msvcrt.CRT_ASSEMBLY_VERSION.split(".", 2)
467:             _MSVCRVER_TO_FULLVER[major + minor] = msvcrt.CRT_ASSEMBLY_VERSION
468:             del major, minor, rest
469:     except ImportError:
470:         # If we are here, means python was not built with MSVC. Not sure what
471:         # to do in that case: manifest building will fail, but it should not be
472:         # used in that case anyway
473:         log.warn('Cannot import msvcrt: using manifest will not be possible')
474: 
475: def msvc_manifest_xml(maj, min):
476:     '''Given a major and minor version of the MSVCR, returns the
477:     corresponding XML file.'''
478:     try:
479:         fullver = _MSVCRVER_TO_FULLVER[str(maj * 10 + min)]
480:     except KeyError:
481:         raise ValueError("Version %d,%d of MSVCRT not supported yet" %
482:                          (maj, min))
483:     # Don't be fooled, it looks like an XML, but it is not. In particular, it
484:     # should not have any space before starting, and its size should be
485:     # divisible by 4, most likely for alignement constraints when the xml is
486:     # embedded in the binary...
487:     # This template was copied directly from the python 2.6 binary (using
488:     # strings.exe from mingw on python.exe).
489:     template = '''\
490: <assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
491:   <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
492:     <security>
493:       <requestedPrivileges>
494:         <requestedExecutionLevel level="asInvoker" uiAccess="false"></requestedExecutionLevel>
495:       </requestedPrivileges>
496:     </security>
497:   </trustInfo>
498:   <dependency>
499:     <dependentAssembly>
500:       <assemblyIdentity type="win32" name="Microsoft.VC%(maj)d%(min)d.CRT" version="%(fullver)s" processorArchitecture="*" publicKeyToken="1fc8b3b9a1e18e3b"></assemblyIdentity>
501:     </dependentAssembly>
502:   </dependency>
503: </assembly>'''
504: 
505:     return template % {'fullver': fullver, 'maj': maj, 'min': min}
506: 
507: def manifest_rc(name, type='dll'):
508:     '''Return the rc file used to generate the res file which will be embedded
509:     as manifest for given manifest file name, of given type ('dll' or
510:     'exe').
511: 
512:     Parameters
513:     ----------
514:     name : str
515:             name of the manifest file to embed
516:     type : str {'dll', 'exe'}
517:             type of the binary which will embed the manifest
518: 
519:     '''
520:     if type == 'dll':
521:         rctype = 2
522:     elif type == 'exe':
523:         rctype = 1
524:     else:
525:         raise ValueError("Type %s not supported" % type)
526: 
527:     return '''\
528: #include "winuser.h"
529: %d RT_MANIFEST %s''' % (rctype, name)
530: 
531: def check_embedded_msvcr_match_linked(msver):
532:     '''msver is the ms runtime version used for the MANIFEST.'''
533:     # check msvcr major version are the same for linking and
534:     # embedding
535:     msvcv = msvc_runtime_library()
536:     if msvcv:
537:         assert msvcv.startswith("msvcr"), msvcv
538:         # Dealing with something like "mscvr90" or "mscvr100", the last
539:         # last digit is the minor release, want int("9") or int("10"):
540:         maj = int(msvcv[5:-1])
541:         if not maj == int(msver):
542:             raise ValueError(
543:                   "Discrepancy between linked msvcr " \
544:                   "(%d) and the one about to be embedded " \
545:                   "(%d)" % (int(msver), maj))
546: 
547: def configtest_name(config):
548:     base = os.path.basename(config._gen_temp_sourcefile("yo", [], "c"))
549:     return os.path.splitext(base)[0]
550: 
551: def manifest_name(config):
552:     # Get configest name (including suffix)
553:     root = configtest_name(config)
554:     exext = config.compiler.exe_extension
555:     return root + exext + ".manifest"
556: 
557: def rc_name(config):
558:     # Get configtest name (including suffix)
559:     root = configtest_name(config)
560:     return root + ".rc"
561: 
562: def generate_manifest(config):
563:     msver = get_build_msvc_version()
564:     if msver is not None:
565:         if msver >= 8:
566:             check_embedded_msvcr_match_linked(msver)
567:             ma = int(msver)
568:             mi = int((msver - ma) * 10)
569:             # Write the manifest file
570:             manxml = msvc_manifest_xml(ma, mi)
571:             man = open(manifest_name(config), "w")
572:             config.temp_files.append(manifest_name(config))
573:             man.write(manxml)
574:             man.close()
575: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_36932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, (-1)), 'str', "\nSupport code for building Python extensions on Windows.\n\n    # NT stuff\n    # 1. Make sure libpython<version>.a exists for gcc.  If not, build it.\n    # 2. Force windows to use gcc (we're struggling with MSVC and g77 support)\n    # 3. Force windows to use g77\n\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import os' statement (line 12)
import os

import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import sys' statement (line 13)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import subprocess' statement (line 14)
import subprocess

import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'subprocess', subprocess, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'import re' statement (line 15)
import re

import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'import numpy.distutils.ccompiler' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_36933 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.distutils.ccompiler')

if (type(import_36933) is not StypyTypeError):

    if (import_36933 != 'pyd_module'):
        __import__(import_36933)
        sys_modules_36934 = sys.modules[import_36933]
        import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.distutils.ccompiler', sys_modules_36934.module_type_store, module_type_store)
    else:
        import numpy.distutils.ccompiler

        import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.distutils.ccompiler', numpy.distutils.ccompiler, module_type_store)

else:
    # Assigning a type to the variable 'numpy.distutils.ccompiler' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.distutils.ccompiler', import_36933)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')




# Obtaining the type of the subscript
int_36935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 20), 'int')
# Getting the type of 'sys' (line 20)
sys_36936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 3), 'sys')
# Obtaining the member 'version_info' of a type (line 20)
version_info_36937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 3), sys_36936, 'version_info')
# Obtaining the member '__getitem__' of a type (line 20)
getitem___36938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 3), version_info_36937, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 20)
subscript_call_result_36939 = invoke(stypy.reporting.localization.Localization(__file__, 20, 3), getitem___36938, int_36935)

int_36940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 25), 'int')
# Applying the binary operator '<' (line 20)
result_lt_36941 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 3), '<', subscript_call_result_36939, int_36940)

# Testing the type of an if condition (line 20)
if_condition_36942 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 20, 0), result_lt_36941)
# Assigning a type to the variable 'if_condition_36942' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'if_condition_36942', if_condition_36942)
# SSA begins for if statement (line 20)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 4))

# 'from numpy.distutils import log' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_36943 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 4), 'numpy.distutils')

if (type(import_36943) is not StypyTypeError):

    if (import_36943 != 'pyd_module'):
        __import__(import_36943)
        sys_modules_36944 = sys.modules[import_36943]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 4), 'numpy.distutils', sys_modules_36944.module_type_store, module_type_store, ['log'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 4), __file__, sys_modules_36944, sys_modules_36944.module_type_store, module_type_store)
    else:
        from numpy.distutils import log

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 4), 'numpy.distutils', None, module_type_store, ['log'], [log])

else:
    # Assigning a type to the variable 'numpy.distutils' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'numpy.distutils', import_36943)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

# SSA branch for the else part of an if statement (line 20)
module_type_store.open_ssa_branch('else')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 4))

# 'from numpy.distutils import log' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_36945 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 4), 'numpy.distutils')

if (type(import_36945) is not StypyTypeError):

    if (import_36945 != 'pyd_module'):
        __import__(import_36945)
        sys_modules_36946 = sys.modules[import_36945]
        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 4), 'numpy.distutils', sys_modules_36946.module_type_store, module_type_store, ['log'])
        nest_module(stypy.reporting.localization.Localization(__file__, 23, 4), __file__, sys_modules_36946, sys_modules_36946.module_type_store, module_type_store)
    else:
        from numpy.distutils import log

        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 4), 'numpy.distutils', None, module_type_store, ['log'], [log])

else:
    # Assigning a type to the variable 'numpy.distutils' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'numpy.distutils', import_36945)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

# SSA join for if statement (line 20)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 0))

# 'import distutils.cygwinccompiler' statement (line 30)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_36947 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'distutils.cygwinccompiler')

if (type(import_36947) is not StypyTypeError):

    if (import_36947 != 'pyd_module'):
        __import__(import_36947)
        sys_modules_36948 = sys.modules[import_36947]
        import_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'distutils.cygwinccompiler', sys_modules_36948.module_type_store, module_type_store)
    else:
        import distutils.cygwinccompiler

        import_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'distutils.cygwinccompiler', distutils.cygwinccompiler, module_type_store)

else:
    # Assigning a type to the variable 'distutils.cygwinccompiler' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'distutils.cygwinccompiler', import_36947)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 31, 0))

# 'from distutils.version import StrictVersion' statement (line 31)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_36949 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'distutils.version')

if (type(import_36949) is not StypyTypeError):

    if (import_36949 != 'pyd_module'):
        __import__(import_36949)
        sys_modules_36950 = sys.modules[import_36949]
        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'distutils.version', sys_modules_36950.module_type_store, module_type_store, ['StrictVersion'])
        nest_module(stypy.reporting.localization.Localization(__file__, 31, 0), __file__, sys_modules_36950, sys_modules_36950.module_type_store, module_type_store)
    else:
        from distutils.version import StrictVersion

        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'distutils.version', None, module_type_store, ['StrictVersion'], [StrictVersion])

else:
    # Assigning a type to the variable 'distutils.version' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'distutils.version', import_36949)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 32, 0))

# 'from numpy.distutils.ccompiler import gen_preprocess_options, gen_lib_options' statement (line 32)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_36951 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'numpy.distutils.ccompiler')

if (type(import_36951) is not StypyTypeError):

    if (import_36951 != 'pyd_module'):
        __import__(import_36951)
        sys_modules_36952 = sys.modules[import_36951]
        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'numpy.distutils.ccompiler', sys_modules_36952.module_type_store, module_type_store, ['gen_preprocess_options', 'gen_lib_options'])
        nest_module(stypy.reporting.localization.Localization(__file__, 32, 0), __file__, sys_modules_36952, sys_modules_36952.module_type_store, module_type_store)
    else:
        from numpy.distutils.ccompiler import gen_preprocess_options, gen_lib_options

        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'numpy.distutils.ccompiler', None, module_type_store, ['gen_preprocess_options', 'gen_lib_options'], [gen_preprocess_options, gen_lib_options])

else:
    # Assigning a type to the variable 'numpy.distutils.ccompiler' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'numpy.distutils.ccompiler', import_36951)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 33, 0))

# 'from distutils.unixccompiler import UnixCCompiler' statement (line 33)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_36953 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'distutils.unixccompiler')

if (type(import_36953) is not StypyTypeError):

    if (import_36953 != 'pyd_module'):
        __import__(import_36953)
        sys_modules_36954 = sys.modules[import_36953]
        import_from_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'distutils.unixccompiler', sys_modules_36954.module_type_store, module_type_store, ['UnixCCompiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 33, 0), __file__, sys_modules_36954, sys_modules_36954.module_type_store, module_type_store)
    else:
        from distutils.unixccompiler import UnixCCompiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'distutils.unixccompiler', None, module_type_store, ['UnixCCompiler'], [UnixCCompiler])

else:
    # Assigning a type to the variable 'distutils.unixccompiler' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'distutils.unixccompiler', import_36953)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 34, 0))

# 'from distutils.msvccompiler import get_build_msvc_version' statement (line 34)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_36955 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'distutils.msvccompiler')

if (type(import_36955) is not StypyTypeError):

    if (import_36955 != 'pyd_module'):
        __import__(import_36955)
        sys_modules_36956 = sys.modules[import_36955]
        import_from_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'distutils.msvccompiler', sys_modules_36956.module_type_store, module_type_store, ['get_build_version'])
        nest_module(stypy.reporting.localization.Localization(__file__, 34, 0), __file__, sys_modules_36956, sys_modules_36956.module_type_store, module_type_store)
    else:
        from distutils.msvccompiler import get_build_version as get_build_msvc_version

        import_from_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'distutils.msvccompiler', None, module_type_store, ['get_build_version'], [get_build_msvc_version])

else:
    # Assigning a type to the variable 'distutils.msvccompiler' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'distutils.msvccompiler', import_36955)

# Adding an alias
module_type_store.add_alias('get_build_msvc_version', 'get_build_version')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 35, 0))

# 'from distutils.errors import DistutilsExecError, CompileError, UnknownFileError' statement (line 35)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_36957 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'distutils.errors')

if (type(import_36957) is not StypyTypeError):

    if (import_36957 != 'pyd_module'):
        __import__(import_36957)
        sys_modules_36958 = sys.modules[import_36957]
        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'distutils.errors', sys_modules_36958.module_type_store, module_type_store, ['DistutilsExecError', 'CompileError', 'UnknownFileError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 35, 0), __file__, sys_modules_36958, sys_modules_36958.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsExecError, CompileError, UnknownFileError

        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'distutils.errors', None, module_type_store, ['DistutilsExecError', 'CompileError', 'UnknownFileError'], [DistutilsExecError, CompileError, UnknownFileError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'distutils.errors', import_36957)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 37, 0))

# 'from numpy.distutils.misc_util import msvc_runtime_library, get_build_architecture' statement (line 37)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_36959 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'numpy.distutils.misc_util')

if (type(import_36959) is not StypyTypeError):

    if (import_36959 != 'pyd_module'):
        __import__(import_36959)
        sys_modules_36960 = sys.modules[import_36959]
        import_from_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'numpy.distutils.misc_util', sys_modules_36960.module_type_store, module_type_store, ['msvc_runtime_library', 'get_build_architecture'])
        nest_module(stypy.reporting.localization.Localization(__file__, 37, 0), __file__, sys_modules_36960, sys_modules_36960.module_type_store, module_type_store)
    else:
        from numpy.distutils.misc_util import msvc_runtime_library, get_build_architecture

        import_from_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'numpy.distutils.misc_util', None, module_type_store, ['msvc_runtime_library', 'get_build_architecture'], [msvc_runtime_library, get_build_architecture])

else:
    # Assigning a type to the variable 'numpy.distutils.misc_util' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'numpy.distutils.misc_util', import_36959)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')


# Assigning a Call to a Name (line 41):

# Assigning a Call to a Name (line 41):

# Call to compile(...): (line 41)
# Processing the call arguments (line 41)
str_36963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 20), 'str', '\\[Ordinal/Name Pointer\\] Table')
# Processing the call keyword arguments (line 41)
kwargs_36964 = {}
# Getting the type of 're' (line 41)
re_36961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 9), 're', False)
# Obtaining the member 'compile' of a type (line 41)
compile_36962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 9), re_36961, 'compile')
# Calling compile(args, kwargs) (line 41)
compile_call_result_36965 = invoke(stypy.reporting.localization.Localization(__file__, 41, 9), compile_36962, *[str_36963], **kwargs_36964)

# Assigning a type to the variable '_START' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), '_START', compile_call_result_36965)

# Assigning a Call to a Name (line 42):

# Assigning a Call to a Name (line 42):

# Call to compile(...): (line 42)
# Processing the call arguments (line 42)
str_36968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 20), 'str', '^\\s+\\[([\\s*[0-9]*)\\] ([a-zA-Z0-9_]*)')
# Processing the call keyword arguments (line 42)
kwargs_36969 = {}
# Getting the type of 're' (line 42)
re_36966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 9), 're', False)
# Obtaining the member 'compile' of a type (line 42)
compile_36967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 9), re_36966, 'compile')
# Calling compile(args, kwargs) (line 42)
compile_call_result_36970 = invoke(stypy.reporting.localization.Localization(__file__, 42, 9), compile_36967, *[str_36968], **kwargs_36969)

# Assigning a type to the variable '_TABLE' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), '_TABLE', compile_call_result_36970)
# Declaration of the 'Mingw32CCompiler' class
# Getting the type of 'distutils' (line 45)
distutils_36971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 23), 'distutils')
# Obtaining the member 'cygwinccompiler' of a type (line 45)
cygwinccompiler_36972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 23), distutils_36971, 'cygwinccompiler')
# Obtaining the member 'CygwinCCompiler' of a type (line 45)
CygwinCCompiler_36973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 23), cygwinccompiler_36972, 'CygwinCCompiler')

class Mingw32CCompiler(CygwinCCompiler_36973, ):
    str_36974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, (-1)), 'str', ' A modified MingW32 compiler compatible with an MSVC built Python.\n\n    ')
    
    # Assigning a Str to a Name (line 50):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_36975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 26), 'int')
        int_36976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 26), 'int')
        int_36977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 24), 'int')
        defaults = [int_36975, int_36976, int_36977]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 52, 4, False)
        # Assigning a type to the variable 'self' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Mingw32CCompiler.__init__', ['verbose', 'dry_run', 'force'], None, None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'self' (line 57)
        self_36982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 60), 'self', False)
        # Getting the type of 'verbose' (line 57)
        verbose_36983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 66), 'verbose', False)
        # Getting the type of 'dry_run' (line 58)
        dry_run_36984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 60), 'dry_run', False)
        # Getting the type of 'force' (line 58)
        force_36985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 69), 'force', False)
        # Processing the call keyword arguments (line 57)
        kwargs_36986 = {}
        # Getting the type of 'distutils' (line 57)
        distutils_36978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'distutils', False)
        # Obtaining the member 'cygwinccompiler' of a type (line 57)
        cygwinccompiler_36979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), distutils_36978, 'cygwinccompiler')
        # Obtaining the member 'CygwinCCompiler' of a type (line 57)
        CygwinCCompiler_36980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), cygwinccompiler_36979, 'CygwinCCompiler')
        # Obtaining the member '__init__' of a type (line 57)
        init___36981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), CygwinCCompiler_36980, '__init__')
        # Calling __init__(args, kwargs) (line 57)
        init___call_result_36987 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), init___36981, *[self_36982, verbose_36983, dry_run_36984, force_36985], **kwargs_36986)
        
        
        # Type idiom detected: calculating its left and rigth part (line 62)
        # Getting the type of 'self' (line 62)
        self_36988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 11), 'self')
        # Obtaining the member 'gcc_version' of a type (line 62)
        gcc_version_36989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 11), self_36988, 'gcc_version')
        # Getting the type of 'None' (line 62)
        None_36990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 31), 'None')
        
        (may_be_36991, more_types_in_union_36992) = may_be_none(gcc_version_36989, None_36990)

        if may_be_36991:

            if more_types_in_union_36992:
                # Runtime conditional SSA (line 62)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 63, 12))
            
            # 'import re' statement (line 63)
            import re

            import_module(stypy.reporting.localization.Localization(__file__, 63, 12), 're', re, module_type_store)
            
            
            # Assigning a Call to a Name (line 64):
            
            # Assigning a Call to a Name (line 64):
            
            # Call to Popen(...): (line 64)
            # Processing the call arguments (line 64)
            
            # Obtaining an instance of the builtin type 'list' (line 64)
            list_36995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 33), 'list')
            # Adding type elements to the builtin type 'list' instance (line 64)
            # Adding element type (line 64)
            str_36996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 34), 'str', 'gcc')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 33), list_36995, str_36996)
            # Adding element type (line 64)
            str_36997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 41), 'str', '-dumpversion')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 33), list_36995, str_36997)
            
            # Processing the call keyword arguments (line 64)
            # Getting the type of 'True' (line 64)
            True_36998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 64), 'True', False)
            keyword_36999 = True_36998
            # Getting the type of 'subprocess' (line 65)
            subprocess_37000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 40), 'subprocess', False)
            # Obtaining the member 'PIPE' of a type (line 65)
            PIPE_37001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 40), subprocess_37000, 'PIPE')
            keyword_37002 = PIPE_37001
            kwargs_37003 = {'shell': keyword_36999, 'stdout': keyword_37002}
            # Getting the type of 'subprocess' (line 64)
            subprocess_36993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 16), 'subprocess', False)
            # Obtaining the member 'Popen' of a type (line 64)
            Popen_36994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 16), subprocess_36993, 'Popen')
            # Calling Popen(args, kwargs) (line 64)
            Popen_call_result_37004 = invoke(stypy.reporting.localization.Localization(__file__, 64, 16), Popen_36994, *[list_36995], **kwargs_37003)
            
            # Assigning a type to the variable 'p' (line 64)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'p', Popen_call_result_37004)
            
            # Assigning a Call to a Name (line 66):
            
            # Assigning a Call to a Name (line 66):
            
            # Call to read(...): (line 66)
            # Processing the call keyword arguments (line 66)
            kwargs_37008 = {}
            # Getting the type of 'p' (line 66)
            p_37005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'p', False)
            # Obtaining the member 'stdout' of a type (line 66)
            stdout_37006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 25), p_37005, 'stdout')
            # Obtaining the member 'read' of a type (line 66)
            read_37007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 25), stdout_37006, 'read')
            # Calling read(args, kwargs) (line 66)
            read_call_result_37009 = invoke(stypy.reporting.localization.Localization(__file__, 66, 25), read_37007, *[], **kwargs_37008)
            
            # Assigning a type to the variable 'out_string' (line 66)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'out_string', read_call_result_37009)
            
            # Call to close(...): (line 67)
            # Processing the call keyword arguments (line 67)
            kwargs_37013 = {}
            # Getting the type of 'p' (line 67)
            p_37010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'p', False)
            # Obtaining the member 'stdout' of a type (line 67)
            stdout_37011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 12), p_37010, 'stdout')
            # Obtaining the member 'close' of a type (line 67)
            close_37012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 12), stdout_37011, 'close')
            # Calling close(args, kwargs) (line 67)
            close_call_result_37014 = invoke(stypy.reporting.localization.Localization(__file__, 67, 12), close_37012, *[], **kwargs_37013)
            
            
            # Assigning a Call to a Name (line 68):
            
            # Assigning a Call to a Name (line 68):
            
            # Call to search(...): (line 68)
            # Processing the call arguments (line 68)
            str_37017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 31), 'str', '(\\d+\\.\\d+)')
            # Getting the type of 'out_string' (line 68)
            out_string_37018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 45), 'out_string', False)
            # Processing the call keyword arguments (line 68)
            kwargs_37019 = {}
            # Getting the type of 're' (line 68)
            re_37015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 21), 're', False)
            # Obtaining the member 'search' of a type (line 68)
            search_37016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 21), re_37015, 'search')
            # Calling search(args, kwargs) (line 68)
            search_call_result_37020 = invoke(stypy.reporting.localization.Localization(__file__, 68, 21), search_37016, *[str_37017, out_string_37018], **kwargs_37019)
            
            # Assigning a type to the variable 'result' (line 68)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'result', search_call_result_37020)
            
            # Getting the type of 'result' (line 69)
            result_37021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 15), 'result')
            # Testing the type of an if condition (line 69)
            if_condition_37022 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 12), result_37021)
            # Assigning a type to the variable 'if_condition_37022' (line 69)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'if_condition_37022', if_condition_37022)
            # SSA begins for if statement (line 69)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Attribute (line 70):
            
            # Assigning a Call to a Attribute (line 70):
            
            # Call to StrictVersion(...): (line 70)
            # Processing the call arguments (line 70)
            
            # Call to group(...): (line 70)
            # Processing the call arguments (line 70)
            int_37026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 62), 'int')
            # Processing the call keyword arguments (line 70)
            kwargs_37027 = {}
            # Getting the type of 'result' (line 70)
            result_37024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 49), 'result', False)
            # Obtaining the member 'group' of a type (line 70)
            group_37025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 49), result_37024, 'group')
            # Calling group(args, kwargs) (line 70)
            group_call_result_37028 = invoke(stypy.reporting.localization.Localization(__file__, 70, 49), group_37025, *[int_37026], **kwargs_37027)
            
            # Processing the call keyword arguments (line 70)
            kwargs_37029 = {}
            # Getting the type of 'StrictVersion' (line 70)
            StrictVersion_37023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 35), 'StrictVersion', False)
            # Calling StrictVersion(args, kwargs) (line 70)
            StrictVersion_call_result_37030 = invoke(stypy.reporting.localization.Localization(__file__, 70, 35), StrictVersion_37023, *[group_call_result_37028], **kwargs_37029)
            
            # Getting the type of 'self' (line 70)
            self_37031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'self')
            # Setting the type of the member 'gcc_version' of a type (line 70)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 16), self_37031, 'gcc_version', StrictVersion_call_result_37030)
            # SSA join for if statement (line 69)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_36992:
                # SSA join for if statement (line 62)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'self' (line 74)
        self_37032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 11), 'self')
        # Obtaining the member 'gcc_version' of a type (line 74)
        gcc_version_37033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 11), self_37032, 'gcc_version')
        str_37034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 31), 'str', '2.91.57')
        # Applying the binary operator '<=' (line 74)
        result_le_37035 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 11), '<=', gcc_version_37033, str_37034)
        
        # Testing the type of an if condition (line 74)
        if_condition_37036 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 8), result_le_37035)
        # Assigning a type to the variable 'if_condition_37036' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'if_condition_37036', if_condition_37036)
        # SSA begins for if statement (line 74)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 75):
        
        # Assigning a Str to a Name (line 75):
        str_37037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 26), 'str', '--entry _DllMain@12')
        # Assigning a type to the variable 'entry_point' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'entry_point', str_37037)
        # SSA branch for the else part of an if statement (line 74)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 77):
        
        # Assigning a Str to a Name (line 77):
        str_37038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 26), 'str', '')
        # Assigning a type to the variable 'entry_point' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'entry_point', str_37038)
        # SSA join for if statement (line 74)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 79)
        self_37039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), 'self')
        # Obtaining the member 'linker_dll' of a type (line 79)
        linker_dll_37040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 11), self_37039, 'linker_dll')
        str_37041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 30), 'str', 'dllwrap')
        # Applying the binary operator '==' (line 79)
        result_eq_37042 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 11), '==', linker_dll_37040, str_37041)
        
        # Testing the type of an if condition (line 79)
        if_condition_37043 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 8), result_eq_37042)
        # Assigning a type to the variable 'if_condition_37043' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'if_condition_37043', if_condition_37043)
        # SSA begins for if statement (line 79)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Attribute (line 86):
        
        # Assigning a Str to a Attribute (line 86):
        str_37044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 26), 'str', 'dllwrap')
        # Getting the type of 'self' (line 86)
        self_37045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'self')
        # Setting the type of the member 'linker' of a type (line 86)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), self_37045, 'linker', str_37044)
        # SSA branch for the else part of an if statement (line 79)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 87)
        self_37046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'self')
        # Obtaining the member 'linker_dll' of a type (line 87)
        linker_dll_37047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 13), self_37046, 'linker_dll')
        str_37048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 32), 'str', 'gcc')
        # Applying the binary operator '==' (line 87)
        result_eq_37049 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 13), '==', linker_dll_37047, str_37048)
        
        # Testing the type of an if condition (line 87)
        if_condition_37050 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 13), result_eq_37049)
        # Assigning a type to the variable 'if_condition_37050' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'if_condition_37050', if_condition_37050)
        # SSA begins for if statement (line 87)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Attribute (line 88):
        
        # Assigning a Str to a Attribute (line 88):
        str_37051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 26), 'str', 'g++')
        # Getting the type of 'self' (line 88)
        self_37052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'self')
        # Setting the type of the member 'linker' of a type (line 88)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 12), self_37052, 'linker', str_37051)
        # SSA join for if statement (line 87)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 79)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to build_import_library(...): (line 93)
        # Processing the call keyword arguments (line 93)
        kwargs_37054 = {}
        # Getting the type of 'build_import_library' (line 93)
        build_import_library_37053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'build_import_library', False)
        # Calling build_import_library(args, kwargs) (line 93)
        build_import_library_call_result_37055 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), build_import_library_37053, *[], **kwargs_37054)
        
        
        # Assigning a Call to a Name (line 96):
        
        # Assigning a Call to a Name (line 96):
        
        # Call to build_msvcr_library(...): (line 96)
        # Processing the call keyword arguments (line 96)
        kwargs_37057 = {}
        # Getting the type of 'build_msvcr_library' (line 96)
        build_msvcr_library_37056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 24), 'build_msvcr_library', False)
        # Calling build_msvcr_library(args, kwargs) (line 96)
        build_msvcr_library_call_result_37058 = invoke(stypy.reporting.localization.Localization(__file__, 96, 24), build_msvcr_library_37056, *[], **kwargs_37057)
        
        # Assigning a type to the variable 'msvcr_success' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'msvcr_success', build_msvcr_library_call_result_37058)
        
        # Assigning a Call to a Name (line 97):
        
        # Assigning a Call to a Name (line 97):
        
        # Call to build_msvcr_library(...): (line 97)
        # Processing the call keyword arguments (line 97)
        # Getting the type of 'True' (line 97)
        True_37060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 54), 'True', False)
        keyword_37061 = True_37060
        kwargs_37062 = {'debug': keyword_37061}
        # Getting the type of 'build_msvcr_library' (line 97)
        build_msvcr_library_37059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 28), 'build_msvcr_library', False)
        # Calling build_msvcr_library(args, kwargs) (line 97)
        build_msvcr_library_call_result_37063 = invoke(stypy.reporting.localization.Localization(__file__, 97, 28), build_msvcr_library_37059, *[], **kwargs_37062)
        
        # Assigning a type to the variable 'msvcr_dbg_success' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'msvcr_dbg_success', build_msvcr_library_call_result_37063)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'msvcr_success' (line 98)
        msvcr_success_37064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 11), 'msvcr_success')
        # Getting the type of 'msvcr_dbg_success' (line 98)
        msvcr_dbg_success_37065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 28), 'msvcr_dbg_success')
        # Applying the binary operator 'or' (line 98)
        result_or_keyword_37066 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 11), 'or', msvcr_success_37064, msvcr_dbg_success_37065)
        
        # Testing the type of an if condition (line 98)
        if_condition_37067 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 8), result_or_keyword_37066)
        # Assigning a type to the variable 'if_condition_37067' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'if_condition_37067', if_condition_37067)
        # SSA begins for if statement (line 98)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to define_macro(...): (line 100)
        # Processing the call arguments (line 100)
        str_37070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 30), 'str', 'NPY_MINGW_USE_CUSTOM_MSVCR')
        # Processing the call keyword arguments (line 100)
        kwargs_37071 = {}
        # Getting the type of 'self' (line 100)
        self_37068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'self', False)
        # Obtaining the member 'define_macro' of a type (line 100)
        define_macro_37069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), self_37068, 'define_macro')
        # Calling define_macro(args, kwargs) (line 100)
        define_macro_call_result_37072 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), define_macro_37069, *[str_37070], **kwargs_37071)
        
        # SSA join for if statement (line 98)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 103):
        
        # Assigning a BinOp to a Name (line 103):
        str_37073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 24), 'str', '0x%03i0')
        
        # Call to int(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Call to lstrip(...): (line 103)
        # Processing the call arguments (line 103)
        str_37079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 70), 'str', 'msvcr')
        # Processing the call keyword arguments (line 103)
        kwargs_37080 = {}
        
        # Call to msvc_runtime_library(...): (line 103)
        # Processing the call keyword arguments (line 103)
        kwargs_37076 = {}
        # Getting the type of 'msvc_runtime_library' (line 103)
        msvc_runtime_library_37075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 40), 'msvc_runtime_library', False)
        # Calling msvc_runtime_library(args, kwargs) (line 103)
        msvc_runtime_library_call_result_37077 = invoke(stypy.reporting.localization.Localization(__file__, 103, 40), msvc_runtime_library_37075, *[], **kwargs_37076)
        
        # Obtaining the member 'lstrip' of a type (line 103)
        lstrip_37078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 40), msvc_runtime_library_call_result_37077, 'lstrip')
        # Calling lstrip(args, kwargs) (line 103)
        lstrip_call_result_37081 = invoke(stypy.reporting.localization.Localization(__file__, 103, 40), lstrip_37078, *[str_37079], **kwargs_37080)
        
        # Processing the call keyword arguments (line 103)
        kwargs_37082 = {}
        # Getting the type of 'int' (line 103)
        int_37074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 36), 'int', False)
        # Calling int(args, kwargs) (line 103)
        int_call_result_37083 = invoke(stypy.reporting.localization.Localization(__file__, 103, 36), int_37074, *[lstrip_call_result_37081], **kwargs_37082)
        
        # Applying the binary operator '%' (line 103)
        result_mod_37084 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 24), '%', str_37073, int_call_result_37083)
        
        # Assigning a type to the variable 'msvcr_version' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'msvcr_version', result_mod_37084)
        
        # Call to define_macro(...): (line 104)
        # Processing the call arguments (line 104)
        str_37087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 26), 'str', '__MSVCRT_VERSION__')
        # Getting the type of 'msvcr_version' (line 104)
        msvcr_version_37088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 48), 'msvcr_version', False)
        # Processing the call keyword arguments (line 104)
        kwargs_37089 = {}
        # Getting the type of 'self' (line 104)
        self_37085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'self', False)
        # Obtaining the member 'define_macro' of a type (line 104)
        define_macro_37086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), self_37085, 'define_macro')
        # Calling define_macro(args, kwargs) (line 104)
        define_macro_call_result_37090 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), define_macro_37086, *[str_37087, msvcr_version_37088], **kwargs_37089)
        
        
        
        
        # Call to get_build_architecture(...): (line 110)
        # Processing the call keyword arguments (line 110)
        kwargs_37092 = {}
        # Getting the type of 'get_build_architecture' (line 110)
        get_build_architecture_37091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'get_build_architecture', False)
        # Calling get_build_architecture(args, kwargs) (line 110)
        get_build_architecture_call_result_37093 = invoke(stypy.reporting.localization.Localization(__file__, 110, 11), get_build_architecture_37091, *[], **kwargs_37092)
        
        str_37094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 39), 'str', 'AMD64')
        # Applying the binary operator '==' (line 110)
        result_eq_37095 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 11), '==', get_build_architecture_call_result_37093, str_37094)
        
        # Testing the type of an if condition (line 110)
        if_condition_37096 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 8), result_eq_37095)
        # Assigning a type to the variable 'if_condition_37096' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'if_condition_37096', if_condition_37096)
        # SSA begins for if statement (line 110)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'self' (line 111)
        self_37097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'self')
        # Obtaining the member 'gcc_version' of a type (line 111)
        gcc_version_37098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 15), self_37097, 'gcc_version')
        str_37099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 34), 'str', '4.0')
        # Applying the binary operator '<' (line 111)
        result_lt_37100 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 15), '<', gcc_version_37098, str_37099)
        
        # Testing the type of an if condition (line 111)
        if_condition_37101 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 12), result_lt_37100)
        # Assigning a type to the variable 'if_condition_37101' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'if_condition_37101', if_condition_37101)
        # SSA begins for if statement (line 111)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_executables(...): (line 112)
        # Processing the call keyword arguments (line 112)
        str_37104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 29), 'str', 'gcc -g -DDEBUG -DMS_WIN64 -mno-cygwin -O0 -Wall')
        keyword_37105 = str_37104
        str_37106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 32), 'str', 'gcc -g -DDEBUG -DMS_WIN64 -mno-cygwin -O0 -Wall -Wstrict-prototypes')
        keyword_37107 = str_37106
        str_37108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 31), 'str', 'gcc -g -mno-cygwin')
        keyword_37109 = str_37108
        str_37110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 30), 'str', 'gcc -g -mno-cygwin -shared')
        keyword_37111 = str_37110
        kwargs_37112 = {'linker_exe': keyword_37109, 'compiler_so': keyword_37107, 'linker_so': keyword_37111, 'compiler': keyword_37105}
        # Getting the type of 'self' (line 112)
        self_37102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'self', False)
        # Obtaining the member 'set_executables' of a type (line 112)
        set_executables_37103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 16), self_37102, 'set_executables')
        # Calling set_executables(args, kwargs) (line 112)
        set_executables_call_result_37113 = invoke(stypy.reporting.localization.Localization(__file__, 112, 16), set_executables_37103, *[], **kwargs_37112)
        
        # SSA branch for the else part of an if statement (line 111)
        module_type_store.open_ssa_branch('else')
        
        # Call to set_executables(...): (line 120)
        # Processing the call keyword arguments (line 120)
        str_37116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 29), 'str', 'gcc -g -DDEBUG -DMS_WIN64 -O0 -Wall')
        keyword_37117 = str_37116
        str_37118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 32), 'str', 'gcc -g -DDEBUG -DMS_WIN64 -O0 -Wall -Wstrict-prototypes')
        keyword_37119 = str_37118
        str_37120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 31), 'str', 'gcc -g')
        keyword_37121 = str_37120
        str_37122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 30), 'str', 'gcc -g -shared')
        keyword_37123 = str_37122
        kwargs_37124 = {'linker_exe': keyword_37121, 'compiler_so': keyword_37119, 'linker_so': keyword_37123, 'compiler': keyword_37117}
        # Getting the type of 'self' (line 120)
        self_37114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'self', False)
        # Obtaining the member 'set_executables' of a type (line 120)
        set_executables_37115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 16), self_37114, 'set_executables')
        # Calling set_executables(args, kwargs) (line 120)
        set_executables_call_result_37125 = invoke(stypy.reporting.localization.Localization(__file__, 120, 16), set_executables_37115, *[], **kwargs_37124)
        
        # SSA join for if statement (line 111)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 110)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 126)
        self_37126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 15), 'self')
        # Obtaining the member 'gcc_version' of a type (line 126)
        gcc_version_37127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 15), self_37126, 'gcc_version')
        str_37128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 35), 'str', '3.0.0')
        # Applying the binary operator '<=' (line 126)
        result_le_37129 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 15), '<=', gcc_version_37127, str_37128)
        
        # Testing the type of an if condition (line 126)
        if_condition_37130 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 126, 12), result_le_37129)
        # Assigning a type to the variable 'if_condition_37130' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'if_condition_37130', if_condition_37130)
        # SSA begins for if statement (line 126)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_executables(...): (line 127)
        # Processing the call keyword arguments (line 127)
        str_37133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 29), 'str', 'gcc -mno-cygwin -O2 -w')
        keyword_37134 = str_37133
        str_37135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 32), 'str', 'gcc -mno-cygwin -mdll -O2 -w -Wstrict-prototypes')
        keyword_37136 = str_37135
        str_37137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 31), 'str', 'g++ -mno-cygwin')
        keyword_37138 = str_37137
        str_37139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 30), 'str', '%s -mno-cygwin -mdll -static %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 133)
        tuple_37140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 133)
        # Adding element type (line 133)
        # Getting the type of 'self' (line 133)
        self_37141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 31), 'self', False)
        # Obtaining the member 'linker' of a type (line 133)
        linker_37142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 31), self_37141, 'linker')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 31), tuple_37140, linker_37142)
        # Adding element type (line 133)
        # Getting the type of 'entry_point' (line 133)
        entry_point_37143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 44), 'entry_point', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 31), tuple_37140, entry_point_37143)
        
        # Applying the binary operator '%' (line 132)
        result_mod_37144 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 30), '%', str_37139, tuple_37140)
        
        keyword_37145 = result_mod_37144
        kwargs_37146 = {'linker_exe': keyword_37138, 'compiler_so': keyword_37136, 'linker_so': keyword_37145, 'compiler': keyword_37134}
        # Getting the type of 'self' (line 127)
        self_37131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'self', False)
        # Obtaining the member 'set_executables' of a type (line 127)
        set_executables_37132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 16), self_37131, 'set_executables')
        # Calling set_executables(args, kwargs) (line 127)
        set_executables_call_result_37147 = invoke(stypy.reporting.localization.Localization(__file__, 127, 16), set_executables_37132, *[], **kwargs_37146)
        
        # SSA branch for the else part of an if statement (line 126)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 134)
        self_37148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 17), 'self')
        # Obtaining the member 'gcc_version' of a type (line 134)
        gcc_version_37149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 17), self_37148, 'gcc_version')
        str_37150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 36), 'str', '4.0')
        # Applying the binary operator '<' (line 134)
        result_lt_37151 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 17), '<', gcc_version_37149, str_37150)
        
        # Testing the type of an if condition (line 134)
        if_condition_37152 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 17), result_lt_37151)
        # Assigning a type to the variable 'if_condition_37152' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 17), 'if_condition_37152', if_condition_37152)
        # SSA begins for if statement (line 134)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_executables(...): (line 135)
        # Processing the call keyword arguments (line 135)
        str_37155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 29), 'str', 'gcc -mno-cygwin -O2 -Wall')
        keyword_37156 = str_37155
        str_37157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 32), 'str', 'gcc -mno-cygwin -O2 -Wall -Wstrict-prototypes')
        keyword_37158 = str_37157
        str_37159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 31), 'str', 'g++ -mno-cygwin')
        keyword_37160 = str_37159
        str_37161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 30), 'str', 'g++ -mno-cygwin -shared')
        keyword_37162 = str_37161
        kwargs_37163 = {'linker_exe': keyword_37160, 'compiler_so': keyword_37158, 'linker_so': keyword_37162, 'compiler': keyword_37156}
        # Getting the type of 'self' (line 135)
        self_37153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'self', False)
        # Obtaining the member 'set_executables' of a type (line 135)
        set_executables_37154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 16), self_37153, 'set_executables')
        # Calling set_executables(args, kwargs) (line 135)
        set_executables_call_result_37164 = invoke(stypy.reporting.localization.Localization(__file__, 135, 16), set_executables_37154, *[], **kwargs_37163)
        
        # SSA branch for the else part of an if statement (line 134)
        module_type_store.open_ssa_branch('else')
        
        # Call to set_executables(...): (line 143)
        # Processing the call keyword arguments (line 143)
        str_37167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 46), 'str', 'gcc -O2 -Wall')
        keyword_37168 = str_37167
        str_37169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 49), 'str', 'gcc -O2 -Wall -Wstrict-prototypes')
        keyword_37170 = str_37169
        str_37171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 48), 'str', 'g++ ')
        keyword_37172 = str_37171
        str_37173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 47), 'str', 'g++ -shared')
        keyword_37174 = str_37173
        kwargs_37175 = {'linker_exe': keyword_37172, 'compiler_so': keyword_37170, 'linker_so': keyword_37174, 'compiler': keyword_37168}
        # Getting the type of 'self' (line 143)
        self_37165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'self', False)
        # Obtaining the member 'set_executables' of a type (line 143)
        set_executables_37166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 16), self_37165, 'set_executables')
        # Calling set_executables(args, kwargs) (line 143)
        set_executables_call_result_37176 = invoke(stypy.reporting.localization.Localization(__file__, 143, 16), set_executables_37166, *[], **kwargs_37175)
        
        # SSA join for if statement (line 134)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 126)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 110)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Attribute (line 149):
        
        # Assigning a List to a Attribute (line 149):
        
        # Obtaining an instance of the builtin type 'list' (line 149)
        list_37177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 149)
        # Adding element type (line 149)
        str_37178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 29), 'str', 'g++')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 28), list_37177, str_37178)
        
        # Getting the type of 'self' (line 149)
        self_37179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'self')
        # Setting the type of the member 'compiler_cxx' of a type (line 149)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), self_37179, 'compiler_cxx', list_37177)
        # Assigning a type to the variable 'stypy_return_type' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def link(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 169)
        None_37180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 30), 'None')
        int_37181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 19), 'int')
        # Getting the type of 'None' (line 171)
        None_37182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 27), 'None')
        # Getting the type of 'None' (line 172)
        None_37183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 28), 'None')
        # Getting the type of 'None' (line 173)
        None_37184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 24), 'None')
        # Getting the type of 'None' (line 174)
        None_37185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 25), 'None')
        defaults = [None_37180, int_37181, None_37182, None_37183, None_37184, None_37185]
        # Create a new context for function 'link'
        module_type_store = module_type_store.open_function_context('link', 161, 4, False)
        # Assigning a type to the variable 'self' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Mingw32CCompiler.link.__dict__.__setitem__('stypy_localization', localization)
        Mingw32CCompiler.link.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Mingw32CCompiler.link.__dict__.__setitem__('stypy_type_store', module_type_store)
        Mingw32CCompiler.link.__dict__.__setitem__('stypy_function_name', 'Mingw32CCompiler.link')
        Mingw32CCompiler.link.__dict__.__setitem__('stypy_param_names_list', ['target_desc', 'objects', 'output_filename', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'])
        Mingw32CCompiler.link.__dict__.__setitem__('stypy_varargs_param_name', None)
        Mingw32CCompiler.link.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Mingw32CCompiler.link.__dict__.__setitem__('stypy_call_defaults', defaults)
        Mingw32CCompiler.link.__dict__.__setitem__('stypy_call_varargs', varargs)
        Mingw32CCompiler.link.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Mingw32CCompiler.link.__dict__.__setitem__('stypy_declared_arg_number', 14)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Mingw32CCompiler.link', ['target_desc', 'objects', 'output_filename', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 177):
        
        # Assigning a Call to a Name (line 177):
        
        # Call to msvc_runtime_library(...): (line 177)
        # Processing the call keyword arguments (line 177)
        kwargs_37187 = {}
        # Getting the type of 'msvc_runtime_library' (line 177)
        msvc_runtime_library_37186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 26), 'msvc_runtime_library', False)
        # Calling msvc_runtime_library(args, kwargs) (line 177)
        msvc_runtime_library_call_result_37188 = invoke(stypy.reporting.localization.Localization(__file__, 177, 26), msvc_runtime_library_37186, *[], **kwargs_37187)
        
        # Assigning a type to the variable 'runtime_library' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'runtime_library', msvc_runtime_library_call_result_37188)
        
        # Getting the type of 'runtime_library' (line 178)
        runtime_library_37189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 11), 'runtime_library')
        # Testing the type of an if condition (line 178)
        if_condition_37190 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 178, 8), runtime_library_37189)
        # Assigning a type to the variable 'if_condition_37190' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'if_condition_37190', if_condition_37190)
        # SSA begins for if statement (line 178)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'libraries' (line 179)
        libraries_37191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 19), 'libraries')
        # Applying the 'not' unary operator (line 179)
        result_not__37192 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 15), 'not', libraries_37191)
        
        # Testing the type of an if condition (line 179)
        if_condition_37193 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 179, 12), result_not__37192)
        # Assigning a type to the variable 'if_condition_37193' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'if_condition_37193', if_condition_37193)
        # SSA begins for if statement (line 179)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 180):
        
        # Assigning a List to a Name (line 180):
        
        # Obtaining an instance of the builtin type 'list' (line 180)
        list_37194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 180)
        
        # Assigning a type to the variable 'libraries' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 16), 'libraries', list_37194)
        # SSA join for if statement (line 179)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'runtime_library' (line 181)
        runtime_library_37197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 29), 'runtime_library', False)
        # Processing the call keyword arguments (line 181)
        kwargs_37198 = {}
        # Getting the type of 'libraries' (line 181)
        libraries_37195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'libraries', False)
        # Obtaining the member 'append' of a type (line 181)
        append_37196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 12), libraries_37195, 'append')
        # Calling append(args, kwargs) (line 181)
        append_call_result_37199 = invoke(stypy.reporting.localization.Localization(__file__, 181, 12), append_37196, *[runtime_library_37197], **kwargs_37198)
        
        # SSA join for if statement (line 178)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Tuple to a Name (line 182):
        
        # Assigning a Tuple to a Name (line 182):
        
        # Obtaining an instance of the builtin type 'tuple' (line 182)
        tuple_37200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 182)
        # Adding element type (line 182)
        # Getting the type of 'self' (line 182)
        self_37201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 16), 'self')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 16), tuple_37200, self_37201)
        # Adding element type (line 182)
        # Getting the type of 'target_desc' (line 183)
        target_desc_37202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 16), 'target_desc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 16), tuple_37200, target_desc_37202)
        # Adding element type (line 182)
        # Getting the type of 'objects' (line 184)
        objects_37203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'objects')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 16), tuple_37200, objects_37203)
        # Adding element type (line 182)
        # Getting the type of 'output_filename' (line 185)
        output_filename_37204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 16), 'output_filename')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 16), tuple_37200, output_filename_37204)
        # Adding element type (line 182)
        # Getting the type of 'output_dir' (line 186)
        output_dir_37205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 16), 'output_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 16), tuple_37200, output_dir_37205)
        # Adding element type (line 182)
        # Getting the type of 'libraries' (line 187)
        libraries_37206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 16), 'libraries')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 16), tuple_37200, libraries_37206)
        # Adding element type (line 182)
        # Getting the type of 'library_dirs' (line 188)
        library_dirs_37207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 16), 'library_dirs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 16), tuple_37200, library_dirs_37207)
        # Adding element type (line 182)
        # Getting the type of 'runtime_library_dirs' (line 189)
        runtime_library_dirs_37208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 16), 'runtime_library_dirs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 16), tuple_37200, runtime_library_dirs_37208)
        # Adding element type (line 182)
        # Getting the type of 'None' (line 190)
        None_37209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 16), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 16), tuple_37200, None_37209)
        # Adding element type (line 182)
        # Getting the type of 'debug' (line 191)
        debug_37210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 16), 'debug')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 16), tuple_37200, debug_37210)
        # Adding element type (line 182)
        # Getting the type of 'extra_preargs' (line 192)
        extra_preargs_37211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 16), 'extra_preargs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 16), tuple_37200, extra_preargs_37211)
        # Adding element type (line 182)
        # Getting the type of 'extra_postargs' (line 193)
        extra_postargs_37212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 16), 'extra_postargs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 16), tuple_37200, extra_postargs_37212)
        # Adding element type (line 182)
        # Getting the type of 'build_temp' (line 194)
        build_temp_37213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 16), 'build_temp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 16), tuple_37200, build_temp_37213)
        # Adding element type (line 182)
        # Getting the type of 'target_lang' (line 195)
        target_lang_37214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 16), 'target_lang')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 16), tuple_37200, target_lang_37214)
        
        # Assigning a type to the variable 'args' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'args', tuple_37200)
        
        
        # Getting the type of 'self' (line 196)
        self_37215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 11), 'self')
        # Obtaining the member 'gcc_version' of a type (line 196)
        gcc_version_37216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 11), self_37215, 'gcc_version')
        str_37217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 30), 'str', '3.0.0')
        # Applying the binary operator '<' (line 196)
        result_lt_37218 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 11), '<', gcc_version_37216, str_37217)
        
        # Testing the type of an if condition (line 196)
        if_condition_37219 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 196, 8), result_lt_37218)
        # Assigning a type to the variable 'if_condition_37219' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'if_condition_37219', if_condition_37219)
        # SSA begins for if statement (line 196)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 197):
        
        # Assigning a Attribute to a Name (line 197):
        # Getting the type of 'distutils' (line 197)
        distutils_37220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 19), 'distutils')
        # Obtaining the member 'cygwinccompiler' of a type (line 197)
        cygwinccompiler_37221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 19), distutils_37220, 'cygwinccompiler')
        # Obtaining the member 'CygwinCCompiler' of a type (line 197)
        CygwinCCompiler_37222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 19), cygwinccompiler_37221, 'CygwinCCompiler')
        # Obtaining the member 'link' of a type (line 197)
        link_37223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 19), CygwinCCompiler_37222, 'link')
        # Assigning a type to the variable 'func' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'func', link_37223)
        # SSA branch for the else part of an if statement (line 196)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 199):
        
        # Assigning a Attribute to a Name (line 199):
        # Getting the type of 'UnixCCompiler' (line 199)
        UnixCCompiler_37224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 19), 'UnixCCompiler')
        # Obtaining the member 'link' of a type (line 199)
        link_37225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 19), UnixCCompiler_37224, 'link')
        # Assigning a type to the variable 'func' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'func', link_37225)
        # SSA join for if statement (line 196)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to func(...): (line 200)
        
        # Obtaining the type of the subscript
        # Getting the type of 'func' (line 200)
        func_37227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 20), 'func', False)
        # Obtaining the member '__code__' of a type (line 200)
        code___37228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 20), func_37227, '__code__')
        # Obtaining the member 'co_argcount' of a type (line 200)
        co_argcount_37229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 20), code___37228, 'co_argcount')
        slice_37230 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 200, 14), None, co_argcount_37229, None)
        # Getting the type of 'args' (line 200)
        args_37231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 14), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 200)
        getitem___37232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 14), args_37231, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 200)
        subscript_call_result_37233 = invoke(stypy.reporting.localization.Localization(__file__, 200, 14), getitem___37232, slice_37230)
        
        # Processing the call keyword arguments (line 200)
        kwargs_37234 = {}
        # Getting the type of 'func' (line 200)
        func_37226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'func', False)
        # Calling func(args, kwargs) (line 200)
        func_call_result_37235 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), func_37226, *[subscript_call_result_37233], **kwargs_37234)
        
        # Assigning a type to the variable 'stypy_return_type' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of 'link(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'link' in the type store
        # Getting the type of 'stypy_return_type' (line 161)
        stypy_return_type_37236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37236)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'link'
        return stypy_return_type_37236


    @norecursion
    def object_filenames(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_37237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 36), 'int')
        str_37238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 37), 'str', '')
        defaults = [int_37237, str_37238]
        # Create a new context for function 'object_filenames'
        module_type_store = module_type_store.open_function_context('object_filenames', 203, 4, False)
        # Assigning a type to the variable 'self' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Mingw32CCompiler.object_filenames.__dict__.__setitem__('stypy_localization', localization)
        Mingw32CCompiler.object_filenames.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Mingw32CCompiler.object_filenames.__dict__.__setitem__('stypy_type_store', module_type_store)
        Mingw32CCompiler.object_filenames.__dict__.__setitem__('stypy_function_name', 'Mingw32CCompiler.object_filenames')
        Mingw32CCompiler.object_filenames.__dict__.__setitem__('stypy_param_names_list', ['source_filenames', 'strip_dir', 'output_dir'])
        Mingw32CCompiler.object_filenames.__dict__.__setitem__('stypy_varargs_param_name', None)
        Mingw32CCompiler.object_filenames.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Mingw32CCompiler.object_filenames.__dict__.__setitem__('stypy_call_defaults', defaults)
        Mingw32CCompiler.object_filenames.__dict__.__setitem__('stypy_call_varargs', varargs)
        Mingw32CCompiler.object_filenames.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Mingw32CCompiler.object_filenames.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Mingw32CCompiler.object_filenames', ['source_filenames', 'strip_dir', 'output_dir'], None, None, defaults, varargs, kwargs)

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

        
        # Type idiom detected: calculating its left and rigth part (line 207)
        # Getting the type of 'output_dir' (line 207)
        output_dir_37239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 11), 'output_dir')
        # Getting the type of 'None' (line 207)
        None_37240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 25), 'None')
        
        (may_be_37241, more_types_in_union_37242) = may_be_none(output_dir_37239, None_37240)

        if may_be_37241:

            if more_types_in_union_37242:
                # Runtime conditional SSA (line 207)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Str to a Name (line 207):
            
            # Assigning a Str to a Name (line 207):
            str_37243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 44), 'str', '')
            # Assigning a type to the variable 'output_dir' (line 207)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 31), 'output_dir', str_37243)

            if more_types_in_union_37242:
                # SSA join for if statement (line 207)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a List to a Name (line 208):
        
        # Assigning a List to a Name (line 208):
        
        # Obtaining an instance of the builtin type 'list' (line 208)
        list_37244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 208)
        
        # Assigning a type to the variable 'obj_names' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'obj_names', list_37244)
        
        # Getting the type of 'source_filenames' (line 209)
        source_filenames_37245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 24), 'source_filenames')
        # Testing the type of a for loop iterable (line 209)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 209, 8), source_filenames_37245)
        # Getting the type of the for loop variable (line 209)
        for_loop_var_37246 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 209, 8), source_filenames_37245)
        # Assigning a type to the variable 'src_name' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'src_name', for_loop_var_37246)
        # SSA begins for a for statement (line 209)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 211):
        
        # Assigning a Call to a Name:
        
        # Call to splitext(...): (line 211)
        # Processing the call arguments (line 211)
        
        # Call to normcase(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'src_name' (line 211)
        src_name_37253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 61), 'src_name', False)
        # Processing the call keyword arguments (line 211)
        kwargs_37254 = {}
        # Getting the type of 'os' (line 211)
        os_37250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 44), 'os', False)
        # Obtaining the member 'path' of a type (line 211)
        path_37251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 44), os_37250, 'path')
        # Obtaining the member 'normcase' of a type (line 211)
        normcase_37252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 44), path_37251, 'normcase')
        # Calling normcase(args, kwargs) (line 211)
        normcase_call_result_37255 = invoke(stypy.reporting.localization.Localization(__file__, 211, 44), normcase_37252, *[src_name_37253], **kwargs_37254)
        
        # Processing the call keyword arguments (line 211)
        kwargs_37256 = {}
        # Getting the type of 'os' (line 211)
        os_37247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 211)
        path_37248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 26), os_37247, 'path')
        # Obtaining the member 'splitext' of a type (line 211)
        splitext_37249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 26), path_37248, 'splitext')
        # Calling splitext(args, kwargs) (line 211)
        splitext_call_result_37257 = invoke(stypy.reporting.localization.Localization(__file__, 211, 26), splitext_37249, *[normcase_call_result_37255], **kwargs_37256)
        
        # Assigning a type to the variable 'call_assignment_36916' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'call_assignment_36916', splitext_call_result_37257)
        
        # Assigning a Call to a Name (line 211):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_37260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 12), 'int')
        # Processing the call keyword arguments
        kwargs_37261 = {}
        # Getting the type of 'call_assignment_36916' (line 211)
        call_assignment_36916_37258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'call_assignment_36916', False)
        # Obtaining the member '__getitem__' of a type (line 211)
        getitem___37259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 12), call_assignment_36916_37258, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_37262 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___37259, *[int_37260], **kwargs_37261)
        
        # Assigning a type to the variable 'call_assignment_36917' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'call_assignment_36917', getitem___call_result_37262)
        
        # Assigning a Name to a Name (line 211):
        # Getting the type of 'call_assignment_36917' (line 211)
        call_assignment_36917_37263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'call_assignment_36917')
        # Assigning a type to the variable 'base' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 13), 'base', call_assignment_36917_37263)
        
        # Assigning a Call to a Name (line 211):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_37266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 12), 'int')
        # Processing the call keyword arguments
        kwargs_37267 = {}
        # Getting the type of 'call_assignment_36916' (line 211)
        call_assignment_36916_37264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'call_assignment_36916', False)
        # Obtaining the member '__getitem__' of a type (line 211)
        getitem___37265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 12), call_assignment_36916_37264, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_37268 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___37265, *[int_37266], **kwargs_37267)
        
        # Assigning a type to the variable 'call_assignment_36918' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'call_assignment_36918', getitem___call_result_37268)
        
        # Assigning a Name to a Name (line 211):
        # Getting the type of 'call_assignment_36918' (line 211)
        call_assignment_36918_37269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'call_assignment_36918')
        # Assigning a type to the variable 'ext' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 19), 'ext', call_assignment_36918_37269)
        
        # Assigning a Call to a Tuple (line 216):
        
        # Assigning a Call to a Name:
        
        # Call to splitdrive(...): (line 216)
        # Processing the call arguments (line 216)
        # Getting the type of 'base' (line 216)
        base_37273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 43), 'base', False)
        # Processing the call keyword arguments (line 216)
        kwargs_37274 = {}
        # Getting the type of 'os' (line 216)
        os_37270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 216)
        path_37271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 24), os_37270, 'path')
        # Obtaining the member 'splitdrive' of a type (line 216)
        splitdrive_37272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 24), path_37271, 'splitdrive')
        # Calling splitdrive(args, kwargs) (line 216)
        splitdrive_call_result_37275 = invoke(stypy.reporting.localization.Localization(__file__, 216, 24), splitdrive_37272, *[base_37273], **kwargs_37274)
        
        # Assigning a type to the variable 'call_assignment_36919' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'call_assignment_36919', splitdrive_call_result_37275)
        
        # Assigning a Call to a Name (line 216):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_37278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 12), 'int')
        # Processing the call keyword arguments
        kwargs_37279 = {}
        # Getting the type of 'call_assignment_36919' (line 216)
        call_assignment_36919_37276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'call_assignment_36919', False)
        # Obtaining the member '__getitem__' of a type (line 216)
        getitem___37277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 12), call_assignment_36919_37276, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_37280 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___37277, *[int_37278], **kwargs_37279)
        
        # Assigning a type to the variable 'call_assignment_36920' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'call_assignment_36920', getitem___call_result_37280)
        
        # Assigning a Name to a Name (line 216):
        # Getting the type of 'call_assignment_36920' (line 216)
        call_assignment_36920_37281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'call_assignment_36920')
        # Assigning a type to the variable 'drv' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'drv', call_assignment_36920_37281)
        
        # Assigning a Call to a Name (line 216):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_37284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 12), 'int')
        # Processing the call keyword arguments
        kwargs_37285 = {}
        # Getting the type of 'call_assignment_36919' (line 216)
        call_assignment_36919_37282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'call_assignment_36919', False)
        # Obtaining the member '__getitem__' of a type (line 216)
        getitem___37283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 12), call_assignment_36919_37282, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_37286 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___37283, *[int_37284], **kwargs_37285)
        
        # Assigning a type to the variable 'call_assignment_36921' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'call_assignment_36921', getitem___call_result_37286)
        
        # Assigning a Name to a Name (line 216):
        # Getting the type of 'call_assignment_36921' (line 216)
        call_assignment_36921_37287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'call_assignment_36921')
        # Assigning a type to the variable 'base' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 17), 'base', call_assignment_36921_37287)
        
        # Getting the type of 'drv' (line 217)
        drv_37288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'drv')
        # Testing the type of an if condition (line 217)
        if_condition_37289 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 217, 12), drv_37288)
        # Assigning a type to the variable 'if_condition_37289' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'if_condition_37289', if_condition_37289)
        # SSA begins for if statement (line 217)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 218):
        
        # Assigning a Subscript to a Name (line 218):
        
        # Obtaining the type of the subscript
        int_37290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 28), 'int')
        slice_37291 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 218, 23), int_37290, None, None)
        # Getting the type of 'base' (line 218)
        base_37292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 23), 'base')
        # Obtaining the member '__getitem__' of a type (line 218)
        getitem___37293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 23), base_37292, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 218)
        subscript_call_result_37294 = invoke(stypy.reporting.localization.Localization(__file__, 218, 23), getitem___37293, slice_37291)
        
        # Assigning a type to the variable 'base' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'base', subscript_call_result_37294)
        # SSA join for if statement (line 217)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'ext' (line 220)
        ext_37295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 15), 'ext')
        # Getting the type of 'self' (line 220)
        self_37296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 27), 'self')
        # Obtaining the member 'src_extensions' of a type (line 220)
        src_extensions_37297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 27), self_37296, 'src_extensions')
        
        # Obtaining an instance of the builtin type 'list' (line 220)
        list_37298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 220)
        # Adding element type (line 220)
        str_37299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 50), 'str', '.rc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 49), list_37298, str_37299)
        # Adding element type (line 220)
        str_37300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 57), 'str', '.res')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 49), list_37298, str_37300)
        
        # Applying the binary operator '+' (line 220)
        result_add_37301 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 27), '+', src_extensions_37297, list_37298)
        
        # Applying the binary operator 'notin' (line 220)
        result_contains_37302 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 15), 'notin', ext_37295, result_add_37301)
        
        # Testing the type of an if condition (line 220)
        if_condition_37303 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 220, 12), result_contains_37302)
        # Assigning a type to the variable 'if_condition_37303' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'if_condition_37303', if_condition_37303)
        # SSA begins for if statement (line 220)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to UnknownFileError(...): (line 221)
        # Processing the call arguments (line 221)
        str_37305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 22), 'str', "unknown file type '%s' (from '%s')")
        
        # Obtaining an instance of the builtin type 'tuple' (line 223)
        tuple_37306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 223)
        # Adding element type (line 223)
        # Getting the type of 'ext' (line 223)
        ext_37307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 23), 'ext', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 23), tuple_37306, ext_37307)
        # Adding element type (line 223)
        # Getting the type of 'src_name' (line 223)
        src_name_37308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 28), 'src_name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 23), tuple_37306, src_name_37308)
        
        # Applying the binary operator '%' (line 222)
        result_mod_37309 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 22), '%', str_37305, tuple_37306)
        
        # Processing the call keyword arguments (line 221)
        kwargs_37310 = {}
        # Getting the type of 'UnknownFileError' (line 221)
        UnknownFileError_37304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 22), 'UnknownFileError', False)
        # Calling UnknownFileError(args, kwargs) (line 221)
        UnknownFileError_call_result_37311 = invoke(stypy.reporting.localization.Localization(__file__, 221, 22), UnknownFileError_37304, *[result_mod_37309], **kwargs_37310)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 221, 16), UnknownFileError_call_result_37311, 'raise parameter', BaseException)
        # SSA join for if statement (line 220)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'strip_dir' (line 224)
        strip_dir_37312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 15), 'strip_dir')
        # Testing the type of an if condition (line 224)
        if_condition_37313 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 224, 12), strip_dir_37312)
        # Assigning a type to the variable 'if_condition_37313' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'if_condition_37313', if_condition_37313)
        # SSA begins for if statement (line 224)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 225):
        
        # Assigning a Call to a Name (line 225):
        
        # Call to basename(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of 'base' (line 225)
        base_37317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 41), 'base', False)
        # Processing the call keyword arguments (line 225)
        kwargs_37318 = {}
        # Getting the type of 'os' (line 225)
        os_37314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 225)
        path_37315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 23), os_37314, 'path')
        # Obtaining the member 'basename' of a type (line 225)
        basename_37316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 23), path_37315, 'basename')
        # Calling basename(args, kwargs) (line 225)
        basename_call_result_37319 = invoke(stypy.reporting.localization.Localization(__file__, 225, 23), basename_37316, *[base_37317], **kwargs_37318)
        
        # Assigning a type to the variable 'base' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 16), 'base', basename_call_result_37319)
        # SSA join for if statement (line 224)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'ext' (line 226)
        ext_37320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 15), 'ext')
        str_37321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 22), 'str', '.res')
        # Applying the binary operator '==' (line 226)
        result_eq_37322 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 15), '==', ext_37320, str_37321)
        
        
        # Getting the type of 'ext' (line 226)
        ext_37323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 32), 'ext')
        str_37324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 39), 'str', '.rc')
        # Applying the binary operator '==' (line 226)
        result_eq_37325 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 32), '==', ext_37323, str_37324)
        
        # Applying the binary operator 'or' (line 226)
        result_or_keyword_37326 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 15), 'or', result_eq_37322, result_eq_37325)
        
        # Testing the type of an if condition (line 226)
        if_condition_37327 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 12), result_or_keyword_37326)
        # Assigning a type to the variable 'if_condition_37327' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'if_condition_37327', if_condition_37327)
        # SSA begins for if statement (line 226)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 228)
        # Processing the call arguments (line 228)
        
        # Call to join(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'output_dir' (line 228)
        output_dir_37333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 48), 'output_dir', False)
        # Getting the type of 'base' (line 229)
        base_37334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 48), 'base', False)
        # Getting the type of 'ext' (line 229)
        ext_37335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 55), 'ext', False)
        # Applying the binary operator '+' (line 229)
        result_add_37336 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 48), '+', base_37334, ext_37335)
        
        # Getting the type of 'self' (line 229)
        self_37337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 61), 'self', False)
        # Obtaining the member 'obj_extension' of a type (line 229)
        obj_extension_37338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 61), self_37337, 'obj_extension')
        # Applying the binary operator '+' (line 229)
        result_add_37339 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 59), '+', result_add_37336, obj_extension_37338)
        
        # Processing the call keyword arguments (line 228)
        kwargs_37340 = {}
        # Getting the type of 'os' (line 228)
        os_37330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 34), 'os', False)
        # Obtaining the member 'path' of a type (line 228)
        path_37331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 34), os_37330, 'path')
        # Obtaining the member 'join' of a type (line 228)
        join_37332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 34), path_37331, 'join')
        # Calling join(args, kwargs) (line 228)
        join_call_result_37341 = invoke(stypy.reporting.localization.Localization(__file__, 228, 34), join_37332, *[output_dir_37333, result_add_37339], **kwargs_37340)
        
        # Processing the call keyword arguments (line 228)
        kwargs_37342 = {}
        # Getting the type of 'obj_names' (line 228)
        obj_names_37328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 16), 'obj_names', False)
        # Obtaining the member 'append' of a type (line 228)
        append_37329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 16), obj_names_37328, 'append')
        # Calling append(args, kwargs) (line 228)
        append_call_result_37343 = invoke(stypy.reporting.localization.Localization(__file__, 228, 16), append_37329, *[join_call_result_37341], **kwargs_37342)
        
        # SSA branch for the else part of an if statement (line 226)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 231)
        # Processing the call arguments (line 231)
        
        # Call to join(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'output_dir' (line 231)
        output_dir_37349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 48), 'output_dir', False)
        # Getting the type of 'base' (line 232)
        base_37350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 48), 'base', False)
        # Getting the type of 'self' (line 232)
        self_37351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 55), 'self', False)
        # Obtaining the member 'obj_extension' of a type (line 232)
        obj_extension_37352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 55), self_37351, 'obj_extension')
        # Applying the binary operator '+' (line 232)
        result_add_37353 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 48), '+', base_37350, obj_extension_37352)
        
        # Processing the call keyword arguments (line 231)
        kwargs_37354 = {}
        # Getting the type of 'os' (line 231)
        os_37346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 34), 'os', False)
        # Obtaining the member 'path' of a type (line 231)
        path_37347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 34), os_37346, 'path')
        # Obtaining the member 'join' of a type (line 231)
        join_37348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 34), path_37347, 'join')
        # Calling join(args, kwargs) (line 231)
        join_call_result_37355 = invoke(stypy.reporting.localization.Localization(__file__, 231, 34), join_37348, *[output_dir_37349, result_add_37353], **kwargs_37354)
        
        # Processing the call keyword arguments (line 231)
        kwargs_37356 = {}
        # Getting the type of 'obj_names' (line 231)
        obj_names_37344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 16), 'obj_names', False)
        # Obtaining the member 'append' of a type (line 231)
        append_37345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 16), obj_names_37344, 'append')
        # Calling append(args, kwargs) (line 231)
        append_call_result_37357 = invoke(stypy.reporting.localization.Localization(__file__, 231, 16), append_37345, *[join_call_result_37355], **kwargs_37356)
        
        # SSA join for if statement (line 226)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'obj_names' (line 233)
        obj_names_37358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 15), 'obj_names')
        # Assigning a type to the variable 'stypy_return_type' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'stypy_return_type', obj_names_37358)
        
        # ################# End of 'object_filenames(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'object_filenames' in the type store
        # Getting the type of 'stypy_return_type' (line 203)
        stypy_return_type_37359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37359)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'object_filenames'
        return stypy_return_type_37359


# Assigning a type to the variable 'Mingw32CCompiler' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'Mingw32CCompiler', Mingw32CCompiler)

# Assigning a Str to a Name (line 50):
str_37360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 20), 'str', 'mingw32')
# Getting the type of 'Mingw32CCompiler'
Mingw32CCompiler_37361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Mingw32CCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Mingw32CCompiler_37361, 'compiler_type', str_37360)

@norecursion
def find_python_dll(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'find_python_dll'
    module_type_store = module_type_store.open_function_context('find_python_dll', 238, 0, False)
    
    # Passed parameters checking function
    find_python_dll.stypy_localization = localization
    find_python_dll.stypy_type_of_self = None
    find_python_dll.stypy_type_store = module_type_store
    find_python_dll.stypy_function_name = 'find_python_dll'
    find_python_dll.stypy_param_names_list = []
    find_python_dll.stypy_varargs_param_name = None
    find_python_dll.stypy_kwargs_param_name = None
    find_python_dll.stypy_call_defaults = defaults
    find_python_dll.stypy_call_varargs = varargs
    find_python_dll.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'find_python_dll', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'find_python_dll', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'find_python_dll(...)' code ##################

    
    # Assigning a ListComp to a Tuple (line 239):
    
    # Assigning a Subscript to a Name (line 239):
    
    # Obtaining the type of the subscript
    int_37362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    int_37367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 57), 'int')
    slice_37368 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 239, 39), None, int_37367, None)
    # Getting the type of 'sys' (line 239)
    sys_37369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 39), 'sys')
    # Obtaining the member 'version_info' of a type (line 239)
    version_info_37370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 39), sys_37369, 'version_info')
    # Obtaining the member '__getitem__' of a type (line 239)
    getitem___37371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 39), version_info_37370, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 239)
    subscript_call_result_37372 = invoke(stypy.reporting.localization.Localization(__file__, 239, 39), getitem___37371, slice_37368)
    
    comprehension_37373 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 23), subscript_call_result_37372)
    # Assigning a type to the variable 'i' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 23), 'i', comprehension_37373)
    
    # Call to int(...): (line 239)
    # Processing the call arguments (line 239)
    # Getting the type of 'i' (line 239)
    i_37364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 27), 'i', False)
    # Processing the call keyword arguments (line 239)
    kwargs_37365 = {}
    # Getting the type of 'int' (line 239)
    int_37363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 23), 'int', False)
    # Calling int(args, kwargs) (line 239)
    int_call_result_37366 = invoke(stypy.reporting.localization.Localization(__file__, 239, 23), int_37363, *[i_37364], **kwargs_37365)
    
    list_37374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 23), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 23), list_37374, int_call_result_37366)
    # Obtaining the member '__getitem__' of a type (line 239)
    getitem___37375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 4), list_37374, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 239)
    subscript_call_result_37376 = invoke(stypy.reporting.localization.Localization(__file__, 239, 4), getitem___37375, int_37362)
    
    # Assigning a type to the variable 'tuple_var_assignment_36922' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'tuple_var_assignment_36922', subscript_call_result_37376)
    
    # Assigning a Subscript to a Name (line 239):
    
    # Obtaining the type of the subscript
    int_37377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    int_37382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 57), 'int')
    slice_37383 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 239, 39), None, int_37382, None)
    # Getting the type of 'sys' (line 239)
    sys_37384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 39), 'sys')
    # Obtaining the member 'version_info' of a type (line 239)
    version_info_37385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 39), sys_37384, 'version_info')
    # Obtaining the member '__getitem__' of a type (line 239)
    getitem___37386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 39), version_info_37385, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 239)
    subscript_call_result_37387 = invoke(stypy.reporting.localization.Localization(__file__, 239, 39), getitem___37386, slice_37383)
    
    comprehension_37388 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 23), subscript_call_result_37387)
    # Assigning a type to the variable 'i' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 23), 'i', comprehension_37388)
    
    # Call to int(...): (line 239)
    # Processing the call arguments (line 239)
    # Getting the type of 'i' (line 239)
    i_37379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 27), 'i', False)
    # Processing the call keyword arguments (line 239)
    kwargs_37380 = {}
    # Getting the type of 'int' (line 239)
    int_37378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 23), 'int', False)
    # Calling int(args, kwargs) (line 239)
    int_call_result_37381 = invoke(stypy.reporting.localization.Localization(__file__, 239, 23), int_37378, *[i_37379], **kwargs_37380)
    
    list_37389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 23), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 23), list_37389, int_call_result_37381)
    # Obtaining the member '__getitem__' of a type (line 239)
    getitem___37390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 4), list_37389, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 239)
    subscript_call_result_37391 = invoke(stypy.reporting.localization.Localization(__file__, 239, 4), getitem___37390, int_37377)
    
    # Assigning a type to the variable 'tuple_var_assignment_36923' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'tuple_var_assignment_36923', subscript_call_result_37391)
    
    # Assigning a Subscript to a Name (line 239):
    
    # Obtaining the type of the subscript
    int_37392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    int_37397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 57), 'int')
    slice_37398 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 239, 39), None, int_37397, None)
    # Getting the type of 'sys' (line 239)
    sys_37399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 39), 'sys')
    # Obtaining the member 'version_info' of a type (line 239)
    version_info_37400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 39), sys_37399, 'version_info')
    # Obtaining the member '__getitem__' of a type (line 239)
    getitem___37401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 39), version_info_37400, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 239)
    subscript_call_result_37402 = invoke(stypy.reporting.localization.Localization(__file__, 239, 39), getitem___37401, slice_37398)
    
    comprehension_37403 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 23), subscript_call_result_37402)
    # Assigning a type to the variable 'i' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 23), 'i', comprehension_37403)
    
    # Call to int(...): (line 239)
    # Processing the call arguments (line 239)
    # Getting the type of 'i' (line 239)
    i_37394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 27), 'i', False)
    # Processing the call keyword arguments (line 239)
    kwargs_37395 = {}
    # Getting the type of 'int' (line 239)
    int_37393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 23), 'int', False)
    # Calling int(args, kwargs) (line 239)
    int_call_result_37396 = invoke(stypy.reporting.localization.Localization(__file__, 239, 23), int_37393, *[i_37394], **kwargs_37395)
    
    list_37404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 23), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 23), list_37404, int_call_result_37396)
    # Obtaining the member '__getitem__' of a type (line 239)
    getitem___37405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 4), list_37404, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 239)
    subscript_call_result_37406 = invoke(stypy.reporting.localization.Localization(__file__, 239, 4), getitem___37405, int_37392)
    
    # Assigning a type to the variable 'tuple_var_assignment_36924' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'tuple_var_assignment_36924', subscript_call_result_37406)
    
    # Assigning a Name to a Name (line 239):
    # Getting the type of 'tuple_var_assignment_36922' (line 239)
    tuple_var_assignment_36922_37407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'tuple_var_assignment_36922')
    # Assigning a type to the variable 'maj' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'maj', tuple_var_assignment_36922_37407)
    
    # Assigning a Name to a Name (line 239):
    # Getting the type of 'tuple_var_assignment_36923' (line 239)
    tuple_var_assignment_36923_37408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'tuple_var_assignment_36923')
    # Assigning a type to the variable 'min' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 9), 'min', tuple_var_assignment_36923_37408)
    
    # Assigning a Name to a Name (line 239):
    # Getting the type of 'tuple_var_assignment_36924' (line 239)
    tuple_var_assignment_36924_37409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'tuple_var_assignment_36924')
    # Assigning a type to the variable 'micro' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 14), 'micro', tuple_var_assignment_36924_37409)
    
    # Assigning a BinOp to a Name (line 240):
    
    # Assigning a BinOp to a Name (line 240):
    str_37410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 14), 'str', 'python%d%d.dll')
    
    # Obtaining an instance of the builtin type 'tuple' (line 240)
    tuple_37411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 240)
    # Adding element type (line 240)
    # Getting the type of 'maj' (line 240)
    maj_37412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 34), 'maj')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 34), tuple_37411, maj_37412)
    # Adding element type (line 240)
    # Getting the type of 'min' (line 240)
    min_37413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 39), 'min')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 34), tuple_37411, min_37413)
    
    # Applying the binary operator '%' (line 240)
    result_mod_37414 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 14), '%', str_37410, tuple_37411)
    
    # Assigning a type to the variable 'dllname' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'dllname', result_mod_37414)
    
    # Call to print(...): (line 241)
    # Processing the call arguments (line 241)
    str_37416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 10), 'str', 'Looking for %s')
    # Getting the type of 'dllname' (line 241)
    dllname_37417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 29), 'dllname', False)
    # Applying the binary operator '%' (line 241)
    result_mod_37418 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 10), '%', str_37416, dllname_37417)
    
    # Processing the call keyword arguments (line 241)
    kwargs_37419 = {}
    # Getting the type of 'print' (line 241)
    print_37415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'print', False)
    # Calling print(args, kwargs) (line 241)
    print_call_result_37420 = invoke(stypy.reporting.localization.Localization(__file__, 241, 4), print_37415, *[result_mod_37418], **kwargs_37419)
    
    
    # Assigning a List to a Name (line 247):
    
    # Assigning a List to a Name (line 247):
    
    # Obtaining an instance of the builtin type 'list' (line 247)
    list_37421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 247)
    # Adding element type (line 247)
    # Getting the type of 'sys' (line 247)
    sys_37422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 16), 'sys')
    # Obtaining the member 'prefix' of a type (line 247)
    prefix_37423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 16), sys_37422, 'prefix')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 15), list_37421, prefix_37423)
    # Adding element type (line 247)
    
    # Call to join(...): (line 247)
    # Processing the call arguments (line 247)
    # Getting the type of 'sys' (line 247)
    sys_37427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 41), 'sys', False)
    # Obtaining the member 'prefix' of a type (line 247)
    prefix_37428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 41), sys_37427, 'prefix')
    str_37429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 53), 'str', 'lib')
    # Processing the call keyword arguments (line 247)
    kwargs_37430 = {}
    # Getting the type of 'os' (line 247)
    os_37424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 28), 'os', False)
    # Obtaining the member 'path' of a type (line 247)
    path_37425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 28), os_37424, 'path')
    # Obtaining the member 'join' of a type (line 247)
    join_37426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 28), path_37425, 'join')
    # Calling join(args, kwargs) (line 247)
    join_call_result_37431 = invoke(stypy.reporting.localization.Localization(__file__, 247, 28), join_37426, *[prefix_37428, str_37429], **kwargs_37430)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 15), list_37421, join_call_result_37431)
    
    # Assigning a type to the variable 'lib_dirs' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'lib_dirs', list_37421)
    
    
    # SSA begins for try-except statement (line 248)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to append(...): (line 249)
    # Processing the call arguments (line 249)
    
    # Call to join(...): (line 249)
    # Processing the call arguments (line 249)
    
    # Obtaining the type of the subscript
    str_37437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 48), 'str', 'SYSTEMROOT')
    # Getting the type of 'os' (line 249)
    os_37438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 37), 'os', False)
    # Obtaining the member 'environ' of a type (line 249)
    environ_37439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 37), os_37438, 'environ')
    # Obtaining the member '__getitem__' of a type (line 249)
    getitem___37440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 37), environ_37439, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 249)
    subscript_call_result_37441 = invoke(stypy.reporting.localization.Localization(__file__, 249, 37), getitem___37440, str_37437)
    
    str_37442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 63), 'str', 'system32')
    # Processing the call keyword arguments (line 249)
    kwargs_37443 = {}
    # Getting the type of 'os' (line 249)
    os_37434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 249)
    path_37435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 24), os_37434, 'path')
    # Obtaining the member 'join' of a type (line 249)
    join_37436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 24), path_37435, 'join')
    # Calling join(args, kwargs) (line 249)
    join_call_result_37444 = invoke(stypy.reporting.localization.Localization(__file__, 249, 24), join_37436, *[subscript_call_result_37441, str_37442], **kwargs_37443)
    
    # Processing the call keyword arguments (line 249)
    kwargs_37445 = {}
    # Getting the type of 'lib_dirs' (line 249)
    lib_dirs_37432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'lib_dirs', False)
    # Obtaining the member 'append' of a type (line 249)
    append_37433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), lib_dirs_37432, 'append')
    # Calling append(args, kwargs) (line 249)
    append_call_result_37446 = invoke(stypy.reporting.localization.Localization(__file__, 249, 8), append_37433, *[join_call_result_37444], **kwargs_37445)
    
    # SSA branch for the except part of a try statement (line 248)
    # SSA branch for the except 'KeyError' branch of a try statement (line 248)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 248)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'lib_dirs' (line 253)
    lib_dirs_37447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 13), 'lib_dirs')
    # Testing the type of a for loop iterable (line 253)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 253, 4), lib_dirs_37447)
    # Getting the type of the for loop variable (line 253)
    for_loop_var_37448 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 253, 4), lib_dirs_37447)
    # Assigning a type to the variable 'd' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'd', for_loop_var_37448)
    # SSA begins for a for statement (line 253)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 254):
    
    # Assigning a Call to a Name (line 254):
    
    # Call to join(...): (line 254)
    # Processing the call arguments (line 254)
    # Getting the type of 'd' (line 254)
    d_37452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 27), 'd', False)
    # Getting the type of 'dllname' (line 254)
    dllname_37453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 30), 'dllname', False)
    # Processing the call keyword arguments (line 254)
    kwargs_37454 = {}
    # Getting the type of 'os' (line 254)
    os_37449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 14), 'os', False)
    # Obtaining the member 'path' of a type (line 254)
    path_37450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 14), os_37449, 'path')
    # Obtaining the member 'join' of a type (line 254)
    join_37451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 14), path_37450, 'join')
    # Calling join(args, kwargs) (line 254)
    join_call_result_37455 = invoke(stypy.reporting.localization.Localization(__file__, 254, 14), join_37451, *[d_37452, dllname_37453], **kwargs_37454)
    
    # Assigning a type to the variable 'dll' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'dll', join_call_result_37455)
    
    
    # Call to exists(...): (line 255)
    # Processing the call arguments (line 255)
    # Getting the type of 'dll' (line 255)
    dll_37459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 26), 'dll', False)
    # Processing the call keyword arguments (line 255)
    kwargs_37460 = {}
    # Getting the type of 'os' (line 255)
    os_37456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 255)
    path_37457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 11), os_37456, 'path')
    # Obtaining the member 'exists' of a type (line 255)
    exists_37458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 11), path_37457, 'exists')
    # Calling exists(args, kwargs) (line 255)
    exists_call_result_37461 = invoke(stypy.reporting.localization.Localization(__file__, 255, 11), exists_37458, *[dll_37459], **kwargs_37460)
    
    # Testing the type of an if condition (line 255)
    if_condition_37462 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 8), exists_call_result_37461)
    # Assigning a type to the variable 'if_condition_37462' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'if_condition_37462', if_condition_37462)
    # SSA begins for if statement (line 255)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'dll' (line 256)
    dll_37463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 19), 'dll')
    # Assigning a type to the variable 'stypy_return_type' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'stypy_return_type', dll_37463)
    # SSA join for if statement (line 255)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to ValueError(...): (line 258)
    # Processing the call arguments (line 258)
    str_37465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 21), 'str', '%s not found in %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 258)
    tuple_37466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 45), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 258)
    # Adding element type (line 258)
    # Getting the type of 'dllname' (line 258)
    dllname_37467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 45), 'dllname', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 45), tuple_37466, dllname_37467)
    # Adding element type (line 258)
    # Getting the type of 'lib_dirs' (line 258)
    lib_dirs_37468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 54), 'lib_dirs', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 45), tuple_37466, lib_dirs_37468)
    
    # Applying the binary operator '%' (line 258)
    result_mod_37469 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 21), '%', str_37465, tuple_37466)
    
    # Processing the call keyword arguments (line 258)
    kwargs_37470 = {}
    # Getting the type of 'ValueError' (line 258)
    ValueError_37464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 10), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 258)
    ValueError_call_result_37471 = invoke(stypy.reporting.localization.Localization(__file__, 258, 10), ValueError_37464, *[result_mod_37469], **kwargs_37470)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 258, 4), ValueError_call_result_37471, 'raise parameter', BaseException)
    
    # ################# End of 'find_python_dll(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'find_python_dll' in the type store
    # Getting the type of 'stypy_return_type' (line 238)
    stypy_return_type_37472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_37472)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'find_python_dll'
    return stypy_return_type_37472

# Assigning a type to the variable 'find_python_dll' (line 238)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 0), 'find_python_dll', find_python_dll)

@norecursion
def dump_table(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'dump_table'
    module_type_store = module_type_store.open_function_context('dump_table', 260, 0, False)
    
    # Passed parameters checking function
    dump_table.stypy_localization = localization
    dump_table.stypy_type_of_self = None
    dump_table.stypy_type_store = module_type_store
    dump_table.stypy_function_name = 'dump_table'
    dump_table.stypy_param_names_list = ['dll']
    dump_table.stypy_varargs_param_name = None
    dump_table.stypy_kwargs_param_name = None
    dump_table.stypy_call_defaults = defaults
    dump_table.stypy_call_varargs = varargs
    dump_table.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dump_table', ['dll'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dump_table', localization, ['dll'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dump_table(...)' code ##################

    
    # Assigning a Call to a Name (line 261):
    
    # Assigning a Call to a Name (line 261):
    
    # Call to Popen(...): (line 261)
    # Processing the call arguments (line 261)
    
    # Obtaining an instance of the builtin type 'list' (line 261)
    list_37475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 261)
    # Adding element type (line 261)
    str_37476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 27), 'str', 'objdump.exe')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 26), list_37475, str_37476)
    # Adding element type (line 261)
    str_37477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 42), 'str', '-p')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 26), list_37475, str_37477)
    # Adding element type (line 261)
    # Getting the type of 'dll' (line 261)
    dll_37478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 48), 'dll', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 26), list_37475, dll_37478)
    
    # Processing the call keyword arguments (line 261)
    # Getting the type of 'subprocess' (line 261)
    subprocess_37479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 61), 'subprocess', False)
    # Obtaining the member 'PIPE' of a type (line 261)
    PIPE_37480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 61), subprocess_37479, 'PIPE')
    keyword_37481 = PIPE_37480
    kwargs_37482 = {'stdout': keyword_37481}
    # Getting the type of 'subprocess' (line 261)
    subprocess_37473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 9), 'subprocess', False)
    # Obtaining the member 'Popen' of a type (line 261)
    Popen_37474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 9), subprocess_37473, 'Popen')
    # Calling Popen(args, kwargs) (line 261)
    Popen_call_result_37483 = invoke(stypy.reporting.localization.Localization(__file__, 261, 9), Popen_37474, *[list_37475], **kwargs_37482)
    
    # Assigning a type to the variable 'st' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'st', Popen_call_result_37483)
    
    # Call to readlines(...): (line 262)
    # Processing the call keyword arguments (line 262)
    kwargs_37487 = {}
    # Getting the type of 'st' (line 262)
    st_37484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 11), 'st', False)
    # Obtaining the member 'stdout' of a type (line 262)
    stdout_37485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 11), st_37484, 'stdout')
    # Obtaining the member 'readlines' of a type (line 262)
    readlines_37486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 11), stdout_37485, 'readlines')
    # Calling readlines(args, kwargs) (line 262)
    readlines_call_result_37488 = invoke(stypy.reporting.localization.Localization(__file__, 262, 11), readlines_37486, *[], **kwargs_37487)
    
    # Assigning a type to the variable 'stypy_return_type' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'stypy_return_type', readlines_call_result_37488)
    
    # ################# End of 'dump_table(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dump_table' in the type store
    # Getting the type of 'stypy_return_type' (line 260)
    stypy_return_type_37489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_37489)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dump_table'
    return stypy_return_type_37489

# Assigning a type to the variable 'dump_table' (line 260)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 0), 'dump_table', dump_table)

@norecursion
def generate_def(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'generate_def'
    module_type_store = module_type_store.open_function_context('generate_def', 264, 0, False)
    
    # Passed parameters checking function
    generate_def.stypy_localization = localization
    generate_def.stypy_type_of_self = None
    generate_def.stypy_type_store = module_type_store
    generate_def.stypy_function_name = 'generate_def'
    generate_def.stypy_param_names_list = ['dll', 'dfile']
    generate_def.stypy_varargs_param_name = None
    generate_def.stypy_kwargs_param_name = None
    generate_def.stypy_call_defaults = defaults
    generate_def.stypy_call_varargs = varargs
    generate_def.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'generate_def', ['dll', 'dfile'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'generate_def', localization, ['dll', 'dfile'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'generate_def(...)' code ##################

    str_37490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, (-1)), 'str', 'Given a dll file location,  get all its exported symbols and dump them\n    into the given def file.\n\n    The .def file will be overwritten')
    
    # Assigning a Call to a Name (line 269):
    
    # Assigning a Call to a Name (line 269):
    
    # Call to dump_table(...): (line 269)
    # Processing the call arguments (line 269)
    # Getting the type of 'dll' (line 269)
    dll_37492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 22), 'dll', False)
    # Processing the call keyword arguments (line 269)
    kwargs_37493 = {}
    # Getting the type of 'dump_table' (line 269)
    dump_table_37491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 11), 'dump_table', False)
    # Calling dump_table(args, kwargs) (line 269)
    dump_table_call_result_37494 = invoke(stypy.reporting.localization.Localization(__file__, 269, 11), dump_table_37491, *[dll_37492], **kwargs_37493)
    
    # Assigning a type to the variable 'dump' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'dump', dump_table_call_result_37494)
    
    
    # Call to range(...): (line 270)
    # Processing the call arguments (line 270)
    
    # Call to len(...): (line 270)
    # Processing the call arguments (line 270)
    # Getting the type of 'dump' (line 270)
    dump_37497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 23), 'dump', False)
    # Processing the call keyword arguments (line 270)
    kwargs_37498 = {}
    # Getting the type of 'len' (line 270)
    len_37496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 19), 'len', False)
    # Calling len(args, kwargs) (line 270)
    len_call_result_37499 = invoke(stypy.reporting.localization.Localization(__file__, 270, 19), len_37496, *[dump_37497], **kwargs_37498)
    
    # Processing the call keyword arguments (line 270)
    kwargs_37500 = {}
    # Getting the type of 'range' (line 270)
    range_37495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 13), 'range', False)
    # Calling range(args, kwargs) (line 270)
    range_call_result_37501 = invoke(stypy.reporting.localization.Localization(__file__, 270, 13), range_37495, *[len_call_result_37499], **kwargs_37500)
    
    # Testing the type of a for loop iterable (line 270)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 270, 4), range_call_result_37501)
    # Getting the type of the for loop variable (line 270)
    for_loop_var_37502 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 270, 4), range_call_result_37501)
    # Assigning a type to the variable 'i' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'i', for_loop_var_37502)
    # SSA begins for a for statement (line 270)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to match(...): (line 271)
    # Processing the call arguments (line 271)
    
    # Call to decode(...): (line 271)
    # Processing the call keyword arguments (line 271)
    kwargs_37510 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 271)
    i_37505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 29), 'i', False)
    # Getting the type of 'dump' (line 271)
    dump_37506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 24), 'dump', False)
    # Obtaining the member '__getitem__' of a type (line 271)
    getitem___37507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 24), dump_37506, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 271)
    subscript_call_result_37508 = invoke(stypy.reporting.localization.Localization(__file__, 271, 24), getitem___37507, i_37505)
    
    # Obtaining the member 'decode' of a type (line 271)
    decode_37509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 24), subscript_call_result_37508, 'decode')
    # Calling decode(args, kwargs) (line 271)
    decode_call_result_37511 = invoke(stypy.reporting.localization.Localization(__file__, 271, 24), decode_37509, *[], **kwargs_37510)
    
    # Processing the call keyword arguments (line 271)
    kwargs_37512 = {}
    # Getting the type of '_START' (line 271)
    _START_37503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 11), '_START', False)
    # Obtaining the member 'match' of a type (line 271)
    match_37504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 11), _START_37503, 'match')
    # Calling match(args, kwargs) (line 271)
    match_call_result_37513 = invoke(stypy.reporting.localization.Localization(__file__, 271, 11), match_37504, *[decode_call_result_37511], **kwargs_37512)
    
    # Testing the type of an if condition (line 271)
    if_condition_37514 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 271, 8), match_call_result_37513)
    # Assigning a type to the variable 'if_condition_37514' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'if_condition_37514', if_condition_37514)
    # SSA begins for if statement (line 271)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 271)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of a for statement (line 270)
    module_type_store.open_ssa_branch('for loop else')
    
    # Call to ValueError(...): (line 274)
    # Processing the call arguments (line 274)
    str_37516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 25), 'str', 'Symbol table not found')
    # Processing the call keyword arguments (line 274)
    kwargs_37517 = {}
    # Getting the type of 'ValueError' (line 274)
    ValueError_37515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 274)
    ValueError_call_result_37518 = invoke(stypy.reporting.localization.Localization(__file__, 274, 14), ValueError_37515, *[str_37516], **kwargs_37517)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 274, 8), ValueError_call_result_37518, 'raise parameter', BaseException)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 276):
    
    # Assigning a List to a Name (line 276):
    
    # Obtaining an instance of the builtin type 'list' (line 276)
    list_37519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 276)
    
    # Assigning a type to the variable 'syms' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'syms', list_37519)
    
    
    # Call to range(...): (line 277)
    # Processing the call arguments (line 277)
    # Getting the type of 'i' (line 277)
    i_37521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 19), 'i', False)
    int_37522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 21), 'int')
    # Applying the binary operator '+' (line 277)
    result_add_37523 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 19), '+', i_37521, int_37522)
    
    
    # Call to len(...): (line 277)
    # Processing the call arguments (line 277)
    # Getting the type of 'dump' (line 277)
    dump_37525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 28), 'dump', False)
    # Processing the call keyword arguments (line 277)
    kwargs_37526 = {}
    # Getting the type of 'len' (line 277)
    len_37524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 24), 'len', False)
    # Calling len(args, kwargs) (line 277)
    len_call_result_37527 = invoke(stypy.reporting.localization.Localization(__file__, 277, 24), len_37524, *[dump_37525], **kwargs_37526)
    
    # Processing the call keyword arguments (line 277)
    kwargs_37528 = {}
    # Getting the type of 'range' (line 277)
    range_37520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 13), 'range', False)
    # Calling range(args, kwargs) (line 277)
    range_call_result_37529 = invoke(stypy.reporting.localization.Localization(__file__, 277, 13), range_37520, *[result_add_37523, len_call_result_37527], **kwargs_37528)
    
    # Testing the type of a for loop iterable (line 277)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 277, 4), range_call_result_37529)
    # Getting the type of the for loop variable (line 277)
    for_loop_var_37530 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 277, 4), range_call_result_37529)
    # Assigning a type to the variable 'j' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'j', for_loop_var_37530)
    # SSA begins for a for statement (line 277)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 278):
    
    # Assigning a Call to a Name (line 278):
    
    # Call to match(...): (line 278)
    # Processing the call arguments (line 278)
    
    # Call to decode(...): (line 278)
    # Processing the call keyword arguments (line 278)
    kwargs_37538 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 278)
    j_37533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 30), 'j', False)
    # Getting the type of 'dump' (line 278)
    dump_37534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 25), 'dump', False)
    # Obtaining the member '__getitem__' of a type (line 278)
    getitem___37535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 25), dump_37534, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 278)
    subscript_call_result_37536 = invoke(stypy.reporting.localization.Localization(__file__, 278, 25), getitem___37535, j_37533)
    
    # Obtaining the member 'decode' of a type (line 278)
    decode_37537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 25), subscript_call_result_37536, 'decode')
    # Calling decode(args, kwargs) (line 278)
    decode_call_result_37539 = invoke(stypy.reporting.localization.Localization(__file__, 278, 25), decode_37537, *[], **kwargs_37538)
    
    # Processing the call keyword arguments (line 278)
    kwargs_37540 = {}
    # Getting the type of '_TABLE' (line 278)
    _TABLE_37531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), '_TABLE', False)
    # Obtaining the member 'match' of a type (line 278)
    match_37532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 12), _TABLE_37531, 'match')
    # Calling match(args, kwargs) (line 278)
    match_call_result_37541 = invoke(stypy.reporting.localization.Localization(__file__, 278, 12), match_37532, *[decode_call_result_37539], **kwargs_37540)
    
    # Assigning a type to the variable 'm' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'm', match_call_result_37541)
    
    # Getting the type of 'm' (line 279)
    m_37542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 11), 'm')
    # Testing the type of an if condition (line 279)
    if_condition_37543 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 279, 8), m_37542)
    # Assigning a type to the variable 'if_condition_37543' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'if_condition_37543', if_condition_37543)
    # SSA begins for if statement (line 279)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 280)
    # Processing the call arguments (line 280)
    
    # Obtaining an instance of the builtin type 'tuple' (line 280)
    tuple_37546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 280)
    # Adding element type (line 280)
    
    # Call to int(...): (line 280)
    # Processing the call arguments (line 280)
    
    # Call to strip(...): (line 280)
    # Processing the call keyword arguments (line 280)
    kwargs_37554 = {}
    
    # Call to group(...): (line 280)
    # Processing the call arguments (line 280)
    int_37550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 37), 'int')
    # Processing the call keyword arguments (line 280)
    kwargs_37551 = {}
    # Getting the type of 'm' (line 280)
    m_37548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 29), 'm', False)
    # Obtaining the member 'group' of a type (line 280)
    group_37549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 29), m_37548, 'group')
    # Calling group(args, kwargs) (line 280)
    group_call_result_37552 = invoke(stypy.reporting.localization.Localization(__file__, 280, 29), group_37549, *[int_37550], **kwargs_37551)
    
    # Obtaining the member 'strip' of a type (line 280)
    strip_37553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 29), group_call_result_37552, 'strip')
    # Calling strip(args, kwargs) (line 280)
    strip_call_result_37555 = invoke(stypy.reporting.localization.Localization(__file__, 280, 29), strip_37553, *[], **kwargs_37554)
    
    # Processing the call keyword arguments (line 280)
    kwargs_37556 = {}
    # Getting the type of 'int' (line 280)
    int_37547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 25), 'int', False)
    # Calling int(args, kwargs) (line 280)
    int_call_result_37557 = invoke(stypy.reporting.localization.Localization(__file__, 280, 25), int_37547, *[strip_call_result_37555], **kwargs_37556)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 25), tuple_37546, int_call_result_37557)
    # Adding element type (line 280)
    
    # Call to group(...): (line 280)
    # Processing the call arguments (line 280)
    int_37560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 58), 'int')
    # Processing the call keyword arguments (line 280)
    kwargs_37561 = {}
    # Getting the type of 'm' (line 280)
    m_37558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 50), 'm', False)
    # Obtaining the member 'group' of a type (line 280)
    group_37559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 50), m_37558, 'group')
    # Calling group(args, kwargs) (line 280)
    group_call_result_37562 = invoke(stypy.reporting.localization.Localization(__file__, 280, 50), group_37559, *[int_37560], **kwargs_37561)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 25), tuple_37546, group_call_result_37562)
    
    # Processing the call keyword arguments (line 280)
    kwargs_37563 = {}
    # Getting the type of 'syms' (line 280)
    syms_37544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'syms', False)
    # Obtaining the member 'append' of a type (line 280)
    append_37545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 12), syms_37544, 'append')
    # Calling append(args, kwargs) (line 280)
    append_call_result_37564 = invoke(stypy.reporting.localization.Localization(__file__, 280, 12), append_37545, *[tuple_37546], **kwargs_37563)
    
    # SSA branch for the else part of an if statement (line 279)
    module_type_store.open_ssa_branch('else')
    # SSA join for if statement (line 279)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 284)
    # Processing the call arguments (line 284)
    # Getting the type of 'syms' (line 284)
    syms_37566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 11), 'syms', False)
    # Processing the call keyword arguments (line 284)
    kwargs_37567 = {}
    # Getting the type of 'len' (line 284)
    len_37565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 7), 'len', False)
    # Calling len(args, kwargs) (line 284)
    len_call_result_37568 = invoke(stypy.reporting.localization.Localization(__file__, 284, 7), len_37565, *[syms_37566], **kwargs_37567)
    
    int_37569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 20), 'int')
    # Applying the binary operator '==' (line 284)
    result_eq_37570 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 7), '==', len_call_result_37568, int_37569)
    
    # Testing the type of an if condition (line 284)
    if_condition_37571 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 284, 4), result_eq_37570)
    # Assigning a type to the variable 'if_condition_37571' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'if_condition_37571', if_condition_37571)
    # SSA begins for if statement (line 284)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 285)
    # Processing the call arguments (line 285)
    str_37574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 17), 'str', 'No symbols found in %s')
    # Getting the type of 'dll' (line 285)
    dll_37575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 44), 'dll', False)
    # Applying the binary operator '%' (line 285)
    result_mod_37576 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 17), '%', str_37574, dll_37575)
    
    # Processing the call keyword arguments (line 285)
    kwargs_37577 = {}
    # Getting the type of 'log' (line 285)
    log_37572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'log', False)
    # Obtaining the member 'warn' of a type (line 285)
    warn_37573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 8), log_37572, 'warn')
    # Calling warn(args, kwargs) (line 285)
    warn_call_result_37578 = invoke(stypy.reporting.localization.Localization(__file__, 285, 8), warn_37573, *[result_mod_37576], **kwargs_37577)
    
    # SSA join for if statement (line 284)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 287):
    
    # Assigning a Call to a Name (line 287):
    
    # Call to open(...): (line 287)
    # Processing the call arguments (line 287)
    # Getting the type of 'dfile' (line 287)
    dfile_37580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 13), 'dfile', False)
    str_37581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 20), 'str', 'w')
    # Processing the call keyword arguments (line 287)
    kwargs_37582 = {}
    # Getting the type of 'open' (line 287)
    open_37579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'open', False)
    # Calling open(args, kwargs) (line 287)
    open_call_result_37583 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), open_37579, *[dfile_37580, str_37581], **kwargs_37582)
    
    # Assigning a type to the variable 'd' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'd', open_call_result_37583)
    
    # Call to write(...): (line 288)
    # Processing the call arguments (line 288)
    str_37586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 12), 'str', 'LIBRARY        %s\n')
    
    # Call to basename(...): (line 288)
    # Processing the call arguments (line 288)
    # Getting the type of 'dll' (line 288)
    dll_37590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 53), 'dll', False)
    # Processing the call keyword arguments (line 288)
    kwargs_37591 = {}
    # Getting the type of 'os' (line 288)
    os_37587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 36), 'os', False)
    # Obtaining the member 'path' of a type (line 288)
    path_37588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 36), os_37587, 'path')
    # Obtaining the member 'basename' of a type (line 288)
    basename_37589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 36), path_37588, 'basename')
    # Calling basename(args, kwargs) (line 288)
    basename_call_result_37592 = invoke(stypy.reporting.localization.Localization(__file__, 288, 36), basename_37589, *[dll_37590], **kwargs_37591)
    
    # Applying the binary operator '%' (line 288)
    result_mod_37593 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 12), '%', str_37586, basename_call_result_37592)
    
    # Processing the call keyword arguments (line 288)
    kwargs_37594 = {}
    # Getting the type of 'd' (line 288)
    d_37584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'd', False)
    # Obtaining the member 'write' of a type (line 288)
    write_37585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 4), d_37584, 'write')
    # Calling write(args, kwargs) (line 288)
    write_call_result_37595 = invoke(stypy.reporting.localization.Localization(__file__, 288, 4), write_37585, *[result_mod_37593], **kwargs_37594)
    
    
    # Call to write(...): (line 289)
    # Processing the call arguments (line 289)
    str_37598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 12), 'str', ';CODE          PRELOAD MOVEABLE DISCARDABLE\n')
    # Processing the call keyword arguments (line 289)
    kwargs_37599 = {}
    # Getting the type of 'd' (line 289)
    d_37596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'd', False)
    # Obtaining the member 'write' of a type (line 289)
    write_37597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 4), d_37596, 'write')
    # Calling write(args, kwargs) (line 289)
    write_call_result_37600 = invoke(stypy.reporting.localization.Localization(__file__, 289, 4), write_37597, *[str_37598], **kwargs_37599)
    
    
    # Call to write(...): (line 290)
    # Processing the call arguments (line 290)
    str_37603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 12), 'str', ';DATA          PRELOAD SINGLE\n')
    # Processing the call keyword arguments (line 290)
    kwargs_37604 = {}
    # Getting the type of 'd' (line 290)
    d_37601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'd', False)
    # Obtaining the member 'write' of a type (line 290)
    write_37602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 4), d_37601, 'write')
    # Calling write(args, kwargs) (line 290)
    write_call_result_37605 = invoke(stypy.reporting.localization.Localization(__file__, 290, 4), write_37602, *[str_37603], **kwargs_37604)
    
    
    # Call to write(...): (line 291)
    # Processing the call arguments (line 291)
    str_37608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 12), 'str', '\nEXPORTS\n')
    # Processing the call keyword arguments (line 291)
    kwargs_37609 = {}
    # Getting the type of 'd' (line 291)
    d_37606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'd', False)
    # Obtaining the member 'write' of a type (line 291)
    write_37607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 4), d_37606, 'write')
    # Calling write(args, kwargs) (line 291)
    write_call_result_37610 = invoke(stypy.reporting.localization.Localization(__file__, 291, 4), write_37607, *[str_37608], **kwargs_37609)
    
    
    # Getting the type of 'syms' (line 292)
    syms_37611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 13), 'syms')
    # Testing the type of a for loop iterable (line 292)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 292, 4), syms_37611)
    # Getting the type of the for loop variable (line 292)
    for_loop_var_37612 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 292, 4), syms_37611)
    # Assigning a type to the variable 's' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 's', for_loop_var_37612)
    # SSA begins for a for statement (line 292)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to write(...): (line 294)
    # Processing the call arguments (line 294)
    str_37615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 16), 'str', '%s\n')
    
    # Obtaining the type of the subscript
    int_37616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 27), 'int')
    # Getting the type of 's' (line 294)
    s_37617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 25), 's', False)
    # Obtaining the member '__getitem__' of a type (line 294)
    getitem___37618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 25), s_37617, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 294)
    subscript_call_result_37619 = invoke(stypy.reporting.localization.Localization(__file__, 294, 25), getitem___37618, int_37616)
    
    # Applying the binary operator '%' (line 294)
    result_mod_37620 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 16), '%', str_37615, subscript_call_result_37619)
    
    # Processing the call keyword arguments (line 294)
    kwargs_37621 = {}
    # Getting the type of 'd' (line 294)
    d_37613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'd', False)
    # Obtaining the member 'write' of a type (line 294)
    write_37614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 8), d_37613, 'write')
    # Calling write(args, kwargs) (line 294)
    write_call_result_37622 = invoke(stypy.reporting.localization.Localization(__file__, 294, 8), write_37614, *[result_mod_37620], **kwargs_37621)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to close(...): (line 295)
    # Processing the call keyword arguments (line 295)
    kwargs_37625 = {}
    # Getting the type of 'd' (line 295)
    d_37623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'd', False)
    # Obtaining the member 'close' of a type (line 295)
    close_37624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 4), d_37623, 'close')
    # Calling close(args, kwargs) (line 295)
    close_call_result_37626 = invoke(stypy.reporting.localization.Localization(__file__, 295, 4), close_37624, *[], **kwargs_37625)
    
    
    # ################# End of 'generate_def(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'generate_def' in the type store
    # Getting the type of 'stypy_return_type' (line 264)
    stypy_return_type_37627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_37627)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'generate_def'
    return stypy_return_type_37627

# Assigning a type to the variable 'generate_def' (line 264)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 0), 'generate_def', generate_def)

@norecursion
def find_dll(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'find_dll'
    module_type_store = module_type_store.open_function_context('find_dll', 297, 0, False)
    
    # Passed parameters checking function
    find_dll.stypy_localization = localization
    find_dll.stypy_type_of_self = None
    find_dll.stypy_type_store = module_type_store
    find_dll.stypy_function_name = 'find_dll'
    find_dll.stypy_param_names_list = ['dll_name']
    find_dll.stypy_varargs_param_name = None
    find_dll.stypy_kwargs_param_name = None
    find_dll.stypy_call_defaults = defaults
    find_dll.stypy_call_varargs = varargs
    find_dll.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'find_dll', ['dll_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'find_dll', localization, ['dll_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'find_dll(...)' code ##################

    
    # Assigning a Subscript to a Name (line 299):
    
    # Assigning a Subscript to a Name (line 299):
    
    # Obtaining the type of the subscript
    
    # Call to get_build_architecture(...): (line 300)
    # Processing the call keyword arguments (line 300)
    kwargs_37629 = {}
    # Getting the type of 'get_build_architecture' (line 300)
    get_build_architecture_37628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 29), 'get_build_architecture', False)
    # Calling get_build_architecture(args, kwargs) (line 300)
    get_build_architecture_call_result_37630 = invoke(stypy.reporting.localization.Localization(__file__, 300, 29), get_build_architecture_37628, *[], **kwargs_37629)
    
    
    # Obtaining an instance of the builtin type 'dict' (line 299)
    dict_37631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 11), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 299)
    # Adding element type (key, value) (line 299)
    str_37632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 12), 'str', 'AMD64')
    str_37633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 22), 'str', 'amd64')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 11), dict_37631, (str_37632, str_37633))
    # Adding element type (key, value) (line 299)
    str_37634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 12), 'str', 'Intel')
    str_37635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 22), 'str', 'x86')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 11), dict_37631, (str_37634, str_37635))
    
    # Obtaining the member '__getitem__' of a type (line 299)
    getitem___37636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 11), dict_37631, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 299)
    subscript_call_result_37637 = invoke(stypy.reporting.localization.Localization(__file__, 299, 11), getitem___37636, get_build_architecture_call_result_37630)
    
    # Assigning a type to the variable 'arch' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'arch', subscript_call_result_37637)

    @norecursion
    def _find_dll_in_winsxs(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_find_dll_in_winsxs'
        module_type_store = module_type_store.open_function_context('_find_dll_in_winsxs', 302, 4, False)
        
        # Passed parameters checking function
        _find_dll_in_winsxs.stypy_localization = localization
        _find_dll_in_winsxs.stypy_type_of_self = None
        _find_dll_in_winsxs.stypy_type_store = module_type_store
        _find_dll_in_winsxs.stypy_function_name = '_find_dll_in_winsxs'
        _find_dll_in_winsxs.stypy_param_names_list = ['dll_name']
        _find_dll_in_winsxs.stypy_varargs_param_name = None
        _find_dll_in_winsxs.stypy_kwargs_param_name = None
        _find_dll_in_winsxs.stypy_call_defaults = defaults
        _find_dll_in_winsxs.stypy_call_varargs = varargs
        _find_dll_in_winsxs.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_find_dll_in_winsxs', ['dll_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_find_dll_in_winsxs', localization, ['dll_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_find_dll_in_winsxs(...)' code ##################

        
        # Assigning a Call to a Name (line 304):
        
        # Assigning a Call to a Name (line 304):
        
        # Call to join(...): (line 304)
        # Processing the call arguments (line 304)
        
        # Obtaining the type of the subscript
        str_37641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 46), 'str', 'WINDIR')
        # Getting the type of 'os' (line 304)
        os_37642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 35), 'os', False)
        # Obtaining the member 'environ' of a type (line 304)
        environ_37643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 35), os_37642, 'environ')
        # Obtaining the member '__getitem__' of a type (line 304)
        getitem___37644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 35), environ_37643, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 304)
        subscript_call_result_37645 = invoke(stypy.reporting.localization.Localization(__file__, 304, 35), getitem___37644, str_37641)
        
        str_37646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 57), 'str', 'winsxs')
        # Processing the call keyword arguments (line 304)
        kwargs_37647 = {}
        # Getting the type of 'os' (line 304)
        os_37638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 22), 'os', False)
        # Obtaining the member 'path' of a type (line 304)
        path_37639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 22), os_37638, 'path')
        # Obtaining the member 'join' of a type (line 304)
        join_37640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 22), path_37639, 'join')
        # Calling join(args, kwargs) (line 304)
        join_call_result_37648 = invoke(stypy.reporting.localization.Localization(__file__, 304, 22), join_37640, *[subscript_call_result_37645, str_37646], **kwargs_37647)
        
        # Assigning a type to the variable 'winsxs_path' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'winsxs_path', join_call_result_37648)
        
        
        
        # Call to exists(...): (line 305)
        # Processing the call arguments (line 305)
        # Getting the type of 'winsxs_path' (line 305)
        winsxs_path_37652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 30), 'winsxs_path', False)
        # Processing the call keyword arguments (line 305)
        kwargs_37653 = {}
        # Getting the type of 'os' (line 305)
        os_37649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 305)
        path_37650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 15), os_37649, 'path')
        # Obtaining the member 'exists' of a type (line 305)
        exists_37651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 15), path_37650, 'exists')
        # Calling exists(args, kwargs) (line 305)
        exists_call_result_37654 = invoke(stypy.reporting.localization.Localization(__file__, 305, 15), exists_37651, *[winsxs_path_37652], **kwargs_37653)
        
        # Applying the 'not' unary operator (line 305)
        result_not__37655 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 11), 'not', exists_call_result_37654)
        
        # Testing the type of an if condition (line 305)
        if_condition_37656 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 305, 8), result_not__37655)
        # Assigning a type to the variable 'if_condition_37656' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'if_condition_37656', if_condition_37656)
        # SSA begins for if statement (line 305)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'None' (line 306)
        None_37657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'stypy_return_type', None_37657)
        # SSA join for if statement (line 305)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to walk(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 'winsxs_path' (line 307)
        winsxs_path_37660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 41), 'winsxs_path', False)
        # Processing the call keyword arguments (line 307)
        kwargs_37661 = {}
        # Getting the type of 'os' (line 307)
        os_37658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 33), 'os', False)
        # Obtaining the member 'walk' of a type (line 307)
        walk_37659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 33), os_37658, 'walk')
        # Calling walk(args, kwargs) (line 307)
        walk_call_result_37662 = invoke(stypy.reporting.localization.Localization(__file__, 307, 33), walk_37659, *[winsxs_path_37660], **kwargs_37661)
        
        # Testing the type of a for loop iterable (line 307)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 307, 8), walk_call_result_37662)
        # Getting the type of the for loop variable (line 307)
        for_loop_var_37663 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 307, 8), walk_call_result_37662)
        # Assigning a type to the variable 'root' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'root', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 8), for_loop_var_37663))
        # Assigning a type to the variable 'dirs' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'dirs', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 8), for_loop_var_37663))
        # Assigning a type to the variable 'files' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'files', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 8), for_loop_var_37663))
        # SSA begins for a for statement (line 307)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'dll_name' (line 308)
        dll_name_37664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 15), 'dll_name')
        # Getting the type of 'files' (line 308)
        files_37665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 27), 'files')
        # Applying the binary operator 'in' (line 308)
        result_contains_37666 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 15), 'in', dll_name_37664, files_37665)
        
        
        # Getting the type of 'arch' (line 308)
        arch_37667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 37), 'arch')
        # Getting the type of 'root' (line 308)
        root_37668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 45), 'root')
        # Applying the binary operator 'in' (line 308)
        result_contains_37669 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 37), 'in', arch_37667, root_37668)
        
        # Applying the binary operator 'and' (line 308)
        result_and_keyword_37670 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 15), 'and', result_contains_37666, result_contains_37669)
        
        # Testing the type of an if condition (line 308)
        if_condition_37671 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 308, 12), result_and_keyword_37670)
        # Assigning a type to the variable 'if_condition_37671' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'if_condition_37671', if_condition_37671)
        # SSA begins for if statement (line 308)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to join(...): (line 309)
        # Processing the call arguments (line 309)
        # Getting the type of 'root' (line 309)
        root_37675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 36), 'root', False)
        # Getting the type of 'dll_name' (line 309)
        dll_name_37676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 42), 'dll_name', False)
        # Processing the call keyword arguments (line 309)
        kwargs_37677 = {}
        # Getting the type of 'os' (line 309)
        os_37672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 309)
        path_37673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 23), os_37672, 'path')
        # Obtaining the member 'join' of a type (line 309)
        join_37674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 23), path_37673, 'join')
        # Calling join(args, kwargs) (line 309)
        join_call_result_37678 = invoke(stypy.reporting.localization.Localization(__file__, 309, 23), join_37674, *[root_37675, dll_name_37676], **kwargs_37677)
        
        # Assigning a type to the variable 'stypy_return_type' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 16), 'stypy_return_type', join_call_result_37678)
        # SSA join for if statement (line 308)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'None' (line 310)
        None_37679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'stypy_return_type', None_37679)
        
        # ################# End of '_find_dll_in_winsxs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_find_dll_in_winsxs' in the type store
        # Getting the type of 'stypy_return_type' (line 302)
        stypy_return_type_37680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37680)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_find_dll_in_winsxs'
        return stypy_return_type_37680

    # Assigning a type to the variable '_find_dll_in_winsxs' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), '_find_dll_in_winsxs', _find_dll_in_winsxs)

    @norecursion
    def _find_dll_in_path(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_find_dll_in_path'
        module_type_store = module_type_store.open_function_context('_find_dll_in_path', 312, 4, False)
        
        # Passed parameters checking function
        _find_dll_in_path.stypy_localization = localization
        _find_dll_in_path.stypy_type_of_self = None
        _find_dll_in_path.stypy_type_store = module_type_store
        _find_dll_in_path.stypy_function_name = '_find_dll_in_path'
        _find_dll_in_path.stypy_param_names_list = ['dll_name']
        _find_dll_in_path.stypy_varargs_param_name = None
        _find_dll_in_path.stypy_kwargs_param_name = None
        _find_dll_in_path.stypy_call_defaults = defaults
        _find_dll_in_path.stypy_call_varargs = varargs
        _find_dll_in_path.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_find_dll_in_path', ['dll_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_find_dll_in_path', localization, ['dll_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_find_dll_in_path(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 315)
        list_37681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 315)
        # Adding element type (line 315)
        # Getting the type of 'sys' (line 315)
        sys_37682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 21), 'sys')
        # Obtaining the member 'prefix' of a type (line 315)
        prefix_37683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 21), sys_37682, 'prefix')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 20), list_37681, prefix_37683)
        
        
        # Call to split(...): (line 315)
        # Processing the call arguments (line 315)
        str_37690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 60), 'str', ';')
        # Processing the call keyword arguments (line 315)
        kwargs_37691 = {}
        
        # Obtaining the type of the subscript
        str_37684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 46), 'str', 'PATH')
        # Getting the type of 'os' (line 315)
        os_37685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 35), 'os', False)
        # Obtaining the member 'environ' of a type (line 315)
        environ_37686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 35), os_37685, 'environ')
        # Obtaining the member '__getitem__' of a type (line 315)
        getitem___37687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 35), environ_37686, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 315)
        subscript_call_result_37688 = invoke(stypy.reporting.localization.Localization(__file__, 315, 35), getitem___37687, str_37684)
        
        # Obtaining the member 'split' of a type (line 315)
        split_37689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 35), subscript_call_result_37688, 'split')
        # Calling split(args, kwargs) (line 315)
        split_call_result_37692 = invoke(stypy.reporting.localization.Localization(__file__, 315, 35), split_37689, *[str_37690], **kwargs_37691)
        
        # Applying the binary operator '+' (line 315)
        result_add_37693 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 20), '+', list_37681, split_call_result_37692)
        
        # Testing the type of a for loop iterable (line 315)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 315, 8), result_add_37693)
        # Getting the type of the for loop variable (line 315)
        for_loop_var_37694 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 315, 8), result_add_37693)
        # Assigning a type to the variable 'path' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'path', for_loop_var_37694)
        # SSA begins for a for statement (line 315)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 316):
        
        # Assigning a Call to a Name (line 316):
        
        # Call to join(...): (line 316)
        # Processing the call arguments (line 316)
        # Getting the type of 'path' (line 316)
        path_37698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 36), 'path', False)
        # Getting the type of 'dll_name' (line 316)
        dll_name_37699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 42), 'dll_name', False)
        # Processing the call keyword arguments (line 316)
        kwargs_37700 = {}
        # Getting the type of 'os' (line 316)
        os_37695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 316)
        path_37696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 23), os_37695, 'path')
        # Obtaining the member 'join' of a type (line 316)
        join_37697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 23), path_37696, 'join')
        # Calling join(args, kwargs) (line 316)
        join_call_result_37701 = invoke(stypy.reporting.localization.Localization(__file__, 316, 23), join_37697, *[path_37698, dll_name_37699], **kwargs_37700)
        
        # Assigning a type to the variable 'filepath' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'filepath', join_call_result_37701)
        
        
        # Call to exists(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'filepath' (line 317)
        filepath_37705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 30), 'filepath', False)
        # Processing the call keyword arguments (line 317)
        kwargs_37706 = {}
        # Getting the type of 'os' (line 317)
        os_37702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 317)
        path_37703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 15), os_37702, 'path')
        # Obtaining the member 'exists' of a type (line 317)
        exists_37704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 15), path_37703, 'exists')
        # Calling exists(args, kwargs) (line 317)
        exists_call_result_37707 = invoke(stypy.reporting.localization.Localization(__file__, 317, 15), exists_37704, *[filepath_37705], **kwargs_37706)
        
        # Testing the type of an if condition (line 317)
        if_condition_37708 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 317, 12), exists_call_result_37707)
        # Assigning a type to the variable 'if_condition_37708' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'if_condition_37708', if_condition_37708)
        # SSA begins for if statement (line 317)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to abspath(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'filepath' (line 318)
        filepath_37712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 39), 'filepath', False)
        # Processing the call keyword arguments (line 318)
        kwargs_37713 = {}
        # Getting the type of 'os' (line 318)
        os_37709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 318)
        path_37710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 23), os_37709, 'path')
        # Obtaining the member 'abspath' of a type (line 318)
        abspath_37711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 23), path_37710, 'abspath')
        # Calling abspath(args, kwargs) (line 318)
        abspath_call_result_37714 = invoke(stypy.reporting.localization.Localization(__file__, 318, 23), abspath_37711, *[filepath_37712], **kwargs_37713)
        
        # Assigning a type to the variable 'stypy_return_type' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 16), 'stypy_return_type', abspath_call_result_37714)
        # SSA join for if statement (line 317)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_find_dll_in_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_find_dll_in_path' in the type store
        # Getting the type of 'stypy_return_type' (line 312)
        stypy_return_type_37715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37715)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_find_dll_in_path'
        return stypy_return_type_37715

    # Assigning a type to the variable '_find_dll_in_path' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), '_find_dll_in_path', _find_dll_in_path)
    
    # Evaluating a boolean operation
    
    # Call to _find_dll_in_winsxs(...): (line 320)
    # Processing the call arguments (line 320)
    # Getting the type of 'dll_name' (line 320)
    dll_name_37717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 31), 'dll_name', False)
    # Processing the call keyword arguments (line 320)
    kwargs_37718 = {}
    # Getting the type of '_find_dll_in_winsxs' (line 320)
    _find_dll_in_winsxs_37716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 11), '_find_dll_in_winsxs', False)
    # Calling _find_dll_in_winsxs(args, kwargs) (line 320)
    _find_dll_in_winsxs_call_result_37719 = invoke(stypy.reporting.localization.Localization(__file__, 320, 11), _find_dll_in_winsxs_37716, *[dll_name_37717], **kwargs_37718)
    
    
    # Call to _find_dll_in_path(...): (line 320)
    # Processing the call arguments (line 320)
    # Getting the type of 'dll_name' (line 320)
    dll_name_37721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 62), 'dll_name', False)
    # Processing the call keyword arguments (line 320)
    kwargs_37722 = {}
    # Getting the type of '_find_dll_in_path' (line 320)
    _find_dll_in_path_37720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 44), '_find_dll_in_path', False)
    # Calling _find_dll_in_path(args, kwargs) (line 320)
    _find_dll_in_path_call_result_37723 = invoke(stypy.reporting.localization.Localization(__file__, 320, 44), _find_dll_in_path_37720, *[dll_name_37721], **kwargs_37722)
    
    # Applying the binary operator 'or' (line 320)
    result_or_keyword_37724 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 11), 'or', _find_dll_in_winsxs_call_result_37719, _find_dll_in_path_call_result_37723)
    
    # Assigning a type to the variable 'stypy_return_type' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'stypy_return_type', result_or_keyword_37724)
    
    # ################# End of 'find_dll(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'find_dll' in the type store
    # Getting the type of 'stypy_return_type' (line 297)
    stypy_return_type_37725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_37725)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'find_dll'
    return stypy_return_type_37725

# Assigning a type to the variable 'find_dll' (line 297)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 0), 'find_dll', find_dll)

@norecursion
def build_msvcr_library(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 322)
    False_37726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 30), 'False')
    defaults = [False_37726]
    # Create a new context for function 'build_msvcr_library'
    module_type_store = module_type_store.open_function_context('build_msvcr_library', 322, 0, False)
    
    # Passed parameters checking function
    build_msvcr_library.stypy_localization = localization
    build_msvcr_library.stypy_type_of_self = None
    build_msvcr_library.stypy_type_store = module_type_store
    build_msvcr_library.stypy_function_name = 'build_msvcr_library'
    build_msvcr_library.stypy_param_names_list = ['debug']
    build_msvcr_library.stypy_varargs_param_name = None
    build_msvcr_library.stypy_kwargs_param_name = None
    build_msvcr_library.stypy_call_defaults = defaults
    build_msvcr_library.stypy_call_varargs = varargs
    build_msvcr_library.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'build_msvcr_library', ['debug'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'build_msvcr_library', localization, ['debug'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'build_msvcr_library(...)' code ##################

    
    
    # Getting the type of 'os' (line 323)
    os_37727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 7), 'os')
    # Obtaining the member 'name' of a type (line 323)
    name_37728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 7), os_37727, 'name')
    str_37729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 18), 'str', 'nt')
    # Applying the binary operator '!=' (line 323)
    result_ne_37730 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 7), '!=', name_37728, str_37729)
    
    # Testing the type of an if condition (line 323)
    if_condition_37731 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 323, 4), result_ne_37730)
    # Assigning a type to the variable 'if_condition_37731' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'if_condition_37731', if_condition_37731)
    # SSA begins for if statement (line 323)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'False' (line 324)
    False_37732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'stypy_return_type', False_37732)
    # SSA join for if statement (line 323)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 326):
    
    # Assigning a Call to a Name (line 326):
    
    # Call to msvc_runtime_library(...): (line 326)
    # Processing the call keyword arguments (line 326)
    kwargs_37734 = {}
    # Getting the type of 'msvc_runtime_library' (line 326)
    msvc_runtime_library_37733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 17), 'msvc_runtime_library', False)
    # Calling msvc_runtime_library(args, kwargs) (line 326)
    msvc_runtime_library_call_result_37735 = invoke(stypy.reporting.localization.Localization(__file__, 326, 17), msvc_runtime_library_37733, *[], **kwargs_37734)
    
    # Assigning a type to the variable 'msvcr_name' (line 326)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'msvcr_name', msvc_runtime_library_call_result_37735)
    
    
    
    # Call to int(...): (line 329)
    # Processing the call arguments (line 329)
    
    # Call to lstrip(...): (line 329)
    # Processing the call arguments (line 329)
    str_37739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 29), 'str', 'msvcr')
    # Processing the call keyword arguments (line 329)
    kwargs_37740 = {}
    # Getting the type of 'msvcr_name' (line 329)
    msvcr_name_37737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 11), 'msvcr_name', False)
    # Obtaining the member 'lstrip' of a type (line 329)
    lstrip_37738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 11), msvcr_name_37737, 'lstrip')
    # Calling lstrip(args, kwargs) (line 329)
    lstrip_call_result_37741 = invoke(stypy.reporting.localization.Localization(__file__, 329, 11), lstrip_37738, *[str_37739], **kwargs_37740)
    
    # Processing the call keyword arguments (line 329)
    kwargs_37742 = {}
    # Getting the type of 'int' (line 329)
    int_37736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 7), 'int', False)
    # Calling int(args, kwargs) (line 329)
    int_call_result_37743 = invoke(stypy.reporting.localization.Localization(__file__, 329, 7), int_37736, *[lstrip_call_result_37741], **kwargs_37742)
    
    int_37744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 41), 'int')
    # Applying the binary operator '<' (line 329)
    result_lt_37745 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 7), '<', int_call_result_37743, int_37744)
    
    # Testing the type of an if condition (line 329)
    if_condition_37746 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 329, 4), result_lt_37745)
    # Assigning a type to the variable 'if_condition_37746' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'if_condition_37746', if_condition_37746)
    # SSA begins for if statement (line 329)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to debug(...): (line 330)
    # Processing the call arguments (line 330)
    str_37749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 18), 'str', 'Skip building msvcr library: custom functionality not present')
    # Processing the call keyword arguments (line 330)
    kwargs_37750 = {}
    # Getting the type of 'log' (line 330)
    log_37747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'log', False)
    # Obtaining the member 'debug' of a type (line 330)
    debug_37748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 8), log_37747, 'debug')
    # Calling debug(args, kwargs) (line 330)
    debug_call_result_37751 = invoke(stypy.reporting.localization.Localization(__file__, 330, 8), debug_37748, *[str_37749], **kwargs_37750)
    
    # Getting the type of 'False' (line 332)
    False_37752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'stypy_return_type', False_37752)
    # SSA join for if statement (line 329)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'debug' (line 334)
    debug_37753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 7), 'debug')
    # Testing the type of an if condition (line 334)
    if_condition_37754 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 4), debug_37753)
    # Assigning a type to the variable 'if_condition_37754' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'if_condition_37754', if_condition_37754)
    # SSA begins for if statement (line 334)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'msvcr_name' (line 335)
    msvcr_name_37755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'msvcr_name')
    str_37756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 22), 'str', 'd')
    # Applying the binary operator '+=' (line 335)
    result_iadd_37757 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 8), '+=', msvcr_name_37755, str_37756)
    # Assigning a type to the variable 'msvcr_name' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'msvcr_name', result_iadd_37757)
    
    # SSA join for if statement (line 334)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 338):
    
    # Assigning a BinOp to a Name (line 338):
    str_37758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 15), 'str', 'lib%s.a')
    # Getting the type of 'msvcr_name' (line 338)
    msvcr_name_37759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 27), 'msvcr_name')
    # Applying the binary operator '%' (line 338)
    result_mod_37760 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 15), '%', str_37758, msvcr_name_37759)
    
    # Assigning a type to the variable 'out_name' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'out_name', result_mod_37760)
    
    # Assigning a Call to a Name (line 339):
    
    # Assigning a Call to a Name (line 339):
    
    # Call to join(...): (line 339)
    # Processing the call arguments (line 339)
    # Getting the type of 'sys' (line 339)
    sys_37764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 28), 'sys', False)
    # Obtaining the member 'prefix' of a type (line 339)
    prefix_37765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 28), sys_37764, 'prefix')
    str_37766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 40), 'str', 'libs')
    # Getting the type of 'out_name' (line 339)
    out_name_37767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 48), 'out_name', False)
    # Processing the call keyword arguments (line 339)
    kwargs_37768 = {}
    # Getting the type of 'os' (line 339)
    os_37761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 339)
    path_37762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 15), os_37761, 'path')
    # Obtaining the member 'join' of a type (line 339)
    join_37763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 15), path_37762, 'join')
    # Calling join(args, kwargs) (line 339)
    join_call_result_37769 = invoke(stypy.reporting.localization.Localization(__file__, 339, 15), join_37763, *[prefix_37765, str_37766, out_name_37767], **kwargs_37768)
    
    # Assigning a type to the variable 'out_file' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'out_file', join_call_result_37769)
    
    
    # Call to isfile(...): (line 340)
    # Processing the call arguments (line 340)
    # Getting the type of 'out_file' (line 340)
    out_file_37773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 22), 'out_file', False)
    # Processing the call keyword arguments (line 340)
    kwargs_37774 = {}
    # Getting the type of 'os' (line 340)
    os_37770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 7), 'os', False)
    # Obtaining the member 'path' of a type (line 340)
    path_37771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 7), os_37770, 'path')
    # Obtaining the member 'isfile' of a type (line 340)
    isfile_37772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 7), path_37771, 'isfile')
    # Calling isfile(args, kwargs) (line 340)
    isfile_call_result_37775 = invoke(stypy.reporting.localization.Localization(__file__, 340, 7), isfile_37772, *[out_file_37773], **kwargs_37774)
    
    # Testing the type of an if condition (line 340)
    if_condition_37776 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 340, 4), isfile_call_result_37775)
    # Assigning a type to the variable 'if_condition_37776' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'if_condition_37776', if_condition_37776)
    # SSA begins for if statement (line 340)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to debug(...): (line 341)
    # Processing the call arguments (line 341)
    str_37779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 18), 'str', 'Skip building msvcr library: "%s" exists')
    
    # Obtaining an instance of the builtin type 'tuple' (line 342)
    tuple_37780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 342)
    # Adding element type (line 342)
    # Getting the type of 'out_file' (line 342)
    out_file_37781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 19), 'out_file', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 19), tuple_37780, out_file_37781)
    
    # Applying the binary operator '%' (line 341)
    result_mod_37782 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 18), '%', str_37779, tuple_37780)
    
    # Processing the call keyword arguments (line 341)
    kwargs_37783 = {}
    # Getting the type of 'log' (line 341)
    log_37777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'log', False)
    # Obtaining the member 'debug' of a type (line 341)
    debug_37778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 8), log_37777, 'debug')
    # Calling debug(args, kwargs) (line 341)
    debug_call_result_37784 = invoke(stypy.reporting.localization.Localization(__file__, 341, 8), debug_37778, *[result_mod_37782], **kwargs_37783)
    
    # Getting the type of 'True' (line 343)
    True_37785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 15), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'stypy_return_type', True_37785)
    # SSA join for if statement (line 340)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 346):
    
    # Assigning a BinOp to a Name (line 346):
    # Getting the type of 'msvcr_name' (line 346)
    msvcr_name_37786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 21), 'msvcr_name')
    str_37787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 34), 'str', '.dll')
    # Applying the binary operator '+' (line 346)
    result_add_37788 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 21), '+', msvcr_name_37786, str_37787)
    
    # Assigning a type to the variable 'msvcr_dll_name' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'msvcr_dll_name', result_add_37788)
    
    # Assigning a Call to a Name (line 347):
    
    # Assigning a Call to a Name (line 347):
    
    # Call to find_dll(...): (line 347)
    # Processing the call arguments (line 347)
    # Getting the type of 'msvcr_dll_name' (line 347)
    msvcr_dll_name_37790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 24), 'msvcr_dll_name', False)
    # Processing the call keyword arguments (line 347)
    kwargs_37791 = {}
    # Getting the type of 'find_dll' (line 347)
    find_dll_37789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 15), 'find_dll', False)
    # Calling find_dll(args, kwargs) (line 347)
    find_dll_call_result_37792 = invoke(stypy.reporting.localization.Localization(__file__, 347, 15), find_dll_37789, *[msvcr_dll_name_37790], **kwargs_37791)
    
    # Assigning a type to the variable 'dll_file' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'dll_file', find_dll_call_result_37792)
    
    
    # Getting the type of 'dll_file' (line 348)
    dll_file_37793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 11), 'dll_file')
    # Applying the 'not' unary operator (line 348)
    result_not__37794 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 7), 'not', dll_file_37793)
    
    # Testing the type of an if condition (line 348)
    if_condition_37795 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 348, 4), result_not__37794)
    # Assigning a type to the variable 'if_condition_37795' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'if_condition_37795', if_condition_37795)
    # SSA begins for if statement (line 348)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 349)
    # Processing the call arguments (line 349)
    str_37798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 17), 'str', 'Cannot build msvcr library: "%s" not found')
    # Getting the type of 'msvcr_dll_name' (line 350)
    msvcr_dll_name_37799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 17), 'msvcr_dll_name', False)
    # Applying the binary operator '%' (line 349)
    result_mod_37800 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 17), '%', str_37798, msvcr_dll_name_37799)
    
    # Processing the call keyword arguments (line 349)
    kwargs_37801 = {}
    # Getting the type of 'log' (line 349)
    log_37796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'log', False)
    # Obtaining the member 'warn' of a type (line 349)
    warn_37797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 8), log_37796, 'warn')
    # Calling warn(args, kwargs) (line 349)
    warn_call_result_37802 = invoke(stypy.reporting.localization.Localization(__file__, 349, 8), warn_37797, *[result_mod_37800], **kwargs_37801)
    
    # Getting the type of 'False' (line 351)
    False_37803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'stypy_return_type', False_37803)
    # SSA join for if statement (line 348)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 353):
    
    # Assigning a BinOp to a Name (line 353):
    str_37804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 15), 'str', 'lib%s.def')
    # Getting the type of 'msvcr_name' (line 353)
    msvcr_name_37805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 29), 'msvcr_name')
    # Applying the binary operator '%' (line 353)
    result_mod_37806 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 15), '%', str_37804, msvcr_name_37805)
    
    # Assigning a type to the variable 'def_name' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'def_name', result_mod_37806)
    
    # Assigning a Call to a Name (line 354):
    
    # Assigning a Call to a Name (line 354):
    
    # Call to join(...): (line 354)
    # Processing the call arguments (line 354)
    # Getting the type of 'sys' (line 354)
    sys_37810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 28), 'sys', False)
    # Obtaining the member 'prefix' of a type (line 354)
    prefix_37811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 28), sys_37810, 'prefix')
    str_37812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 40), 'str', 'libs')
    # Getting the type of 'def_name' (line 354)
    def_name_37813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 48), 'def_name', False)
    # Processing the call keyword arguments (line 354)
    kwargs_37814 = {}
    # Getting the type of 'os' (line 354)
    os_37807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 354)
    path_37808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 15), os_37807, 'path')
    # Obtaining the member 'join' of a type (line 354)
    join_37809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 15), path_37808, 'join')
    # Calling join(args, kwargs) (line 354)
    join_call_result_37815 = invoke(stypy.reporting.localization.Localization(__file__, 354, 15), join_37809, *[prefix_37811, str_37812, def_name_37813], **kwargs_37814)
    
    # Assigning a type to the variable 'def_file' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'def_file', join_call_result_37815)
    
    # Call to info(...): (line 356)
    # Processing the call arguments (line 356)
    str_37818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 13), 'str', 'Building msvcr library: "%s" (from %s)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 357)
    tuple_37819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 357)
    # Adding element type (line 357)
    # Getting the type of 'out_file' (line 357)
    out_file_37820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 16), 'out_file', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 16), tuple_37819, out_file_37820)
    # Adding element type (line 357)
    # Getting the type of 'dll_file' (line 357)
    dll_file_37821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 26), 'dll_file', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 16), tuple_37819, dll_file_37821)
    
    # Applying the binary operator '%' (line 356)
    result_mod_37822 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 13), '%', str_37818, tuple_37819)
    
    # Processing the call keyword arguments (line 356)
    kwargs_37823 = {}
    # Getting the type of 'log' (line 356)
    log_37816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'log', False)
    # Obtaining the member 'info' of a type (line 356)
    info_37817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 4), log_37816, 'info')
    # Calling info(args, kwargs) (line 356)
    info_call_result_37824 = invoke(stypy.reporting.localization.Localization(__file__, 356, 4), info_37817, *[result_mod_37822], **kwargs_37823)
    
    
    # Call to generate_def(...): (line 360)
    # Processing the call arguments (line 360)
    # Getting the type of 'dll_file' (line 360)
    dll_file_37826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 17), 'dll_file', False)
    # Getting the type of 'def_file' (line 360)
    def_file_37827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 27), 'def_file', False)
    # Processing the call keyword arguments (line 360)
    kwargs_37828 = {}
    # Getting the type of 'generate_def' (line 360)
    generate_def_37825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'generate_def', False)
    # Calling generate_def(args, kwargs) (line 360)
    generate_def_call_result_37829 = invoke(stypy.reporting.localization.Localization(__file__, 360, 4), generate_def_37825, *[dll_file_37826, def_file_37827], **kwargs_37828)
    
    
    # Assigning a List to a Name (line 363):
    
    # Assigning a List to a Name (line 363):
    
    # Obtaining an instance of the builtin type 'list' (line 363)
    list_37830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 363)
    # Adding element type (line 363)
    str_37831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 11), 'str', 'dlltool')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 10), list_37830, str_37831)
    # Adding element type (line 363)
    str_37832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 22), 'str', '-d')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 10), list_37830, str_37832)
    # Adding element type (line 363)
    # Getting the type of 'def_file' (line 363)
    def_file_37833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 28), 'def_file')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 10), list_37830, def_file_37833)
    # Adding element type (line 363)
    str_37834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 38), 'str', '-l')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 10), list_37830, str_37834)
    # Adding element type (line 363)
    # Getting the type of 'out_file' (line 363)
    out_file_37835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 44), 'out_file')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 10), list_37830, out_file_37835)
    
    # Assigning a type to the variable 'cmd' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 4), 'cmd', list_37830)
    
    # Assigning a Call to a Name (line 364):
    
    # Assigning a Call to a Name (line 364):
    
    # Call to call(...): (line 364)
    # Processing the call arguments (line 364)
    # Getting the type of 'cmd' (line 364)
    cmd_37838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 30), 'cmd', False)
    # Processing the call keyword arguments (line 364)
    kwargs_37839 = {}
    # Getting the type of 'subprocess' (line 364)
    subprocess_37836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 14), 'subprocess', False)
    # Obtaining the member 'call' of a type (line 364)
    call_37837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 14), subprocess_37836, 'call')
    # Calling call(args, kwargs) (line 364)
    call_call_result_37840 = invoke(stypy.reporting.localization.Localization(__file__, 364, 14), call_37837, *[cmd_37838], **kwargs_37839)
    
    # Assigning a type to the variable 'retcode' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'retcode', call_call_result_37840)
    
    # Call to remove(...): (line 367)
    # Processing the call arguments (line 367)
    # Getting the type of 'def_file' (line 367)
    def_file_37843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 14), 'def_file', False)
    # Processing the call keyword arguments (line 367)
    kwargs_37844 = {}
    # Getting the type of 'os' (line 367)
    os_37841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'os', False)
    # Obtaining the member 'remove' of a type (line 367)
    remove_37842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 4), os_37841, 'remove')
    # Calling remove(args, kwargs) (line 367)
    remove_call_result_37845 = invoke(stypy.reporting.localization.Localization(__file__, 367, 4), remove_37842, *[def_file_37843], **kwargs_37844)
    
    
    # Getting the type of 'retcode' (line 369)
    retcode_37846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 16), 'retcode')
    # Applying the 'not' unary operator (line 369)
    result_not__37847 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 12), 'not', retcode_37846)
    
    # Assigning a type to the variable 'stypy_return_type' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'stypy_return_type', result_not__37847)
    
    # ################# End of 'build_msvcr_library(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'build_msvcr_library' in the type store
    # Getting the type of 'stypy_return_type' (line 322)
    stypy_return_type_37848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_37848)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'build_msvcr_library'
    return stypy_return_type_37848

# Assigning a type to the variable 'build_msvcr_library' (line 322)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 0), 'build_msvcr_library', build_msvcr_library)

@norecursion
def build_import_library(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'build_import_library'
    module_type_store = module_type_store.open_function_context('build_import_library', 371, 0, False)
    
    # Passed parameters checking function
    build_import_library.stypy_localization = localization
    build_import_library.stypy_type_of_self = None
    build_import_library.stypy_type_store = module_type_store
    build_import_library.stypy_function_name = 'build_import_library'
    build_import_library.stypy_param_names_list = []
    build_import_library.stypy_varargs_param_name = None
    build_import_library.stypy_kwargs_param_name = None
    build_import_library.stypy_call_defaults = defaults
    build_import_library.stypy_call_varargs = varargs
    build_import_library.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'build_import_library', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'build_import_library', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'build_import_library(...)' code ##################

    
    
    # Getting the type of 'os' (line 372)
    os_37849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 7), 'os')
    # Obtaining the member 'name' of a type (line 372)
    name_37850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 7), os_37849, 'name')
    str_37851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 18), 'str', 'nt')
    # Applying the binary operator '!=' (line 372)
    result_ne_37852 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 7), '!=', name_37850, str_37851)
    
    # Testing the type of an if condition (line 372)
    if_condition_37853 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 372, 4), result_ne_37852)
    # Assigning a type to the variable 'if_condition_37853' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'if_condition_37853', if_condition_37853)
    # SSA begins for if statement (line 372)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Assigning a type to the variable 'stypy_return_type' (line 373)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 372)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 375):
    
    # Assigning a Call to a Name (line 375):
    
    # Call to get_build_architecture(...): (line 375)
    # Processing the call keyword arguments (line 375)
    kwargs_37855 = {}
    # Getting the type of 'get_build_architecture' (line 375)
    get_build_architecture_37854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 11), 'get_build_architecture', False)
    # Calling get_build_architecture(args, kwargs) (line 375)
    get_build_architecture_call_result_37856 = invoke(stypy.reporting.localization.Localization(__file__, 375, 11), get_build_architecture_37854, *[], **kwargs_37855)
    
    # Assigning a type to the variable 'arch' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'arch', get_build_architecture_call_result_37856)
    
    
    # Getting the type of 'arch' (line 376)
    arch_37857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 7), 'arch')
    str_37858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 15), 'str', 'AMD64')
    # Applying the binary operator '==' (line 376)
    result_eq_37859 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 7), '==', arch_37857, str_37858)
    
    # Testing the type of an if condition (line 376)
    if_condition_37860 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 376, 4), result_eq_37859)
    # Assigning a type to the variable 'if_condition_37860' (line 376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'if_condition_37860', if_condition_37860)
    # SSA begins for if statement (line 376)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _build_import_library_amd64(...): (line 377)
    # Processing the call keyword arguments (line 377)
    kwargs_37862 = {}
    # Getting the type of '_build_import_library_amd64' (line 377)
    _build_import_library_amd64_37861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 15), '_build_import_library_amd64', False)
    # Calling _build_import_library_amd64(args, kwargs) (line 377)
    _build_import_library_amd64_call_result_37863 = invoke(stypy.reporting.localization.Localization(__file__, 377, 15), _build_import_library_amd64_37861, *[], **kwargs_37862)
    
    # Assigning a type to the variable 'stypy_return_type' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'stypy_return_type', _build_import_library_amd64_call_result_37863)
    # SSA branch for the else part of an if statement (line 376)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'arch' (line 378)
    arch_37864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 9), 'arch')
    str_37865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 17), 'str', 'Intel')
    # Applying the binary operator '==' (line 378)
    result_eq_37866 = python_operator(stypy.reporting.localization.Localization(__file__, 378, 9), '==', arch_37864, str_37865)
    
    # Testing the type of an if condition (line 378)
    if_condition_37867 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 378, 9), result_eq_37866)
    # Assigning a type to the variable 'if_condition_37867' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 9), 'if_condition_37867', if_condition_37867)
    # SSA begins for if statement (line 378)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _build_import_library_x86(...): (line 379)
    # Processing the call keyword arguments (line 379)
    kwargs_37869 = {}
    # Getting the type of '_build_import_library_x86' (line 379)
    _build_import_library_x86_37868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 15), '_build_import_library_x86', False)
    # Calling _build_import_library_x86(args, kwargs) (line 379)
    _build_import_library_x86_call_result_37870 = invoke(stypy.reporting.localization.Localization(__file__, 379, 15), _build_import_library_x86_37868, *[], **kwargs_37869)
    
    # Assigning a type to the variable 'stypy_return_type' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'stypy_return_type', _build_import_library_x86_call_result_37870)
    # SSA branch for the else part of an if statement (line 378)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 381)
    # Processing the call arguments (line 381)
    str_37872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 25), 'str', 'Unhandled arch %s')
    # Getting the type of 'arch' (line 381)
    arch_37873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 47), 'arch', False)
    # Applying the binary operator '%' (line 381)
    result_mod_37874 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 25), '%', str_37872, arch_37873)
    
    # Processing the call keyword arguments (line 381)
    kwargs_37875 = {}
    # Getting the type of 'ValueError' (line 381)
    ValueError_37871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 381)
    ValueError_call_result_37876 = invoke(stypy.reporting.localization.Localization(__file__, 381, 14), ValueError_37871, *[result_mod_37874], **kwargs_37875)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 381, 8), ValueError_call_result_37876, 'raise parameter', BaseException)
    # SSA join for if statement (line 378)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 376)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'build_import_library(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'build_import_library' in the type store
    # Getting the type of 'stypy_return_type' (line 371)
    stypy_return_type_37877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_37877)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'build_import_library'
    return stypy_return_type_37877

# Assigning a type to the variable 'build_import_library' (line 371)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 0), 'build_import_library', build_import_library)

@norecursion
def _build_import_library_amd64(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_build_import_library_amd64'
    module_type_store = module_type_store.open_function_context('_build_import_library_amd64', 383, 0, False)
    
    # Passed parameters checking function
    _build_import_library_amd64.stypy_localization = localization
    _build_import_library_amd64.stypy_type_of_self = None
    _build_import_library_amd64.stypy_type_store = module_type_store
    _build_import_library_amd64.stypy_function_name = '_build_import_library_amd64'
    _build_import_library_amd64.stypy_param_names_list = []
    _build_import_library_amd64.stypy_varargs_param_name = None
    _build_import_library_amd64.stypy_kwargs_param_name = None
    _build_import_library_amd64.stypy_call_defaults = defaults
    _build_import_library_amd64.stypy_call_varargs = varargs
    _build_import_library_amd64.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_build_import_library_amd64', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_build_import_library_amd64', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_build_import_library_amd64(...)' code ##################

    
    # Assigning a Call to a Name (line 384):
    
    # Assigning a Call to a Name (line 384):
    
    # Call to find_python_dll(...): (line 384)
    # Processing the call keyword arguments (line 384)
    kwargs_37879 = {}
    # Getting the type of 'find_python_dll' (line 384)
    find_python_dll_37878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 15), 'find_python_dll', False)
    # Calling find_python_dll(args, kwargs) (line 384)
    find_python_dll_call_result_37880 = invoke(stypy.reporting.localization.Localization(__file__, 384, 15), find_python_dll_37878, *[], **kwargs_37879)
    
    # Assigning a type to the variable 'dll_file' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'dll_file', find_python_dll_call_result_37880)
    
    # Assigning a BinOp to a Name (line 386):
    
    # Assigning a BinOp to a Name (line 386):
    str_37881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 15), 'str', 'libpython%d%d.a')
    
    # Call to tuple(...): (line 386)
    # Processing the call arguments (line 386)
    
    # Obtaining the type of the subscript
    int_37883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 59), 'int')
    slice_37884 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 386, 41), None, int_37883, None)
    # Getting the type of 'sys' (line 386)
    sys_37885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 41), 'sys', False)
    # Obtaining the member 'version_info' of a type (line 386)
    version_info_37886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 41), sys_37885, 'version_info')
    # Obtaining the member '__getitem__' of a type (line 386)
    getitem___37887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 41), version_info_37886, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 386)
    subscript_call_result_37888 = invoke(stypy.reporting.localization.Localization(__file__, 386, 41), getitem___37887, slice_37884)
    
    # Processing the call keyword arguments (line 386)
    kwargs_37889 = {}
    # Getting the type of 'tuple' (line 386)
    tuple_37882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 35), 'tuple', False)
    # Calling tuple(args, kwargs) (line 386)
    tuple_call_result_37890 = invoke(stypy.reporting.localization.Localization(__file__, 386, 35), tuple_37882, *[subscript_call_result_37888], **kwargs_37889)
    
    # Applying the binary operator '%' (line 386)
    result_mod_37891 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 15), '%', str_37881, tuple_call_result_37890)
    
    # Assigning a type to the variable 'out_name' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'out_name', result_mod_37891)
    
    # Assigning a Call to a Name (line 387):
    
    # Assigning a Call to a Name (line 387):
    
    # Call to join(...): (line 387)
    # Processing the call arguments (line 387)
    # Getting the type of 'sys' (line 387)
    sys_37895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 28), 'sys', False)
    # Obtaining the member 'prefix' of a type (line 387)
    prefix_37896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 28), sys_37895, 'prefix')
    str_37897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 40), 'str', 'libs')
    # Getting the type of 'out_name' (line 387)
    out_name_37898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 48), 'out_name', False)
    # Processing the call keyword arguments (line 387)
    kwargs_37899 = {}
    # Getting the type of 'os' (line 387)
    os_37892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 387)
    path_37893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 15), os_37892, 'path')
    # Obtaining the member 'join' of a type (line 387)
    join_37894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 15), path_37893, 'join')
    # Calling join(args, kwargs) (line 387)
    join_call_result_37900 = invoke(stypy.reporting.localization.Localization(__file__, 387, 15), join_37894, *[prefix_37896, str_37897, out_name_37898], **kwargs_37899)
    
    # Assigning a type to the variable 'out_file' (line 387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'out_file', join_call_result_37900)
    
    
    # Call to isfile(...): (line 388)
    # Processing the call arguments (line 388)
    # Getting the type of 'out_file' (line 388)
    out_file_37904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 22), 'out_file', False)
    # Processing the call keyword arguments (line 388)
    kwargs_37905 = {}
    # Getting the type of 'os' (line 388)
    os_37901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 7), 'os', False)
    # Obtaining the member 'path' of a type (line 388)
    path_37902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 7), os_37901, 'path')
    # Obtaining the member 'isfile' of a type (line 388)
    isfile_37903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 7), path_37902, 'isfile')
    # Calling isfile(args, kwargs) (line 388)
    isfile_call_result_37906 = invoke(stypy.reporting.localization.Localization(__file__, 388, 7), isfile_37903, *[out_file_37904], **kwargs_37905)
    
    # Testing the type of an if condition (line 388)
    if_condition_37907 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 388, 4), isfile_call_result_37906)
    # Assigning a type to the variable 'if_condition_37907' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'if_condition_37907', if_condition_37907)
    # SSA begins for if statement (line 388)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to debug(...): (line 389)
    # Processing the call arguments (line 389)
    str_37910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 18), 'str', 'Skip building import library: "%s" exists')
    # Getting the type of 'out_file' (line 390)
    out_file_37911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 19), 'out_file', False)
    # Applying the binary operator '%' (line 389)
    result_mod_37912 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 18), '%', str_37910, out_file_37911)
    
    # Processing the call keyword arguments (line 389)
    kwargs_37913 = {}
    # Getting the type of 'log' (line 389)
    log_37908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'log', False)
    # Obtaining the member 'debug' of a type (line 389)
    debug_37909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 8), log_37908, 'debug')
    # Calling debug(args, kwargs) (line 389)
    debug_call_result_37914 = invoke(stypy.reporting.localization.Localization(__file__, 389, 8), debug_37909, *[result_mod_37912], **kwargs_37913)
    
    # Assigning a type to the variable 'stypy_return_type' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 388)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 393):
    
    # Assigning a BinOp to a Name (line 393):
    str_37915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 15), 'str', 'python%d%d.def')
    
    # Call to tuple(...): (line 393)
    # Processing the call arguments (line 393)
    
    # Obtaining the type of the subscript
    int_37917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 58), 'int')
    slice_37918 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 393, 40), None, int_37917, None)
    # Getting the type of 'sys' (line 393)
    sys_37919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 40), 'sys', False)
    # Obtaining the member 'version_info' of a type (line 393)
    version_info_37920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 40), sys_37919, 'version_info')
    # Obtaining the member '__getitem__' of a type (line 393)
    getitem___37921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 40), version_info_37920, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 393)
    subscript_call_result_37922 = invoke(stypy.reporting.localization.Localization(__file__, 393, 40), getitem___37921, slice_37918)
    
    # Processing the call keyword arguments (line 393)
    kwargs_37923 = {}
    # Getting the type of 'tuple' (line 393)
    tuple_37916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 34), 'tuple', False)
    # Calling tuple(args, kwargs) (line 393)
    tuple_call_result_37924 = invoke(stypy.reporting.localization.Localization(__file__, 393, 34), tuple_37916, *[subscript_call_result_37922], **kwargs_37923)
    
    # Applying the binary operator '%' (line 393)
    result_mod_37925 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 15), '%', str_37915, tuple_call_result_37924)
    
    # Assigning a type to the variable 'def_name' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'def_name', result_mod_37925)
    
    # Assigning a Call to a Name (line 394):
    
    # Assigning a Call to a Name (line 394):
    
    # Call to join(...): (line 394)
    # Processing the call arguments (line 394)
    # Getting the type of 'sys' (line 394)
    sys_37929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 28), 'sys', False)
    # Obtaining the member 'prefix' of a type (line 394)
    prefix_37930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 28), sys_37929, 'prefix')
    str_37931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 40), 'str', 'libs')
    # Getting the type of 'def_name' (line 394)
    def_name_37932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 48), 'def_name', False)
    # Processing the call keyword arguments (line 394)
    kwargs_37933 = {}
    # Getting the type of 'os' (line 394)
    os_37926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 394)
    path_37927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 15), os_37926, 'path')
    # Obtaining the member 'join' of a type (line 394)
    join_37928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 15), path_37927, 'join')
    # Calling join(args, kwargs) (line 394)
    join_call_result_37934 = invoke(stypy.reporting.localization.Localization(__file__, 394, 15), join_37928, *[prefix_37930, str_37931, def_name_37932], **kwargs_37933)
    
    # Assigning a type to the variable 'def_file' (line 394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'def_file', join_call_result_37934)
    
    # Call to info(...): (line 396)
    # Processing the call arguments (line 396)
    str_37937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 13), 'str', 'Building import library (arch=AMD64): "%s" (from %s)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 397)
    tuple_37938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 397)
    # Adding element type (line 397)
    # Getting the type of 'out_file' (line 397)
    out_file_37939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 14), 'out_file', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 14), tuple_37938, out_file_37939)
    # Adding element type (line 397)
    # Getting the type of 'dll_file' (line 397)
    dll_file_37940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 24), 'dll_file', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 14), tuple_37938, dll_file_37940)
    
    # Applying the binary operator '%' (line 396)
    result_mod_37941 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 13), '%', str_37937, tuple_37938)
    
    # Processing the call keyword arguments (line 396)
    kwargs_37942 = {}
    # Getting the type of 'log' (line 396)
    log_37935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 4), 'log', False)
    # Obtaining the member 'info' of a type (line 396)
    info_37936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 4), log_37935, 'info')
    # Calling info(args, kwargs) (line 396)
    info_call_result_37943 = invoke(stypy.reporting.localization.Localization(__file__, 396, 4), info_37936, *[result_mod_37941], **kwargs_37942)
    
    
    # Call to generate_def(...): (line 399)
    # Processing the call arguments (line 399)
    # Getting the type of 'dll_file' (line 399)
    dll_file_37945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 17), 'dll_file', False)
    # Getting the type of 'def_file' (line 399)
    def_file_37946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 27), 'def_file', False)
    # Processing the call keyword arguments (line 399)
    kwargs_37947 = {}
    # Getting the type of 'generate_def' (line 399)
    generate_def_37944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 4), 'generate_def', False)
    # Calling generate_def(args, kwargs) (line 399)
    generate_def_call_result_37948 = invoke(stypy.reporting.localization.Localization(__file__, 399, 4), generate_def_37944, *[dll_file_37945, def_file_37946], **kwargs_37947)
    
    
    # Assigning a List to a Name (line 401):
    
    # Assigning a List to a Name (line 401):
    
    # Obtaining an instance of the builtin type 'list' (line 401)
    list_37949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 401)
    # Adding element type (line 401)
    str_37950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 11), 'str', 'dlltool')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 10), list_37949, str_37950)
    # Adding element type (line 401)
    str_37951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 22), 'str', '-d')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 10), list_37949, str_37951)
    # Adding element type (line 401)
    # Getting the type of 'def_file' (line 401)
    def_file_37952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 28), 'def_file')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 10), list_37949, def_file_37952)
    # Adding element type (line 401)
    str_37953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 38), 'str', '-l')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 10), list_37949, str_37953)
    # Adding element type (line 401)
    # Getting the type of 'out_file' (line 401)
    out_file_37954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 44), 'out_file')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 10), list_37949, out_file_37954)
    
    # Assigning a type to the variable 'cmd' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'cmd', list_37949)
    
    # Call to Popen(...): (line 402)
    # Processing the call arguments (line 402)
    # Getting the type of 'cmd' (line 402)
    cmd_37957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 21), 'cmd', False)
    # Processing the call keyword arguments (line 402)
    kwargs_37958 = {}
    # Getting the type of 'subprocess' (line 402)
    subprocess_37955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'subprocess', False)
    # Obtaining the member 'Popen' of a type (line 402)
    Popen_37956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 4), subprocess_37955, 'Popen')
    # Calling Popen(args, kwargs) (line 402)
    Popen_call_result_37959 = invoke(stypy.reporting.localization.Localization(__file__, 402, 4), Popen_37956, *[cmd_37957], **kwargs_37958)
    
    
    # ################# End of '_build_import_library_amd64(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_build_import_library_amd64' in the type store
    # Getting the type of 'stypy_return_type' (line 383)
    stypy_return_type_37960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_37960)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_build_import_library_amd64'
    return stypy_return_type_37960

# Assigning a type to the variable '_build_import_library_amd64' (line 383)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 0), '_build_import_library_amd64', _build_import_library_amd64)

@norecursion
def _build_import_library_x86(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_build_import_library_x86'
    module_type_store = module_type_store.open_function_context('_build_import_library_x86', 404, 0, False)
    
    # Passed parameters checking function
    _build_import_library_x86.stypy_localization = localization
    _build_import_library_x86.stypy_type_of_self = None
    _build_import_library_x86.stypy_type_store = module_type_store
    _build_import_library_x86.stypy_function_name = '_build_import_library_x86'
    _build_import_library_x86.stypy_param_names_list = []
    _build_import_library_x86.stypy_varargs_param_name = None
    _build_import_library_x86.stypy_kwargs_param_name = None
    _build_import_library_x86.stypy_call_defaults = defaults
    _build_import_library_x86.stypy_call_varargs = varargs
    _build_import_library_x86.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_build_import_library_x86', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_build_import_library_x86', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_build_import_library_x86(...)' code ##################

    str_37961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, (-1)), 'str', ' Build the import libraries for Mingw32-gcc on Windows\n    ')
    
    # Assigning a BinOp to a Name (line 407):
    
    # Assigning a BinOp to a Name (line 407):
    str_37962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 15), 'str', 'python%d%d.lib')
    
    # Call to tuple(...): (line 407)
    # Processing the call arguments (line 407)
    
    # Obtaining the type of the subscript
    int_37964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 58), 'int')
    slice_37965 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 407, 40), None, int_37964, None)
    # Getting the type of 'sys' (line 407)
    sys_37966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 40), 'sys', False)
    # Obtaining the member 'version_info' of a type (line 407)
    version_info_37967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 40), sys_37966, 'version_info')
    # Obtaining the member '__getitem__' of a type (line 407)
    getitem___37968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 40), version_info_37967, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 407)
    subscript_call_result_37969 = invoke(stypy.reporting.localization.Localization(__file__, 407, 40), getitem___37968, slice_37965)
    
    # Processing the call keyword arguments (line 407)
    kwargs_37970 = {}
    # Getting the type of 'tuple' (line 407)
    tuple_37963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 34), 'tuple', False)
    # Calling tuple(args, kwargs) (line 407)
    tuple_call_result_37971 = invoke(stypy.reporting.localization.Localization(__file__, 407, 34), tuple_37963, *[subscript_call_result_37969], **kwargs_37970)
    
    # Applying the binary operator '%' (line 407)
    result_mod_37972 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 15), '%', str_37962, tuple_call_result_37971)
    
    # Assigning a type to the variable 'lib_name' (line 407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'lib_name', result_mod_37972)
    
    # Assigning a Call to a Name (line 408):
    
    # Assigning a Call to a Name (line 408):
    
    # Call to join(...): (line 408)
    # Processing the call arguments (line 408)
    # Getting the type of 'sys' (line 408)
    sys_37976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 28), 'sys', False)
    # Obtaining the member 'prefix' of a type (line 408)
    prefix_37977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 28), sys_37976, 'prefix')
    str_37978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 40), 'str', 'libs')
    # Getting the type of 'lib_name' (line 408)
    lib_name_37979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 48), 'lib_name', False)
    # Processing the call keyword arguments (line 408)
    kwargs_37980 = {}
    # Getting the type of 'os' (line 408)
    os_37973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 408)
    path_37974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 15), os_37973, 'path')
    # Obtaining the member 'join' of a type (line 408)
    join_37975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 15), path_37974, 'join')
    # Calling join(args, kwargs) (line 408)
    join_call_result_37981 = invoke(stypy.reporting.localization.Localization(__file__, 408, 15), join_37975, *[prefix_37977, str_37978, lib_name_37979], **kwargs_37980)
    
    # Assigning a type to the variable 'lib_file' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'lib_file', join_call_result_37981)
    
    # Assigning a BinOp to a Name (line 409):
    
    # Assigning a BinOp to a Name (line 409):
    str_37982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 15), 'str', 'libpython%d%d.a')
    
    # Call to tuple(...): (line 409)
    # Processing the call arguments (line 409)
    
    # Obtaining the type of the subscript
    int_37984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 59), 'int')
    slice_37985 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 409, 41), None, int_37984, None)
    # Getting the type of 'sys' (line 409)
    sys_37986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 41), 'sys', False)
    # Obtaining the member 'version_info' of a type (line 409)
    version_info_37987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 41), sys_37986, 'version_info')
    # Obtaining the member '__getitem__' of a type (line 409)
    getitem___37988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 41), version_info_37987, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 409)
    subscript_call_result_37989 = invoke(stypy.reporting.localization.Localization(__file__, 409, 41), getitem___37988, slice_37985)
    
    # Processing the call keyword arguments (line 409)
    kwargs_37990 = {}
    # Getting the type of 'tuple' (line 409)
    tuple_37983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 35), 'tuple', False)
    # Calling tuple(args, kwargs) (line 409)
    tuple_call_result_37991 = invoke(stypy.reporting.localization.Localization(__file__, 409, 35), tuple_37983, *[subscript_call_result_37989], **kwargs_37990)
    
    # Applying the binary operator '%' (line 409)
    result_mod_37992 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 15), '%', str_37982, tuple_call_result_37991)
    
    # Assigning a type to the variable 'out_name' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'out_name', result_mod_37992)
    
    # Assigning a Call to a Name (line 410):
    
    # Assigning a Call to a Name (line 410):
    
    # Call to join(...): (line 410)
    # Processing the call arguments (line 410)
    # Getting the type of 'sys' (line 410)
    sys_37996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 28), 'sys', False)
    # Obtaining the member 'prefix' of a type (line 410)
    prefix_37997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 28), sys_37996, 'prefix')
    str_37998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 40), 'str', 'libs')
    # Getting the type of 'out_name' (line 410)
    out_name_37999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 48), 'out_name', False)
    # Processing the call keyword arguments (line 410)
    kwargs_38000 = {}
    # Getting the type of 'os' (line 410)
    os_37993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 410)
    path_37994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 15), os_37993, 'path')
    # Obtaining the member 'join' of a type (line 410)
    join_37995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 15), path_37994, 'join')
    # Calling join(args, kwargs) (line 410)
    join_call_result_38001 = invoke(stypy.reporting.localization.Localization(__file__, 410, 15), join_37995, *[prefix_37997, str_37998, out_name_37999], **kwargs_38000)
    
    # Assigning a type to the variable 'out_file' (line 410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'out_file', join_call_result_38001)
    
    
    
    # Call to isfile(...): (line 411)
    # Processing the call arguments (line 411)
    # Getting the type of 'lib_file' (line 411)
    lib_file_38005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 26), 'lib_file', False)
    # Processing the call keyword arguments (line 411)
    kwargs_38006 = {}
    # Getting the type of 'os' (line 411)
    os_38002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 411)
    path_38003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 11), os_38002, 'path')
    # Obtaining the member 'isfile' of a type (line 411)
    isfile_38004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 11), path_38003, 'isfile')
    # Calling isfile(args, kwargs) (line 411)
    isfile_call_result_38007 = invoke(stypy.reporting.localization.Localization(__file__, 411, 11), isfile_38004, *[lib_file_38005], **kwargs_38006)
    
    # Applying the 'not' unary operator (line 411)
    result_not__38008 = python_operator(stypy.reporting.localization.Localization(__file__, 411, 7), 'not', isfile_call_result_38007)
    
    # Testing the type of an if condition (line 411)
    if_condition_38009 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 411, 4), result_not__38008)
    # Assigning a type to the variable 'if_condition_38009' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'if_condition_38009', if_condition_38009)
    # SSA begins for if statement (line 411)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 412)
    # Processing the call arguments (line 412)
    str_38012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 17), 'str', 'Cannot build import library: "%s" not found')
    # Getting the type of 'lib_file' (line 412)
    lib_file_38013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 66), 'lib_file', False)
    # Applying the binary operator '%' (line 412)
    result_mod_38014 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 17), '%', str_38012, lib_file_38013)
    
    # Processing the call keyword arguments (line 412)
    kwargs_38015 = {}
    # Getting the type of 'log' (line 412)
    log_38010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'log', False)
    # Obtaining the member 'warn' of a type (line 412)
    warn_38011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 8), log_38010, 'warn')
    # Calling warn(args, kwargs) (line 412)
    warn_call_result_38016 = invoke(stypy.reporting.localization.Localization(__file__, 412, 8), warn_38011, *[result_mod_38014], **kwargs_38015)
    
    # Assigning a type to the variable 'stypy_return_type' (line 413)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 411)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isfile(...): (line 414)
    # Processing the call arguments (line 414)
    # Getting the type of 'out_file' (line 414)
    out_file_38020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 22), 'out_file', False)
    # Processing the call keyword arguments (line 414)
    kwargs_38021 = {}
    # Getting the type of 'os' (line 414)
    os_38017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 7), 'os', False)
    # Obtaining the member 'path' of a type (line 414)
    path_38018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 7), os_38017, 'path')
    # Obtaining the member 'isfile' of a type (line 414)
    isfile_38019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 7), path_38018, 'isfile')
    # Calling isfile(args, kwargs) (line 414)
    isfile_call_result_38022 = invoke(stypy.reporting.localization.Localization(__file__, 414, 7), isfile_38019, *[out_file_38020], **kwargs_38021)
    
    # Testing the type of an if condition (line 414)
    if_condition_38023 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 414, 4), isfile_call_result_38022)
    # Assigning a type to the variable 'if_condition_38023' (line 414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'if_condition_38023', if_condition_38023)
    # SSA begins for if statement (line 414)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to debug(...): (line 415)
    # Processing the call arguments (line 415)
    str_38026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 18), 'str', 'Skip building import library: "%s" exists')
    # Getting the type of 'out_file' (line 415)
    out_file_38027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 65), 'out_file', False)
    # Applying the binary operator '%' (line 415)
    result_mod_38028 = python_operator(stypy.reporting.localization.Localization(__file__, 415, 18), '%', str_38026, out_file_38027)
    
    # Processing the call keyword arguments (line 415)
    kwargs_38029 = {}
    # Getting the type of 'log' (line 415)
    log_38024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'log', False)
    # Obtaining the member 'debug' of a type (line 415)
    debug_38025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 8), log_38024, 'debug')
    # Calling debug(args, kwargs) (line 415)
    debug_call_result_38030 = invoke(stypy.reporting.localization.Localization(__file__, 415, 8), debug_38025, *[result_mod_38028], **kwargs_38029)
    
    # Assigning a type to the variable 'stypy_return_type' (line 416)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 414)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to info(...): (line 417)
    # Processing the call arguments (line 417)
    str_38033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 13), 'str', 'Building import library (ARCH=x86): "%s"')
    # Getting the type of 'out_file' (line 417)
    out_file_38034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 59), 'out_file', False)
    # Applying the binary operator '%' (line 417)
    result_mod_38035 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 13), '%', str_38033, out_file_38034)
    
    # Processing the call keyword arguments (line 417)
    kwargs_38036 = {}
    # Getting the type of 'log' (line 417)
    log_38031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'log', False)
    # Obtaining the member 'info' of a type (line 417)
    info_38032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 4), log_38031, 'info')
    # Calling info(args, kwargs) (line 417)
    info_call_result_38037 = invoke(stypy.reporting.localization.Localization(__file__, 417, 4), info_38032, *[result_mod_38035], **kwargs_38036)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 419, 4))
    
    # 'from numpy.distutils import lib2def' statement (line 419)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
    import_38038 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 419, 4), 'numpy.distutils')

    if (type(import_38038) is not StypyTypeError):

        if (import_38038 != 'pyd_module'):
            __import__(import_38038)
            sys_modules_38039 = sys.modules[import_38038]
            import_from_module(stypy.reporting.localization.Localization(__file__, 419, 4), 'numpy.distutils', sys_modules_38039.module_type_store, module_type_store, ['lib2def'])
            nest_module(stypy.reporting.localization.Localization(__file__, 419, 4), __file__, sys_modules_38039, sys_modules_38039.module_type_store, module_type_store)
        else:
            from numpy.distutils import lib2def

            import_from_module(stypy.reporting.localization.Localization(__file__, 419, 4), 'numpy.distutils', None, module_type_store, ['lib2def'], [lib2def])

    else:
        # Assigning a type to the variable 'numpy.distutils' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'numpy.distutils', import_38038)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')
    
    
    # Assigning a BinOp to a Name (line 421):
    
    # Assigning a BinOp to a Name (line 421):
    str_38040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 15), 'str', 'python%d%d.def')
    
    # Call to tuple(...): (line 421)
    # Processing the call arguments (line 421)
    
    # Obtaining the type of the subscript
    int_38042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 58), 'int')
    slice_38043 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 421, 40), None, int_38042, None)
    # Getting the type of 'sys' (line 421)
    sys_38044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 40), 'sys', False)
    # Obtaining the member 'version_info' of a type (line 421)
    version_info_38045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 40), sys_38044, 'version_info')
    # Obtaining the member '__getitem__' of a type (line 421)
    getitem___38046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 40), version_info_38045, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 421)
    subscript_call_result_38047 = invoke(stypy.reporting.localization.Localization(__file__, 421, 40), getitem___38046, slice_38043)
    
    # Processing the call keyword arguments (line 421)
    kwargs_38048 = {}
    # Getting the type of 'tuple' (line 421)
    tuple_38041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 34), 'tuple', False)
    # Calling tuple(args, kwargs) (line 421)
    tuple_call_result_38049 = invoke(stypy.reporting.localization.Localization(__file__, 421, 34), tuple_38041, *[subscript_call_result_38047], **kwargs_38048)
    
    # Applying the binary operator '%' (line 421)
    result_mod_38050 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 15), '%', str_38040, tuple_call_result_38049)
    
    # Assigning a type to the variable 'def_name' (line 421)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 4), 'def_name', result_mod_38050)
    
    # Assigning a Call to a Name (line 422):
    
    # Assigning a Call to a Name (line 422):
    
    # Call to join(...): (line 422)
    # Processing the call arguments (line 422)
    # Getting the type of 'sys' (line 422)
    sys_38054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 28), 'sys', False)
    # Obtaining the member 'prefix' of a type (line 422)
    prefix_38055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 28), sys_38054, 'prefix')
    str_38056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 40), 'str', 'libs')
    # Getting the type of 'def_name' (line 422)
    def_name_38057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 48), 'def_name', False)
    # Processing the call keyword arguments (line 422)
    kwargs_38058 = {}
    # Getting the type of 'os' (line 422)
    os_38051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 422)
    path_38052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 15), os_38051, 'path')
    # Obtaining the member 'join' of a type (line 422)
    join_38053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 15), path_38052, 'join')
    # Calling join(args, kwargs) (line 422)
    join_call_result_38059 = invoke(stypy.reporting.localization.Localization(__file__, 422, 15), join_38053, *[prefix_38055, str_38056, def_name_38057], **kwargs_38058)
    
    # Assigning a type to the variable 'def_file' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'def_file', join_call_result_38059)
    
    # Assigning a BinOp to a Name (line 423):
    
    # Assigning a BinOp to a Name (line 423):
    str_38060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 13), 'str', '%s %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 423)
    tuple_38061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 423)
    # Adding element type (line 423)
    # Getting the type of 'lib2def' (line 423)
    lib2def_38062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 24), 'lib2def')
    # Obtaining the member 'DEFAULT_NM' of a type (line 423)
    DEFAULT_NM_38063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 24), lib2def_38062, 'DEFAULT_NM')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 24), tuple_38061, DEFAULT_NM_38063)
    # Adding element type (line 423)
    # Getting the type of 'lib_file' (line 423)
    lib_file_38064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 44), 'lib_file')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 24), tuple_38061, lib_file_38064)
    
    # Applying the binary operator '%' (line 423)
    result_mod_38065 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 13), '%', str_38060, tuple_38061)
    
    # Assigning a type to the variable 'nm_cmd' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'nm_cmd', result_mod_38065)
    
    # Assigning a Call to a Name (line 424):
    
    # Assigning a Call to a Name (line 424):
    
    # Call to getnm(...): (line 424)
    # Processing the call arguments (line 424)
    # Getting the type of 'nm_cmd' (line 424)
    nm_cmd_38068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 30), 'nm_cmd', False)
    # Processing the call keyword arguments (line 424)
    kwargs_38069 = {}
    # Getting the type of 'lib2def' (line 424)
    lib2def_38066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 16), 'lib2def', False)
    # Obtaining the member 'getnm' of a type (line 424)
    getnm_38067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 16), lib2def_38066, 'getnm')
    # Calling getnm(args, kwargs) (line 424)
    getnm_call_result_38070 = invoke(stypy.reporting.localization.Localization(__file__, 424, 16), getnm_38067, *[nm_cmd_38068], **kwargs_38069)
    
    # Assigning a type to the variable 'nm_output' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'nm_output', getnm_call_result_38070)
    
    # Assigning a Call to a Tuple (line 425):
    
    # Assigning a Call to a Name:
    
    # Call to parse_nm(...): (line 425)
    # Processing the call arguments (line 425)
    # Getting the type of 'nm_output' (line 425)
    nm_output_38073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 36), 'nm_output', False)
    # Processing the call keyword arguments (line 425)
    kwargs_38074 = {}
    # Getting the type of 'lib2def' (line 425)
    lib2def_38071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 19), 'lib2def', False)
    # Obtaining the member 'parse_nm' of a type (line 425)
    parse_nm_38072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 19), lib2def_38071, 'parse_nm')
    # Calling parse_nm(args, kwargs) (line 425)
    parse_nm_call_result_38075 = invoke(stypy.reporting.localization.Localization(__file__, 425, 19), parse_nm_38072, *[nm_output_38073], **kwargs_38074)
    
    # Assigning a type to the variable 'call_assignment_36925' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'call_assignment_36925', parse_nm_call_result_38075)
    
    # Assigning a Call to a Name (line 425):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_38078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 4), 'int')
    # Processing the call keyword arguments
    kwargs_38079 = {}
    # Getting the type of 'call_assignment_36925' (line 425)
    call_assignment_36925_38076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'call_assignment_36925', False)
    # Obtaining the member '__getitem__' of a type (line 425)
    getitem___38077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 4), call_assignment_36925_38076, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_38080 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___38077, *[int_38078], **kwargs_38079)
    
    # Assigning a type to the variable 'call_assignment_36926' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'call_assignment_36926', getitem___call_result_38080)
    
    # Assigning a Name to a Name (line 425):
    # Getting the type of 'call_assignment_36926' (line 425)
    call_assignment_36926_38081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'call_assignment_36926')
    # Assigning a type to the variable 'dlist' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'dlist', call_assignment_36926_38081)
    
    # Assigning a Call to a Name (line 425):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_38084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 4), 'int')
    # Processing the call keyword arguments
    kwargs_38085 = {}
    # Getting the type of 'call_assignment_36925' (line 425)
    call_assignment_36925_38082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'call_assignment_36925', False)
    # Obtaining the member '__getitem__' of a type (line 425)
    getitem___38083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 4), call_assignment_36925_38082, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_38086 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___38083, *[int_38084], **kwargs_38085)
    
    # Assigning a type to the variable 'call_assignment_36927' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'call_assignment_36927', getitem___call_result_38086)
    
    # Assigning a Name to a Name (line 425):
    # Getting the type of 'call_assignment_36927' (line 425)
    call_assignment_36927_38087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'call_assignment_36927')
    # Assigning a type to the variable 'flist' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 11), 'flist', call_assignment_36927_38087)
    
    # Call to output_def(...): (line 426)
    # Processing the call arguments (line 426)
    # Getting the type of 'dlist' (line 426)
    dlist_38090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 23), 'dlist', False)
    # Getting the type of 'flist' (line 426)
    flist_38091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 30), 'flist', False)
    # Getting the type of 'lib2def' (line 426)
    lib2def_38092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 37), 'lib2def', False)
    # Obtaining the member 'DEF_HEADER' of a type (line 426)
    DEF_HEADER_38093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 37), lib2def_38092, 'DEF_HEADER')
    
    # Call to open(...): (line 426)
    # Processing the call arguments (line 426)
    # Getting the type of 'def_file' (line 426)
    def_file_38095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 62), 'def_file', False)
    str_38096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 72), 'str', 'w')
    # Processing the call keyword arguments (line 426)
    kwargs_38097 = {}
    # Getting the type of 'open' (line 426)
    open_38094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 57), 'open', False)
    # Calling open(args, kwargs) (line 426)
    open_call_result_38098 = invoke(stypy.reporting.localization.Localization(__file__, 426, 57), open_38094, *[def_file_38095, str_38096], **kwargs_38097)
    
    # Processing the call keyword arguments (line 426)
    kwargs_38099 = {}
    # Getting the type of 'lib2def' (line 426)
    lib2def_38088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'lib2def', False)
    # Obtaining the member 'output_def' of a type (line 426)
    output_def_38089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 4), lib2def_38088, 'output_def')
    # Calling output_def(args, kwargs) (line 426)
    output_def_call_result_38100 = invoke(stypy.reporting.localization.Localization(__file__, 426, 4), output_def_38089, *[dlist_38090, flist_38091, DEF_HEADER_38093, open_call_result_38098], **kwargs_38099)
    
    
    # Assigning a BinOp to a Name (line 428):
    
    # Assigning a BinOp to a Name (line 428):
    str_38101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 15), 'str', 'python%d%d.dll')
    
    # Call to tuple(...): (line 428)
    # Processing the call arguments (line 428)
    
    # Obtaining the type of the subscript
    int_38103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 58), 'int')
    slice_38104 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 428, 40), None, int_38103, None)
    # Getting the type of 'sys' (line 428)
    sys_38105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 40), 'sys', False)
    # Obtaining the member 'version_info' of a type (line 428)
    version_info_38106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 40), sys_38105, 'version_info')
    # Obtaining the member '__getitem__' of a type (line 428)
    getitem___38107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 40), version_info_38106, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 428)
    subscript_call_result_38108 = invoke(stypy.reporting.localization.Localization(__file__, 428, 40), getitem___38107, slice_38104)
    
    # Processing the call keyword arguments (line 428)
    kwargs_38109 = {}
    # Getting the type of 'tuple' (line 428)
    tuple_38102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 34), 'tuple', False)
    # Calling tuple(args, kwargs) (line 428)
    tuple_call_result_38110 = invoke(stypy.reporting.localization.Localization(__file__, 428, 34), tuple_38102, *[subscript_call_result_38108], **kwargs_38109)
    
    # Applying the binary operator '%' (line 428)
    result_mod_38111 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 15), '%', str_38101, tuple_call_result_38110)
    
    # Assigning a type to the variable 'dll_name' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'dll_name', result_mod_38111)
    
    # Assigning a Tuple to a Name (line 429):
    
    # Assigning a Tuple to a Name (line 429):
    
    # Obtaining an instance of the builtin type 'tuple' (line 429)
    tuple_38112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 429)
    # Adding element type (line 429)
    # Getting the type of 'dll_name' (line 429)
    dll_name_38113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'dll_name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 12), tuple_38112, dll_name_38113)
    # Adding element type (line 429)
    # Getting the type of 'def_file' (line 429)
    def_file_38114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 22), 'def_file')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 12), tuple_38112, def_file_38114)
    # Adding element type (line 429)
    # Getting the type of 'out_file' (line 429)
    out_file_38115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 32), 'out_file')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 12), tuple_38112, out_file_38115)
    
    # Assigning a type to the variable 'args' (line 429)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 4), 'args', tuple_38112)
    
    # Assigning a BinOp to a Name (line 430):
    
    # Assigning a BinOp to a Name (line 430):
    str_38116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 10), 'str', 'dlltool --dllname %s --def %s --output-lib %s')
    # Getting the type of 'args' (line 430)
    args_38117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 60), 'args')
    # Applying the binary operator '%' (line 430)
    result_mod_38118 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 10), '%', str_38116, args_38117)
    
    # Assigning a type to the variable 'cmd' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'cmd', result_mod_38118)
    
    # Assigning a Call to a Name (line 431):
    
    # Assigning a Call to a Name (line 431):
    
    # Call to system(...): (line 431)
    # Processing the call arguments (line 431)
    # Getting the type of 'cmd' (line 431)
    cmd_38121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 23), 'cmd', False)
    # Processing the call keyword arguments (line 431)
    kwargs_38122 = {}
    # Getting the type of 'os' (line 431)
    os_38119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 13), 'os', False)
    # Obtaining the member 'system' of a type (line 431)
    system_38120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 13), os_38119, 'system')
    # Calling system(args, kwargs) (line 431)
    system_call_result_38123 = invoke(stypy.reporting.localization.Localization(__file__, 431, 13), system_38120, *[cmd_38121], **kwargs_38122)
    
    # Assigning a type to the variable 'status' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'status', system_call_result_38123)
    
    # Getting the type of 'status' (line 433)
    status_38124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 7), 'status')
    # Testing the type of an if condition (line 433)
    if_condition_38125 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 433, 4), status_38124)
    # Assigning a type to the variable 'if_condition_38125' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 4), 'if_condition_38125', if_condition_38125)
    # SSA begins for if statement (line 433)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 434)
    # Processing the call arguments (line 434)
    str_38128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 17), 'str', 'Failed to build import library for gcc. Linking will fail.')
    # Processing the call keyword arguments (line 434)
    kwargs_38129 = {}
    # Getting the type of 'log' (line 434)
    log_38126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'log', False)
    # Obtaining the member 'warn' of a type (line 434)
    warn_38127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 8), log_38126, 'warn')
    # Calling warn(args, kwargs) (line 434)
    warn_call_result_38130 = invoke(stypy.reporting.localization.Localization(__file__, 434, 8), warn_38127, *[str_38128], **kwargs_38129)
    
    # SSA join for if statement (line 433)
    module_type_store = module_type_store.join_ssa_context()
    
    # Assigning a type to the variable 'stypy_return_type' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 4), 'stypy_return_type', types.NoneType)
    
    # ################# End of '_build_import_library_x86(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_build_import_library_x86' in the type store
    # Getting the type of 'stypy_return_type' (line 404)
    stypy_return_type_38131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38131)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_build_import_library_x86'
    return stypy_return_type_38131

# Assigning a type to the variable '_build_import_library_x86' (line 404)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 0), '_build_import_library_x86', _build_import_library_x86)

# Assigning a Dict to a Name (line 454):

# Assigning a Dict to a Name (line 454):

# Obtaining an instance of the builtin type 'dict' (line 454)
dict_38132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 23), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 454)

# Assigning a type to the variable '_MSVCRVER_TO_FULLVER' (line 454)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 0), '_MSVCRVER_TO_FULLVER', dict_38132)


# Getting the type of 'sys' (line 455)
sys_38133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 3), 'sys')
# Obtaining the member 'platform' of a type (line 455)
platform_38134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 3), sys_38133, 'platform')
str_38135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 19), 'str', 'win32')
# Applying the binary operator '==' (line 455)
result_eq_38136 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 3), '==', platform_38134, str_38135)

# Testing the type of an if condition (line 455)
if_condition_38137 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 455, 0), result_eq_38136)
# Assigning a type to the variable 'if_condition_38137' (line 455)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 0), 'if_condition_38137', if_condition_38137)
# SSA begins for if statement (line 455)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')


# SSA begins for try-except statement (line 456)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 457, 8))

# 'import msvcrt' statement (line 457)
import msvcrt

import_module(stypy.reporting.localization.Localization(__file__, 457, 8), 'msvcrt', msvcrt, module_type_store)


# Assigning a Str to a Subscript (line 460):

# Assigning a Str to a Subscript (line 460):
str_38138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 37), 'str', '8.0.50727.42')
# Getting the type of '_MSVCRVER_TO_FULLVER' (line 460)
_MSVCRVER_TO_FULLVER_38139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), '_MSVCRVER_TO_FULLVER')
str_38140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 29), 'str', '80')
# Storing an element on a container (line 460)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 8), _MSVCRVER_TO_FULLVER_38139, (str_38140, str_38138))

# Assigning a Str to a Subscript (line 461):

# Assigning a Str to a Subscript (line 461):
str_38141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 37), 'str', '9.0.21022.8')
# Getting the type of '_MSVCRVER_TO_FULLVER' (line 461)
_MSVCRVER_TO_FULLVER_38142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), '_MSVCRVER_TO_FULLVER')
str_38143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 29), 'str', '90')
# Storing an element on a container (line 461)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 461, 8), _MSVCRVER_TO_FULLVER_38142, (str_38143, str_38141))

# Assigning a Str to a Subscript (line 464):

# Assigning a Str to a Subscript (line 464):
str_38144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 38), 'str', '10.0.30319.460')
# Getting the type of '_MSVCRVER_TO_FULLVER' (line 464)
_MSVCRVER_TO_FULLVER_38145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), '_MSVCRVER_TO_FULLVER')
str_38146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 29), 'str', '100')
# Storing an element on a container (line 464)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 8), _MSVCRVER_TO_FULLVER_38145, (str_38146, str_38144))

# Type idiom detected: calculating its left and rigth part (line 465)
str_38147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 27), 'str', 'CRT_ASSEMBLY_VERSION')
# Getting the type of 'msvcrt' (line 465)
msvcrt_38148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 19), 'msvcrt')

(may_be_38149, more_types_in_union_38150) = may_provide_member(str_38147, msvcrt_38148)

if may_be_38149:

    if more_types_in_union_38150:
        # Runtime conditional SSA (line 465)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
    else:
        module_type_store = module_type_store

    # Assigning a type to the variable 'msvcrt' (line 465)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'msvcrt', remove_not_member_provider_from_union(msvcrt_38148, 'CRT_ASSEMBLY_VERSION'))
    
    # Assigning a Call to a Tuple (line 466):
    
    # Assigning a Call to a Name:
    
    # Call to split(...): (line 466)
    # Processing the call arguments (line 466)
    str_38154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 67), 'str', '.')
    int_38155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 72), 'int')
    # Processing the call keyword arguments (line 466)
    kwargs_38156 = {}
    # Getting the type of 'msvcrt' (line 466)
    msvcrt_38151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 33), 'msvcrt', False)
    # Obtaining the member 'CRT_ASSEMBLY_VERSION' of a type (line 466)
    CRT_ASSEMBLY_VERSION_38152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 33), msvcrt_38151, 'CRT_ASSEMBLY_VERSION')
    # Obtaining the member 'split' of a type (line 466)
    split_38153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 33), CRT_ASSEMBLY_VERSION_38152, 'split')
    # Calling split(args, kwargs) (line 466)
    split_call_result_38157 = invoke(stypy.reporting.localization.Localization(__file__, 466, 33), split_38153, *[str_38154, int_38155], **kwargs_38156)
    
    # Assigning a type to the variable 'call_assignment_36928' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'call_assignment_36928', split_call_result_38157)
    
    # Assigning a Call to a Name (line 466):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_38160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 12), 'int')
    # Processing the call keyword arguments
    kwargs_38161 = {}
    # Getting the type of 'call_assignment_36928' (line 466)
    call_assignment_36928_38158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'call_assignment_36928', False)
    # Obtaining the member '__getitem__' of a type (line 466)
    getitem___38159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 12), call_assignment_36928_38158, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_38162 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___38159, *[int_38160], **kwargs_38161)
    
    # Assigning a type to the variable 'call_assignment_36929' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'call_assignment_36929', getitem___call_result_38162)
    
    # Assigning a Name to a Name (line 466):
    # Getting the type of 'call_assignment_36929' (line 466)
    call_assignment_36929_38163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'call_assignment_36929')
    # Assigning a type to the variable 'major' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'major', call_assignment_36929_38163)
    
    # Assigning a Call to a Name (line 466):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_38166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 12), 'int')
    # Processing the call keyword arguments
    kwargs_38167 = {}
    # Getting the type of 'call_assignment_36928' (line 466)
    call_assignment_36928_38164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'call_assignment_36928', False)
    # Obtaining the member '__getitem__' of a type (line 466)
    getitem___38165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 12), call_assignment_36928_38164, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_38168 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___38165, *[int_38166], **kwargs_38167)
    
    # Assigning a type to the variable 'call_assignment_36930' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'call_assignment_36930', getitem___call_result_38168)
    
    # Assigning a Name to a Name (line 466):
    # Getting the type of 'call_assignment_36930' (line 466)
    call_assignment_36930_38169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'call_assignment_36930')
    # Assigning a type to the variable 'minor' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 19), 'minor', call_assignment_36930_38169)
    
    # Assigning a Call to a Name (line 466):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_38172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 12), 'int')
    # Processing the call keyword arguments
    kwargs_38173 = {}
    # Getting the type of 'call_assignment_36928' (line 466)
    call_assignment_36928_38170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'call_assignment_36928', False)
    # Obtaining the member '__getitem__' of a type (line 466)
    getitem___38171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 12), call_assignment_36928_38170, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_38174 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___38171, *[int_38172], **kwargs_38173)
    
    # Assigning a type to the variable 'call_assignment_36931' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'call_assignment_36931', getitem___call_result_38174)
    
    # Assigning a Name to a Name (line 466):
    # Getting the type of 'call_assignment_36931' (line 466)
    call_assignment_36931_38175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'call_assignment_36931')
    # Assigning a type to the variable 'rest' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 26), 'rest', call_assignment_36931_38175)
    
    # Assigning a Attribute to a Subscript (line 467):
    
    # Assigning a Attribute to a Subscript (line 467):
    # Getting the type of 'msvcrt' (line 467)
    msvcrt_38176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 50), 'msvcrt')
    # Obtaining the member 'CRT_ASSEMBLY_VERSION' of a type (line 467)
    CRT_ASSEMBLY_VERSION_38177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 50), msvcrt_38176, 'CRT_ASSEMBLY_VERSION')
    # Getting the type of '_MSVCRVER_TO_FULLVER' (line 467)
    _MSVCRVER_TO_FULLVER_38178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 12), '_MSVCRVER_TO_FULLVER')
    # Getting the type of 'major' (line 467)
    major_38179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 33), 'major')
    # Getting the type of 'minor' (line 467)
    minor_38180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 41), 'minor')
    # Applying the binary operator '+' (line 467)
    result_add_38181 = python_operator(stypy.reporting.localization.Localization(__file__, 467, 33), '+', major_38179, minor_38180)
    
    # Storing an element on a container (line 467)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 467, 12), _MSVCRVER_TO_FULLVER_38178, (result_add_38181, CRT_ASSEMBLY_VERSION_38177))
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 468, 12), module_type_store, 'major')
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 468, 12), module_type_store, 'minor')
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 468, 12), module_type_store, 'rest')

    if more_types_in_union_38150:
        # SSA join for if statement (line 465)
        module_type_store = module_type_store.join_ssa_context()



# SSA branch for the except part of a try statement (line 456)
# SSA branch for the except 'ImportError' branch of a try statement (line 456)
module_type_store.open_ssa_branch('except')

# Call to warn(...): (line 473)
# Processing the call arguments (line 473)
str_38184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 17), 'str', 'Cannot import msvcrt: using manifest will not be possible')
# Processing the call keyword arguments (line 473)
kwargs_38185 = {}
# Getting the type of 'log' (line 473)
log_38182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'log', False)
# Obtaining the member 'warn' of a type (line 473)
warn_38183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 8), log_38182, 'warn')
# Calling warn(args, kwargs) (line 473)
warn_call_result_38186 = invoke(stypy.reporting.localization.Localization(__file__, 473, 8), warn_38183, *[str_38184], **kwargs_38185)

# SSA join for try-except statement (line 456)
module_type_store = module_type_store.join_ssa_context()

# SSA join for if statement (line 455)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def msvc_manifest_xml(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'msvc_manifest_xml'
    module_type_store = module_type_store.open_function_context('msvc_manifest_xml', 475, 0, False)
    
    # Passed parameters checking function
    msvc_manifest_xml.stypy_localization = localization
    msvc_manifest_xml.stypy_type_of_self = None
    msvc_manifest_xml.stypy_type_store = module_type_store
    msvc_manifest_xml.stypy_function_name = 'msvc_manifest_xml'
    msvc_manifest_xml.stypy_param_names_list = ['maj', 'min']
    msvc_manifest_xml.stypy_varargs_param_name = None
    msvc_manifest_xml.stypy_kwargs_param_name = None
    msvc_manifest_xml.stypy_call_defaults = defaults
    msvc_manifest_xml.stypy_call_varargs = varargs
    msvc_manifest_xml.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'msvc_manifest_xml', ['maj', 'min'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'msvc_manifest_xml', localization, ['maj', 'min'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'msvc_manifest_xml(...)' code ##################

    str_38187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, (-1)), 'str', 'Given a major and minor version of the MSVCR, returns the\n    corresponding XML file.')
    
    
    # SSA begins for try-except statement (line 478)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 479):
    
    # Assigning a Subscript to a Name (line 479):
    
    # Obtaining the type of the subscript
    
    # Call to str(...): (line 479)
    # Processing the call arguments (line 479)
    # Getting the type of 'maj' (line 479)
    maj_38189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 43), 'maj', False)
    int_38190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 49), 'int')
    # Applying the binary operator '*' (line 479)
    result_mul_38191 = python_operator(stypy.reporting.localization.Localization(__file__, 479, 43), '*', maj_38189, int_38190)
    
    # Getting the type of 'min' (line 479)
    min_38192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 54), 'min', False)
    # Applying the binary operator '+' (line 479)
    result_add_38193 = python_operator(stypy.reporting.localization.Localization(__file__, 479, 43), '+', result_mul_38191, min_38192)
    
    # Processing the call keyword arguments (line 479)
    kwargs_38194 = {}
    # Getting the type of 'str' (line 479)
    str_38188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 39), 'str', False)
    # Calling str(args, kwargs) (line 479)
    str_call_result_38195 = invoke(stypy.reporting.localization.Localization(__file__, 479, 39), str_38188, *[result_add_38193], **kwargs_38194)
    
    # Getting the type of '_MSVCRVER_TO_FULLVER' (line 479)
    _MSVCRVER_TO_FULLVER_38196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 18), '_MSVCRVER_TO_FULLVER')
    # Obtaining the member '__getitem__' of a type (line 479)
    getitem___38197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 18), _MSVCRVER_TO_FULLVER_38196, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 479)
    subscript_call_result_38198 = invoke(stypy.reporting.localization.Localization(__file__, 479, 18), getitem___38197, str_call_result_38195)
    
    # Assigning a type to the variable 'fullver' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'fullver', subscript_call_result_38198)
    # SSA branch for the except part of a try statement (line 478)
    # SSA branch for the except 'KeyError' branch of a try statement (line 478)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 481)
    # Processing the call arguments (line 481)
    str_38200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 25), 'str', 'Version %d,%d of MSVCRT not supported yet')
    
    # Obtaining an instance of the builtin type 'tuple' (line 482)
    tuple_38201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 482)
    # Adding element type (line 482)
    # Getting the type of 'maj' (line 482)
    maj_38202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 26), 'maj', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 26), tuple_38201, maj_38202)
    # Adding element type (line 482)
    # Getting the type of 'min' (line 482)
    min_38203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 31), 'min', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 26), tuple_38201, min_38203)
    
    # Applying the binary operator '%' (line 481)
    result_mod_38204 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 25), '%', str_38200, tuple_38201)
    
    # Processing the call keyword arguments (line 481)
    kwargs_38205 = {}
    # Getting the type of 'ValueError' (line 481)
    ValueError_38199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 481)
    ValueError_call_result_38206 = invoke(stypy.reporting.localization.Localization(__file__, 481, 14), ValueError_38199, *[result_mod_38204], **kwargs_38205)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 481, 8), ValueError_call_result_38206, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 478)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Str to a Name (line 489):
    
    # Assigning a Str to a Name (line 489):
    str_38207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, (-1)), 'str', '<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">\n  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">\n    <security>\n      <requestedPrivileges>\n        <requestedExecutionLevel level="asInvoker" uiAccess="false"></requestedExecutionLevel>\n      </requestedPrivileges>\n    </security>\n  </trustInfo>\n  <dependency>\n    <dependentAssembly>\n      <assemblyIdentity type="win32" name="Microsoft.VC%(maj)d%(min)d.CRT" version="%(fullver)s" processorArchitecture="*" publicKeyToken="1fc8b3b9a1e18e3b"></assemblyIdentity>\n    </dependentAssembly>\n  </dependency>\n</assembly>')
    # Assigning a type to the variable 'template' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 4), 'template', str_38207)
    # Getting the type of 'template' (line 505)
    template_38208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 11), 'template')
    
    # Obtaining an instance of the builtin type 'dict' (line 505)
    dict_38209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 22), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 505)
    # Adding element type (key, value) (line 505)
    str_38210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 23), 'str', 'fullver')
    # Getting the type of 'fullver' (line 505)
    fullver_38211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 34), 'fullver')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 22), dict_38209, (str_38210, fullver_38211))
    # Adding element type (key, value) (line 505)
    str_38212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 43), 'str', 'maj')
    # Getting the type of 'maj' (line 505)
    maj_38213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 50), 'maj')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 22), dict_38209, (str_38212, maj_38213))
    # Adding element type (key, value) (line 505)
    str_38214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 55), 'str', 'min')
    # Getting the type of 'min' (line 505)
    min_38215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 62), 'min')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 22), dict_38209, (str_38214, min_38215))
    
    # Applying the binary operator '%' (line 505)
    result_mod_38216 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 11), '%', template_38208, dict_38209)
    
    # Assigning a type to the variable 'stypy_return_type' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'stypy_return_type', result_mod_38216)
    
    # ################# End of 'msvc_manifest_xml(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'msvc_manifest_xml' in the type store
    # Getting the type of 'stypy_return_type' (line 475)
    stypy_return_type_38217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38217)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'msvc_manifest_xml'
    return stypy_return_type_38217

# Assigning a type to the variable 'msvc_manifest_xml' (line 475)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 0), 'msvc_manifest_xml', msvc_manifest_xml)

@norecursion
def manifest_rc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_38218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 27), 'str', 'dll')
    defaults = [str_38218]
    # Create a new context for function 'manifest_rc'
    module_type_store = module_type_store.open_function_context('manifest_rc', 507, 0, False)
    
    # Passed parameters checking function
    manifest_rc.stypy_localization = localization
    manifest_rc.stypy_type_of_self = None
    manifest_rc.stypy_type_store = module_type_store
    manifest_rc.stypy_function_name = 'manifest_rc'
    manifest_rc.stypy_param_names_list = ['name', 'type']
    manifest_rc.stypy_varargs_param_name = None
    manifest_rc.stypy_kwargs_param_name = None
    manifest_rc.stypy_call_defaults = defaults
    manifest_rc.stypy_call_varargs = varargs
    manifest_rc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'manifest_rc', ['name', 'type'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'manifest_rc', localization, ['name', 'type'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'manifest_rc(...)' code ##################

    str_38219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, (-1)), 'str', "Return the rc file used to generate the res file which will be embedded\n    as manifest for given manifest file name, of given type ('dll' or\n    'exe').\n\n    Parameters\n    ----------\n    name : str\n            name of the manifest file to embed\n    type : str {'dll', 'exe'}\n            type of the binary which will embed the manifest\n\n    ")
    
    
    # Getting the type of 'type' (line 520)
    type_38220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 7), 'type')
    str_38221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 15), 'str', 'dll')
    # Applying the binary operator '==' (line 520)
    result_eq_38222 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 7), '==', type_38220, str_38221)
    
    # Testing the type of an if condition (line 520)
    if_condition_38223 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 520, 4), result_eq_38222)
    # Assigning a type to the variable 'if_condition_38223' (line 520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 4), 'if_condition_38223', if_condition_38223)
    # SSA begins for if statement (line 520)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 521):
    
    # Assigning a Num to a Name (line 521):
    int_38224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 17), 'int')
    # Assigning a type to the variable 'rctype' (line 521)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'rctype', int_38224)
    # SSA branch for the else part of an if statement (line 520)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'type' (line 522)
    type_38225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 9), 'type')
    str_38226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 17), 'str', 'exe')
    # Applying the binary operator '==' (line 522)
    result_eq_38227 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 9), '==', type_38225, str_38226)
    
    # Testing the type of an if condition (line 522)
    if_condition_38228 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 522, 9), result_eq_38227)
    # Assigning a type to the variable 'if_condition_38228' (line 522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 9), 'if_condition_38228', if_condition_38228)
    # SSA begins for if statement (line 522)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 523):
    
    # Assigning a Num to a Name (line 523):
    int_38229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 17), 'int')
    # Assigning a type to the variable 'rctype' (line 523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 8), 'rctype', int_38229)
    # SSA branch for the else part of an if statement (line 522)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 525)
    # Processing the call arguments (line 525)
    str_38231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 25), 'str', 'Type %s not supported')
    # Getting the type of 'type' (line 525)
    type_38232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 51), 'type', False)
    # Applying the binary operator '%' (line 525)
    result_mod_38233 = python_operator(stypy.reporting.localization.Localization(__file__, 525, 25), '%', str_38231, type_38232)
    
    # Processing the call keyword arguments (line 525)
    kwargs_38234 = {}
    # Getting the type of 'ValueError' (line 525)
    ValueError_38230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 525)
    ValueError_call_result_38235 = invoke(stypy.reporting.localization.Localization(__file__, 525, 14), ValueError_38230, *[result_mod_38233], **kwargs_38234)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 525, 8), ValueError_call_result_38235, 'raise parameter', BaseException)
    # SSA join for if statement (line 522)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 520)
    module_type_store = module_type_store.join_ssa_context()
    
    str_38236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, (-1)), 'str', '#include "winuser.h"\n%d RT_MANIFEST %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 529)
    tuple_38237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 529)
    # Adding element type (line 529)
    # Getting the type of 'rctype' (line 529)
    rctype_38238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 24), 'rctype')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 24), tuple_38237, rctype_38238)
    # Adding element type (line 529)
    # Getting the type of 'name' (line 529)
    name_38239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 32), 'name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 24), tuple_38237, name_38239)
    
    # Applying the binary operator '%' (line 529)
    result_mod_38240 = python_operator(stypy.reporting.localization.Localization(__file__, 529, (-1)), '%', str_38236, tuple_38237)
    
    # Assigning a type to the variable 'stypy_return_type' (line 527)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 4), 'stypy_return_type', result_mod_38240)
    
    # ################# End of 'manifest_rc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'manifest_rc' in the type store
    # Getting the type of 'stypy_return_type' (line 507)
    stypy_return_type_38241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38241)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'manifest_rc'
    return stypy_return_type_38241

# Assigning a type to the variable 'manifest_rc' (line 507)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 0), 'manifest_rc', manifest_rc)

@norecursion
def check_embedded_msvcr_match_linked(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_embedded_msvcr_match_linked'
    module_type_store = module_type_store.open_function_context('check_embedded_msvcr_match_linked', 531, 0, False)
    
    # Passed parameters checking function
    check_embedded_msvcr_match_linked.stypy_localization = localization
    check_embedded_msvcr_match_linked.stypy_type_of_self = None
    check_embedded_msvcr_match_linked.stypy_type_store = module_type_store
    check_embedded_msvcr_match_linked.stypy_function_name = 'check_embedded_msvcr_match_linked'
    check_embedded_msvcr_match_linked.stypy_param_names_list = ['msver']
    check_embedded_msvcr_match_linked.stypy_varargs_param_name = None
    check_embedded_msvcr_match_linked.stypy_kwargs_param_name = None
    check_embedded_msvcr_match_linked.stypy_call_defaults = defaults
    check_embedded_msvcr_match_linked.stypy_call_varargs = varargs
    check_embedded_msvcr_match_linked.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_embedded_msvcr_match_linked', ['msver'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_embedded_msvcr_match_linked', localization, ['msver'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_embedded_msvcr_match_linked(...)' code ##################

    str_38242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 4), 'str', 'msver is the ms runtime version used for the MANIFEST.')
    
    # Assigning a Call to a Name (line 535):
    
    # Assigning a Call to a Name (line 535):
    
    # Call to msvc_runtime_library(...): (line 535)
    # Processing the call keyword arguments (line 535)
    kwargs_38244 = {}
    # Getting the type of 'msvc_runtime_library' (line 535)
    msvc_runtime_library_38243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 12), 'msvc_runtime_library', False)
    # Calling msvc_runtime_library(args, kwargs) (line 535)
    msvc_runtime_library_call_result_38245 = invoke(stypy.reporting.localization.Localization(__file__, 535, 12), msvc_runtime_library_38243, *[], **kwargs_38244)
    
    # Assigning a type to the variable 'msvcv' (line 535)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'msvcv', msvc_runtime_library_call_result_38245)
    
    # Getting the type of 'msvcv' (line 536)
    msvcv_38246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 7), 'msvcv')
    # Testing the type of an if condition (line 536)
    if_condition_38247 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 536, 4), msvcv_38246)
    # Assigning a type to the variable 'if_condition_38247' (line 536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'if_condition_38247', if_condition_38247)
    # SSA begins for if statement (line 536)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Evaluating assert statement condition
    
    # Call to startswith(...): (line 537)
    # Processing the call arguments (line 537)
    str_38250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 32), 'str', 'msvcr')
    # Processing the call keyword arguments (line 537)
    kwargs_38251 = {}
    # Getting the type of 'msvcv' (line 537)
    msvcv_38248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 15), 'msvcv', False)
    # Obtaining the member 'startswith' of a type (line 537)
    startswith_38249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 15), msvcv_38248, 'startswith')
    # Calling startswith(args, kwargs) (line 537)
    startswith_call_result_38252 = invoke(stypy.reporting.localization.Localization(__file__, 537, 15), startswith_38249, *[str_38250], **kwargs_38251)
    
    
    # Assigning a Call to a Name (line 540):
    
    # Assigning a Call to a Name (line 540):
    
    # Call to int(...): (line 540)
    # Processing the call arguments (line 540)
    
    # Obtaining the type of the subscript
    int_38254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 24), 'int')
    int_38255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 26), 'int')
    slice_38256 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 540, 18), int_38254, int_38255, None)
    # Getting the type of 'msvcv' (line 540)
    msvcv_38257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 18), 'msvcv', False)
    # Obtaining the member '__getitem__' of a type (line 540)
    getitem___38258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 18), msvcv_38257, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 540)
    subscript_call_result_38259 = invoke(stypy.reporting.localization.Localization(__file__, 540, 18), getitem___38258, slice_38256)
    
    # Processing the call keyword arguments (line 540)
    kwargs_38260 = {}
    # Getting the type of 'int' (line 540)
    int_38253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 14), 'int', False)
    # Calling int(args, kwargs) (line 540)
    int_call_result_38261 = invoke(stypy.reporting.localization.Localization(__file__, 540, 14), int_38253, *[subscript_call_result_38259], **kwargs_38260)
    
    # Assigning a type to the variable 'maj' (line 540)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'maj', int_call_result_38261)
    
    
    
    # Getting the type of 'maj' (line 541)
    maj_38262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 15), 'maj')
    
    # Call to int(...): (line 541)
    # Processing the call arguments (line 541)
    # Getting the type of 'msver' (line 541)
    msver_38264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 26), 'msver', False)
    # Processing the call keyword arguments (line 541)
    kwargs_38265 = {}
    # Getting the type of 'int' (line 541)
    int_38263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 22), 'int', False)
    # Calling int(args, kwargs) (line 541)
    int_call_result_38266 = invoke(stypy.reporting.localization.Localization(__file__, 541, 22), int_38263, *[msver_38264], **kwargs_38265)
    
    # Applying the binary operator '==' (line 541)
    result_eq_38267 = python_operator(stypy.reporting.localization.Localization(__file__, 541, 15), '==', maj_38262, int_call_result_38266)
    
    # Applying the 'not' unary operator (line 541)
    result_not__38268 = python_operator(stypy.reporting.localization.Localization(__file__, 541, 11), 'not', result_eq_38267)
    
    # Testing the type of an if condition (line 541)
    if_condition_38269 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 541, 8), result_not__38268)
    # Assigning a type to the variable 'if_condition_38269' (line 541)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 8), 'if_condition_38269', if_condition_38269)
    # SSA begins for if statement (line 541)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 542)
    # Processing the call arguments (line 542)
    str_38271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 18), 'str', 'Discrepancy between linked msvcr (%d) and the one about to be embedded (%d)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 545)
    tuple_38272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 545)
    # Adding element type (line 545)
    
    # Call to int(...): (line 545)
    # Processing the call arguments (line 545)
    # Getting the type of 'msver' (line 545)
    msver_38274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 32), 'msver', False)
    # Processing the call keyword arguments (line 545)
    kwargs_38275 = {}
    # Getting the type of 'int' (line 545)
    int_38273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 28), 'int', False)
    # Calling int(args, kwargs) (line 545)
    int_call_result_38276 = invoke(stypy.reporting.localization.Localization(__file__, 545, 28), int_38273, *[msver_38274], **kwargs_38275)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 545, 28), tuple_38272, int_call_result_38276)
    # Adding element type (line 545)
    # Getting the type of 'maj' (line 545)
    maj_38277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 40), 'maj', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 545, 28), tuple_38272, maj_38277)
    
    # Applying the binary operator '%' (line 543)
    result_mod_38278 = python_operator(stypy.reporting.localization.Localization(__file__, 543, 18), '%', str_38271, tuple_38272)
    
    # Processing the call keyword arguments (line 542)
    kwargs_38279 = {}
    # Getting the type of 'ValueError' (line 542)
    ValueError_38270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 542)
    ValueError_call_result_38280 = invoke(stypy.reporting.localization.Localization(__file__, 542, 18), ValueError_38270, *[result_mod_38278], **kwargs_38279)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 542, 12), ValueError_call_result_38280, 'raise parameter', BaseException)
    # SSA join for if statement (line 541)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 536)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'check_embedded_msvcr_match_linked(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_embedded_msvcr_match_linked' in the type store
    # Getting the type of 'stypy_return_type' (line 531)
    stypy_return_type_38281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38281)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_embedded_msvcr_match_linked'
    return stypy_return_type_38281

# Assigning a type to the variable 'check_embedded_msvcr_match_linked' (line 531)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 0), 'check_embedded_msvcr_match_linked', check_embedded_msvcr_match_linked)

@norecursion
def configtest_name(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'configtest_name'
    module_type_store = module_type_store.open_function_context('configtest_name', 547, 0, False)
    
    # Passed parameters checking function
    configtest_name.stypy_localization = localization
    configtest_name.stypy_type_of_self = None
    configtest_name.stypy_type_store = module_type_store
    configtest_name.stypy_function_name = 'configtest_name'
    configtest_name.stypy_param_names_list = ['config']
    configtest_name.stypy_varargs_param_name = None
    configtest_name.stypy_kwargs_param_name = None
    configtest_name.stypy_call_defaults = defaults
    configtest_name.stypy_call_varargs = varargs
    configtest_name.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'configtest_name', ['config'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'configtest_name', localization, ['config'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'configtest_name(...)' code ##################

    
    # Assigning a Call to a Name (line 548):
    
    # Assigning a Call to a Name (line 548):
    
    # Call to basename(...): (line 548)
    # Processing the call arguments (line 548)
    
    # Call to _gen_temp_sourcefile(...): (line 548)
    # Processing the call arguments (line 548)
    str_38287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 56), 'str', 'yo')
    
    # Obtaining an instance of the builtin type 'list' (line 548)
    list_38288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 62), 'list')
    # Adding type elements to the builtin type 'list' instance (line 548)
    
    str_38289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 66), 'str', 'c')
    # Processing the call keyword arguments (line 548)
    kwargs_38290 = {}
    # Getting the type of 'config' (line 548)
    config_38285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 28), 'config', False)
    # Obtaining the member '_gen_temp_sourcefile' of a type (line 548)
    _gen_temp_sourcefile_38286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 28), config_38285, '_gen_temp_sourcefile')
    # Calling _gen_temp_sourcefile(args, kwargs) (line 548)
    _gen_temp_sourcefile_call_result_38291 = invoke(stypy.reporting.localization.Localization(__file__, 548, 28), _gen_temp_sourcefile_38286, *[str_38287, list_38288, str_38289], **kwargs_38290)
    
    # Processing the call keyword arguments (line 548)
    kwargs_38292 = {}
    # Getting the type of 'os' (line 548)
    os_38282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 548)
    path_38283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 11), os_38282, 'path')
    # Obtaining the member 'basename' of a type (line 548)
    basename_38284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 11), path_38283, 'basename')
    # Calling basename(args, kwargs) (line 548)
    basename_call_result_38293 = invoke(stypy.reporting.localization.Localization(__file__, 548, 11), basename_38284, *[_gen_temp_sourcefile_call_result_38291], **kwargs_38292)
    
    # Assigning a type to the variable 'base' (line 548)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 4), 'base', basename_call_result_38293)
    
    # Obtaining the type of the subscript
    int_38294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 34), 'int')
    
    # Call to splitext(...): (line 549)
    # Processing the call arguments (line 549)
    # Getting the type of 'base' (line 549)
    base_38298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 28), 'base', False)
    # Processing the call keyword arguments (line 549)
    kwargs_38299 = {}
    # Getting the type of 'os' (line 549)
    os_38295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 549)
    path_38296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 11), os_38295, 'path')
    # Obtaining the member 'splitext' of a type (line 549)
    splitext_38297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 11), path_38296, 'splitext')
    # Calling splitext(args, kwargs) (line 549)
    splitext_call_result_38300 = invoke(stypy.reporting.localization.Localization(__file__, 549, 11), splitext_38297, *[base_38298], **kwargs_38299)
    
    # Obtaining the member '__getitem__' of a type (line 549)
    getitem___38301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 11), splitext_call_result_38300, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 549)
    subscript_call_result_38302 = invoke(stypy.reporting.localization.Localization(__file__, 549, 11), getitem___38301, int_38294)
    
    # Assigning a type to the variable 'stypy_return_type' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'stypy_return_type', subscript_call_result_38302)
    
    # ################# End of 'configtest_name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configtest_name' in the type store
    # Getting the type of 'stypy_return_type' (line 547)
    stypy_return_type_38303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38303)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configtest_name'
    return stypy_return_type_38303

# Assigning a type to the variable 'configtest_name' (line 547)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 0), 'configtest_name', configtest_name)

@norecursion
def manifest_name(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'manifest_name'
    module_type_store = module_type_store.open_function_context('manifest_name', 551, 0, False)
    
    # Passed parameters checking function
    manifest_name.stypy_localization = localization
    manifest_name.stypy_type_of_self = None
    manifest_name.stypy_type_store = module_type_store
    manifest_name.stypy_function_name = 'manifest_name'
    manifest_name.stypy_param_names_list = ['config']
    manifest_name.stypy_varargs_param_name = None
    manifest_name.stypy_kwargs_param_name = None
    manifest_name.stypy_call_defaults = defaults
    manifest_name.stypy_call_varargs = varargs
    manifest_name.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'manifest_name', ['config'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'manifest_name', localization, ['config'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'manifest_name(...)' code ##################

    
    # Assigning a Call to a Name (line 553):
    
    # Assigning a Call to a Name (line 553):
    
    # Call to configtest_name(...): (line 553)
    # Processing the call arguments (line 553)
    # Getting the type of 'config' (line 553)
    config_38305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 27), 'config', False)
    # Processing the call keyword arguments (line 553)
    kwargs_38306 = {}
    # Getting the type of 'configtest_name' (line 553)
    configtest_name_38304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 11), 'configtest_name', False)
    # Calling configtest_name(args, kwargs) (line 553)
    configtest_name_call_result_38307 = invoke(stypy.reporting.localization.Localization(__file__, 553, 11), configtest_name_38304, *[config_38305], **kwargs_38306)
    
    # Assigning a type to the variable 'root' (line 553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'root', configtest_name_call_result_38307)
    
    # Assigning a Attribute to a Name (line 554):
    
    # Assigning a Attribute to a Name (line 554):
    # Getting the type of 'config' (line 554)
    config_38308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 12), 'config')
    # Obtaining the member 'compiler' of a type (line 554)
    compiler_38309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 12), config_38308, 'compiler')
    # Obtaining the member 'exe_extension' of a type (line 554)
    exe_extension_38310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 12), compiler_38309, 'exe_extension')
    # Assigning a type to the variable 'exext' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'exext', exe_extension_38310)
    # Getting the type of 'root' (line 555)
    root_38311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 11), 'root')
    # Getting the type of 'exext' (line 555)
    exext_38312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 18), 'exext')
    # Applying the binary operator '+' (line 555)
    result_add_38313 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 11), '+', root_38311, exext_38312)
    
    str_38314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 26), 'str', '.manifest')
    # Applying the binary operator '+' (line 555)
    result_add_38315 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 24), '+', result_add_38313, str_38314)
    
    # Assigning a type to the variable 'stypy_return_type' (line 555)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 4), 'stypy_return_type', result_add_38315)
    
    # ################# End of 'manifest_name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'manifest_name' in the type store
    # Getting the type of 'stypy_return_type' (line 551)
    stypy_return_type_38316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38316)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'manifest_name'
    return stypy_return_type_38316

# Assigning a type to the variable 'manifest_name' (line 551)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 0), 'manifest_name', manifest_name)

@norecursion
def rc_name(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'rc_name'
    module_type_store = module_type_store.open_function_context('rc_name', 557, 0, False)
    
    # Passed parameters checking function
    rc_name.stypy_localization = localization
    rc_name.stypy_type_of_self = None
    rc_name.stypy_type_store = module_type_store
    rc_name.stypy_function_name = 'rc_name'
    rc_name.stypy_param_names_list = ['config']
    rc_name.stypy_varargs_param_name = None
    rc_name.stypy_kwargs_param_name = None
    rc_name.stypy_call_defaults = defaults
    rc_name.stypy_call_varargs = varargs
    rc_name.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rc_name', ['config'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rc_name', localization, ['config'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rc_name(...)' code ##################

    
    # Assigning a Call to a Name (line 559):
    
    # Assigning a Call to a Name (line 559):
    
    # Call to configtest_name(...): (line 559)
    # Processing the call arguments (line 559)
    # Getting the type of 'config' (line 559)
    config_38318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 27), 'config', False)
    # Processing the call keyword arguments (line 559)
    kwargs_38319 = {}
    # Getting the type of 'configtest_name' (line 559)
    configtest_name_38317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 11), 'configtest_name', False)
    # Calling configtest_name(args, kwargs) (line 559)
    configtest_name_call_result_38320 = invoke(stypy.reporting.localization.Localization(__file__, 559, 11), configtest_name_38317, *[config_38318], **kwargs_38319)
    
    # Assigning a type to the variable 'root' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'root', configtest_name_call_result_38320)
    # Getting the type of 'root' (line 560)
    root_38321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 11), 'root')
    str_38322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 18), 'str', '.rc')
    # Applying the binary operator '+' (line 560)
    result_add_38323 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 11), '+', root_38321, str_38322)
    
    # Assigning a type to the variable 'stypy_return_type' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'stypy_return_type', result_add_38323)
    
    # ################# End of 'rc_name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rc_name' in the type store
    # Getting the type of 'stypy_return_type' (line 557)
    stypy_return_type_38324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38324)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rc_name'
    return stypy_return_type_38324

# Assigning a type to the variable 'rc_name' (line 557)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 0), 'rc_name', rc_name)

@norecursion
def generate_manifest(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'generate_manifest'
    module_type_store = module_type_store.open_function_context('generate_manifest', 562, 0, False)
    
    # Passed parameters checking function
    generate_manifest.stypy_localization = localization
    generate_manifest.stypy_type_of_self = None
    generate_manifest.stypy_type_store = module_type_store
    generate_manifest.stypy_function_name = 'generate_manifest'
    generate_manifest.stypy_param_names_list = ['config']
    generate_manifest.stypy_varargs_param_name = None
    generate_manifest.stypy_kwargs_param_name = None
    generate_manifest.stypy_call_defaults = defaults
    generate_manifest.stypy_call_varargs = varargs
    generate_manifest.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'generate_manifest', ['config'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'generate_manifest', localization, ['config'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'generate_manifest(...)' code ##################

    
    # Assigning a Call to a Name (line 563):
    
    # Assigning a Call to a Name (line 563):
    
    # Call to get_build_msvc_version(...): (line 563)
    # Processing the call keyword arguments (line 563)
    kwargs_38326 = {}
    # Getting the type of 'get_build_msvc_version' (line 563)
    get_build_msvc_version_38325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 12), 'get_build_msvc_version', False)
    # Calling get_build_msvc_version(args, kwargs) (line 563)
    get_build_msvc_version_call_result_38327 = invoke(stypy.reporting.localization.Localization(__file__, 563, 12), get_build_msvc_version_38325, *[], **kwargs_38326)
    
    # Assigning a type to the variable 'msver' (line 563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 4), 'msver', get_build_msvc_version_call_result_38327)
    
    # Type idiom detected: calculating its left and rigth part (line 564)
    # Getting the type of 'msver' (line 564)
    msver_38328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 4), 'msver')
    # Getting the type of 'None' (line 564)
    None_38329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 20), 'None')
    
    (may_be_38330, more_types_in_union_38331) = may_not_be_none(msver_38328, None_38329)

    if may_be_38330:

        if more_types_in_union_38331:
            # Runtime conditional SSA (line 564)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Getting the type of 'msver' (line 565)
        msver_38332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 11), 'msver')
        int_38333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 20), 'int')
        # Applying the binary operator '>=' (line 565)
        result_ge_38334 = python_operator(stypy.reporting.localization.Localization(__file__, 565, 11), '>=', msver_38332, int_38333)
        
        # Testing the type of an if condition (line 565)
        if_condition_38335 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 565, 8), result_ge_38334)
        # Assigning a type to the variable 'if_condition_38335' (line 565)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 8), 'if_condition_38335', if_condition_38335)
        # SSA begins for if statement (line 565)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to check_embedded_msvcr_match_linked(...): (line 566)
        # Processing the call arguments (line 566)
        # Getting the type of 'msver' (line 566)
        msver_38337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 46), 'msver', False)
        # Processing the call keyword arguments (line 566)
        kwargs_38338 = {}
        # Getting the type of 'check_embedded_msvcr_match_linked' (line 566)
        check_embedded_msvcr_match_linked_38336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 12), 'check_embedded_msvcr_match_linked', False)
        # Calling check_embedded_msvcr_match_linked(args, kwargs) (line 566)
        check_embedded_msvcr_match_linked_call_result_38339 = invoke(stypy.reporting.localization.Localization(__file__, 566, 12), check_embedded_msvcr_match_linked_38336, *[msver_38337], **kwargs_38338)
        
        
        # Assigning a Call to a Name (line 567):
        
        # Assigning a Call to a Name (line 567):
        
        # Call to int(...): (line 567)
        # Processing the call arguments (line 567)
        # Getting the type of 'msver' (line 567)
        msver_38341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 21), 'msver', False)
        # Processing the call keyword arguments (line 567)
        kwargs_38342 = {}
        # Getting the type of 'int' (line 567)
        int_38340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 17), 'int', False)
        # Calling int(args, kwargs) (line 567)
        int_call_result_38343 = invoke(stypy.reporting.localization.Localization(__file__, 567, 17), int_38340, *[msver_38341], **kwargs_38342)
        
        # Assigning a type to the variable 'ma' (line 567)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 12), 'ma', int_call_result_38343)
        
        # Assigning a Call to a Name (line 568):
        
        # Assigning a Call to a Name (line 568):
        
        # Call to int(...): (line 568)
        # Processing the call arguments (line 568)
        # Getting the type of 'msver' (line 568)
        msver_38345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 22), 'msver', False)
        # Getting the type of 'ma' (line 568)
        ma_38346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 30), 'ma', False)
        # Applying the binary operator '-' (line 568)
        result_sub_38347 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 22), '-', msver_38345, ma_38346)
        
        int_38348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 36), 'int')
        # Applying the binary operator '*' (line 568)
        result_mul_38349 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 21), '*', result_sub_38347, int_38348)
        
        # Processing the call keyword arguments (line 568)
        kwargs_38350 = {}
        # Getting the type of 'int' (line 568)
        int_38344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 17), 'int', False)
        # Calling int(args, kwargs) (line 568)
        int_call_result_38351 = invoke(stypy.reporting.localization.Localization(__file__, 568, 17), int_38344, *[result_mul_38349], **kwargs_38350)
        
        # Assigning a type to the variable 'mi' (line 568)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 12), 'mi', int_call_result_38351)
        
        # Assigning a Call to a Name (line 570):
        
        # Assigning a Call to a Name (line 570):
        
        # Call to msvc_manifest_xml(...): (line 570)
        # Processing the call arguments (line 570)
        # Getting the type of 'ma' (line 570)
        ma_38353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 39), 'ma', False)
        # Getting the type of 'mi' (line 570)
        mi_38354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 43), 'mi', False)
        # Processing the call keyword arguments (line 570)
        kwargs_38355 = {}
        # Getting the type of 'msvc_manifest_xml' (line 570)
        msvc_manifest_xml_38352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 21), 'msvc_manifest_xml', False)
        # Calling msvc_manifest_xml(args, kwargs) (line 570)
        msvc_manifest_xml_call_result_38356 = invoke(stypy.reporting.localization.Localization(__file__, 570, 21), msvc_manifest_xml_38352, *[ma_38353, mi_38354], **kwargs_38355)
        
        # Assigning a type to the variable 'manxml' (line 570)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 12), 'manxml', msvc_manifest_xml_call_result_38356)
        
        # Assigning a Call to a Name (line 571):
        
        # Assigning a Call to a Name (line 571):
        
        # Call to open(...): (line 571)
        # Processing the call arguments (line 571)
        
        # Call to manifest_name(...): (line 571)
        # Processing the call arguments (line 571)
        # Getting the type of 'config' (line 571)
        config_38359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 37), 'config', False)
        # Processing the call keyword arguments (line 571)
        kwargs_38360 = {}
        # Getting the type of 'manifest_name' (line 571)
        manifest_name_38358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 23), 'manifest_name', False)
        # Calling manifest_name(args, kwargs) (line 571)
        manifest_name_call_result_38361 = invoke(stypy.reporting.localization.Localization(__file__, 571, 23), manifest_name_38358, *[config_38359], **kwargs_38360)
        
        str_38362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 46), 'str', 'w')
        # Processing the call keyword arguments (line 571)
        kwargs_38363 = {}
        # Getting the type of 'open' (line 571)
        open_38357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 18), 'open', False)
        # Calling open(args, kwargs) (line 571)
        open_call_result_38364 = invoke(stypy.reporting.localization.Localization(__file__, 571, 18), open_38357, *[manifest_name_call_result_38361, str_38362], **kwargs_38363)
        
        # Assigning a type to the variable 'man' (line 571)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 12), 'man', open_call_result_38364)
        
        # Call to append(...): (line 572)
        # Processing the call arguments (line 572)
        
        # Call to manifest_name(...): (line 572)
        # Processing the call arguments (line 572)
        # Getting the type of 'config' (line 572)
        config_38369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 51), 'config', False)
        # Processing the call keyword arguments (line 572)
        kwargs_38370 = {}
        # Getting the type of 'manifest_name' (line 572)
        manifest_name_38368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 37), 'manifest_name', False)
        # Calling manifest_name(args, kwargs) (line 572)
        manifest_name_call_result_38371 = invoke(stypy.reporting.localization.Localization(__file__, 572, 37), manifest_name_38368, *[config_38369], **kwargs_38370)
        
        # Processing the call keyword arguments (line 572)
        kwargs_38372 = {}
        # Getting the type of 'config' (line 572)
        config_38365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 12), 'config', False)
        # Obtaining the member 'temp_files' of a type (line 572)
        temp_files_38366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 12), config_38365, 'temp_files')
        # Obtaining the member 'append' of a type (line 572)
        append_38367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 12), temp_files_38366, 'append')
        # Calling append(args, kwargs) (line 572)
        append_call_result_38373 = invoke(stypy.reporting.localization.Localization(__file__, 572, 12), append_38367, *[manifest_name_call_result_38371], **kwargs_38372)
        
        
        # Call to write(...): (line 573)
        # Processing the call arguments (line 573)
        # Getting the type of 'manxml' (line 573)
        manxml_38376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 22), 'manxml', False)
        # Processing the call keyword arguments (line 573)
        kwargs_38377 = {}
        # Getting the type of 'man' (line 573)
        man_38374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 12), 'man', False)
        # Obtaining the member 'write' of a type (line 573)
        write_38375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 12), man_38374, 'write')
        # Calling write(args, kwargs) (line 573)
        write_call_result_38378 = invoke(stypy.reporting.localization.Localization(__file__, 573, 12), write_38375, *[manxml_38376], **kwargs_38377)
        
        
        # Call to close(...): (line 574)
        # Processing the call keyword arguments (line 574)
        kwargs_38381 = {}
        # Getting the type of 'man' (line 574)
        man_38379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 12), 'man', False)
        # Obtaining the member 'close' of a type (line 574)
        close_38380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 12), man_38379, 'close')
        # Calling close(args, kwargs) (line 574)
        close_call_result_38382 = invoke(stypy.reporting.localization.Localization(__file__, 574, 12), close_38380, *[], **kwargs_38381)
        
        # SSA join for if statement (line 565)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_38331:
            # SSA join for if statement (line 564)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'generate_manifest(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'generate_manifest' in the type store
    # Getting the type of 'stypy_return_type' (line 562)
    stypy_return_type_38383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38383)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'generate_manifest'
    return stypy_return_type_38383

# Assigning a type to the variable 'generate_manifest' (line 562)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 0), 'generate_manifest', generate_manifest)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

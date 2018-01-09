
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Modified version of build_ext that handles fortran source files.
2: 
3: '''
4: from __future__ import division, absolute_import, print_function
5: 
6: import os
7: import sys
8: from glob import glob
9: 
10: from distutils.dep_util import newer_group
11: from distutils.command.build_ext import build_ext as old_build_ext
12: from distutils.errors import DistutilsFileError, DistutilsSetupError,\
13:      DistutilsError
14: from distutils.file_util import copy_file
15: 
16: from numpy.distutils import log
17: from numpy.distutils.exec_command import exec_command
18: from numpy.distutils.system_info import combine_paths
19: from numpy.distutils.misc_util import filter_sources, has_f_sources, \
20:      has_cxx_sources, get_ext_source_files, \
21:      get_numpy_include_dirs, is_sequence, get_build_architecture, \
22:      msvc_version
23: from numpy.distutils.command.config_compiler import show_fortran_compilers
24: 
25: try:
26:     set
27: except NameError:
28:     from sets import Set as set
29: 
30: class build_ext (old_build_ext):
31: 
32:     description = "build C/C++/F extensions (compile/link to build directory)"
33: 
34:     user_options = old_build_ext.user_options + [
35:         ('fcompiler=', None,
36:          "specify the Fortran compiler type"),
37:         ('parallel=', 'j',
38:          "number of parallel jobs"),
39:         ]
40: 
41:     help_options = old_build_ext.help_options + [
42:         ('help-fcompiler', None, "list available Fortran compilers",
43:          show_fortran_compilers),
44:         ]
45: 
46:     def initialize_options(self):
47:         old_build_ext.initialize_options(self)
48:         self.fcompiler = None
49:         self.parallel = None
50: 
51:     def finalize_options(self):
52:         if self.parallel:
53:             try:
54:                 self.parallel = int(self.parallel)
55:             except ValueError:
56:                 raise ValueError("--parallel/-j argument must be an integer")
57: 
58:         # Ensure that self.include_dirs and self.distribution.include_dirs
59:         # refer to the same list object. finalize_options will modify
60:         # self.include_dirs, but self.distribution.include_dirs is used
61:         # during the actual build.
62:         # self.include_dirs is None unless paths are specified with
63:         # --include-dirs.
64:         # The include paths will be passed to the compiler in the order:
65:         # numpy paths, --include-dirs paths, Python include path.
66:         if isinstance(self.include_dirs, str):
67:             self.include_dirs = self.include_dirs.split(os.pathsep)
68:         incl_dirs = self.include_dirs or []
69:         if self.distribution.include_dirs is None:
70:             self.distribution.include_dirs = []
71:         self.include_dirs = self.distribution.include_dirs
72:         self.include_dirs.extend(incl_dirs)
73: 
74:         old_build_ext.finalize_options(self)
75:         self.set_undefined_options('build', ('parallel', 'parallel'))
76: 
77:     def run(self):
78:         if not self.extensions:
79:             return
80: 
81:         # Make sure that extension sources are complete.
82:         self.run_command('build_src')
83: 
84:         if self.distribution.has_c_libraries():
85:             if self.inplace:
86:                 if self.distribution.have_run.get('build_clib'):
87:                     log.warn('build_clib already run, it is too late to ' \
88:                             'ensure in-place build of build_clib')
89:                     build_clib = self.distribution.get_command_obj('build_clib')
90:                 else:
91:                     build_clib = self.distribution.get_command_obj('build_clib')
92:                     build_clib.inplace = 1
93:                     build_clib.ensure_finalized()
94:                     build_clib.run()
95:                     self.distribution.have_run['build_clib'] = 1
96: 
97:             else:
98:                 self.run_command('build_clib')
99:                 build_clib = self.get_finalized_command('build_clib')
100:             self.library_dirs.append(build_clib.build_clib)
101:         else:
102:             build_clib = None
103: 
104:         # Not including C libraries to the list of
105:         # extension libraries automatically to prevent
106:         # bogus linking commands. Extensions must
107:         # explicitly specify the C libraries that they use.
108: 
109:         from distutils.ccompiler import new_compiler
110:         from numpy.distutils.fcompiler import new_fcompiler
111: 
112:         compiler_type = self.compiler
113:         # Initialize C compiler:
114:         self.compiler = new_compiler(compiler=compiler_type,
115:                                      verbose=self.verbose,
116:                                      dry_run=self.dry_run,
117:                                      force=self.force)
118:         self.compiler.customize(self.distribution)
119:         self.compiler.customize_cmd(self)
120:         self.compiler.show_customization()
121: 
122:         # Create mapping of libraries built by build_clib:
123:         clibs = {}
124:         if build_clib is not None:
125:             for libname, build_info in build_clib.libraries or []:
126:                 if libname in clibs and clibs[libname] != build_info:
127:                     log.warn('library %r defined more than once,'\
128:                              ' overwriting build_info\n%s... \nwith\n%s...' \
129:                              % (libname, repr(clibs[libname])[:300], repr(build_info)[:300]))
130:                 clibs[libname] = build_info
131:         # .. and distribution libraries:
132:         for libname, build_info in self.distribution.libraries or []:
133:             if libname in clibs:
134:                 # build_clib libraries have a precedence before distribution ones
135:                 continue
136:             clibs[libname] = build_info
137: 
138:         # Determine if C++/Fortran 77/Fortran 90 compilers are needed.
139:         # Update extension libraries, library_dirs, and macros.
140:         all_languages = set()
141:         for ext in self.extensions:
142:             ext_languages = set()
143:             c_libs = []
144:             c_lib_dirs = []
145:             macros = []
146:             for libname in ext.libraries:
147:                 if libname in clibs:
148:                     binfo = clibs[libname]
149:                     c_libs += binfo.get('libraries', [])
150:                     c_lib_dirs += binfo.get('library_dirs', [])
151:                     for m in binfo.get('macros', []):
152:                         if m not in macros:
153:                             macros.append(m)
154: 
155:                 for l in clibs.get(libname, {}).get('source_languages', []):
156:                     ext_languages.add(l)
157:             if c_libs:
158:                 new_c_libs = ext.libraries + c_libs
159:                 log.info('updating extension %r libraries from %r to %r'
160:                          % (ext.name, ext.libraries, new_c_libs))
161:                 ext.libraries = new_c_libs
162:                 ext.library_dirs = ext.library_dirs + c_lib_dirs
163:             if macros:
164:                 log.info('extending extension %r defined_macros with %r'
165:                          % (ext.name, macros))
166:                 ext.define_macros = ext.define_macros + macros
167: 
168:             # determine extension languages
169:             if has_f_sources(ext.sources):
170:                 ext_languages.add('f77')
171:             if has_cxx_sources(ext.sources):
172:                 ext_languages.add('c++')
173:             l = ext.language or self.compiler.detect_language(ext.sources)
174:             if l:
175:                 ext_languages.add(l)
176:             # reset language attribute for choosing proper linker
177:             if 'c++' in ext_languages:
178:                 ext_language = 'c++'
179:             elif 'f90' in ext_languages:
180:                 ext_language = 'f90'
181:             elif 'f77' in ext_languages:
182:                 ext_language = 'f77'
183:             else:
184:                 ext_language = 'c' # default
185:             if l and l != ext_language and ext.language:
186:                 log.warn('resetting extension %r language from %r to %r.' %
187:                          (ext.name, l, ext_language))
188:             ext.language = ext_language
189:             # global language
190:             all_languages.update(ext_languages)
191: 
192:         need_f90_compiler = 'f90' in all_languages
193:         need_f77_compiler = 'f77' in all_languages
194:         need_cxx_compiler = 'c++' in all_languages
195: 
196:         # Initialize C++ compiler:
197:         if need_cxx_compiler:
198:             self._cxx_compiler = new_compiler(compiler=compiler_type,
199:                                              verbose=self.verbose,
200:                                              dry_run=self.dry_run,
201:                                              force=self.force)
202:             compiler = self._cxx_compiler
203:             compiler.customize(self.distribution, need_cxx=need_cxx_compiler)
204:             compiler.customize_cmd(self)
205:             compiler.show_customization()
206:             self._cxx_compiler = compiler.cxx_compiler()
207:         else:
208:             self._cxx_compiler = None
209: 
210:         # Initialize Fortran 77 compiler:
211:         if need_f77_compiler:
212:             ctype = self.fcompiler
213:             self._f77_compiler = new_fcompiler(compiler=self.fcompiler,
214:                                                verbose=self.verbose,
215:                                                dry_run=self.dry_run,
216:                                                force=self.force,
217:                                                requiref90=False,
218:                                                c_compiler=self.compiler)
219:             fcompiler = self._f77_compiler
220:             if fcompiler:
221:                 ctype = fcompiler.compiler_type
222:                 fcompiler.customize(self.distribution)
223:             if fcompiler and fcompiler.get_version():
224:                 fcompiler.customize_cmd(self)
225:                 fcompiler.show_customization()
226:             else:
227:                 self.warn('f77_compiler=%s is not available.' %
228:                           (ctype))
229:                 self._f77_compiler = None
230:         else:
231:             self._f77_compiler = None
232: 
233:         # Initialize Fortran 90 compiler:
234:         if need_f90_compiler:
235:             ctype = self.fcompiler
236:             self._f90_compiler = new_fcompiler(compiler=self.fcompiler,
237:                                                verbose=self.verbose,
238:                                                dry_run=self.dry_run,
239:                                                force=self.force,
240:                                                requiref90=True,
241:                                                c_compiler = self.compiler)
242:             fcompiler = self._f90_compiler
243:             if fcompiler:
244:                 ctype = fcompiler.compiler_type
245:                 fcompiler.customize(self.distribution)
246:             if fcompiler and fcompiler.get_version():
247:                 fcompiler.customize_cmd(self)
248:                 fcompiler.show_customization()
249:             else:
250:                 self.warn('f90_compiler=%s is not available.' %
251:                           (ctype))
252:                 self._f90_compiler = None
253:         else:
254:             self._f90_compiler = None
255: 
256:         # Build extensions
257:         self.build_extensions()
258: 
259: 
260:     def swig_sources(self, sources):
261:         # Do nothing. Swig sources have beed handled in build_src command.
262:         return sources
263: 
264:     def build_extension(self, ext):
265:         sources = ext.sources
266:         if sources is None or not is_sequence(sources):
267:             raise DistutilsSetupError(
268:                 ("in 'ext_modules' option (extension '%s'), " +
269:                  "'sources' must be present and must be " +
270:                  "a list of source filenames") % ext.name)
271:         sources = list(sources)
272: 
273:         if not sources:
274:             return
275: 
276:         fullname = self.get_ext_fullname(ext.name)
277:         if self.inplace:
278:             modpath = fullname.split('.')
279:             package = '.'.join(modpath[0:-1])
280:             base = modpath[-1]
281:             build_py = self.get_finalized_command('build_py')
282:             package_dir = build_py.get_package_dir(package)
283:             ext_filename = os.path.join(package_dir,
284:                                         self.get_ext_filename(base))
285:         else:
286:             ext_filename = os.path.join(self.build_lib,
287:                                         self.get_ext_filename(fullname))
288:         depends = sources + ext.depends
289: 
290:         if not (self.force or newer_group(depends, ext_filename, 'newer')):
291:             log.debug("skipping '%s' extension (up-to-date)", ext.name)
292:             return
293:         else:
294:             log.info("building '%s' extension", ext.name)
295: 
296:         extra_args = ext.extra_compile_args or []
297:         macros = ext.define_macros[:]
298:         for undef in ext.undef_macros:
299:             macros.append((undef,))
300: 
301:         c_sources, cxx_sources, f_sources, fmodule_sources = \
302:                    filter_sources(ext.sources)
303: 
304: 
305: 
306:         if self.compiler.compiler_type=='msvc':
307:             if cxx_sources:
308:                 # Needed to compile kiva.agg._agg extension.
309:                 extra_args.append('/Zm1000')
310:             # this hack works around the msvc compiler attributes
311:             # problem, msvc uses its own convention :(
312:             c_sources += cxx_sources
313:             cxx_sources = []
314: 
315:         # Set Fortran/C++ compilers for compilation and linking.
316:         if ext.language=='f90':
317:             fcompiler = self._f90_compiler
318:         elif ext.language=='f77':
319:             fcompiler = self._f77_compiler
320:         else: # in case ext.language is c++, for instance
321:             fcompiler = self._f90_compiler or self._f77_compiler
322:         if fcompiler is not None:
323:             fcompiler.extra_f77_compile_args = (ext.extra_f77_compile_args or []) if hasattr(ext, 'extra_f77_compile_args') else []
324:             fcompiler.extra_f90_compile_args = (ext.extra_f90_compile_args or []) if hasattr(ext, 'extra_f90_compile_args') else []
325:         cxx_compiler = self._cxx_compiler
326: 
327:         # check for the availability of required compilers
328:         if cxx_sources and cxx_compiler is None:
329:             raise DistutilsError("extension %r has C++ sources" \
330:                   "but no C++ compiler found" % (ext.name))
331:         if (f_sources or fmodule_sources) and fcompiler is None:
332:             raise DistutilsError("extension %r has Fortran sources " \
333:                   "but no Fortran compiler found" % (ext.name))
334:         if ext.language in ['f77', 'f90'] and fcompiler is None:
335:             self.warn("extension %r has Fortran libraries " \
336:                   "but no Fortran linker found, using default linker" % (ext.name))
337:         if ext.language=='c++' and cxx_compiler is None:
338:             self.warn("extension %r has C++ libraries " \
339:                   "but no C++ linker found, using default linker" % (ext.name))
340: 
341:         kws = {'depends':ext.depends}
342:         output_dir = self.build_temp
343: 
344:         include_dirs = ext.include_dirs + get_numpy_include_dirs()
345: 
346:         c_objects = []
347:         if c_sources:
348:             log.info("compiling C sources")
349:             c_objects = self.compiler.compile(c_sources,
350:                                               output_dir=output_dir,
351:                                               macros=macros,
352:                                               include_dirs=include_dirs,
353:                                               debug=self.debug,
354:                                               extra_postargs=extra_args,
355:                                               **kws)
356: 
357:         if cxx_sources:
358:             log.info("compiling C++ sources")
359:             c_objects += cxx_compiler.compile(cxx_sources,
360:                                               output_dir=output_dir,
361:                                               macros=macros,
362:                                               include_dirs=include_dirs,
363:                                               debug=self.debug,
364:                                               extra_postargs=extra_args,
365:                                               **kws)
366: 
367:         extra_postargs = []
368:         f_objects = []
369:         if fmodule_sources:
370:             log.info("compiling Fortran 90 module sources")
371:             module_dirs = ext.module_dirs[:]
372:             module_build_dir = os.path.join(
373:                 self.build_temp, os.path.dirname(
374:                     self.get_ext_filename(fullname)))
375: 
376:             self.mkpath(module_build_dir)
377:             if fcompiler.module_dir_switch is None:
378:                 existing_modules = glob('*.mod')
379:             extra_postargs += fcompiler.module_options(
380:                 module_dirs, module_build_dir)
381:             f_objects += fcompiler.compile(fmodule_sources,
382:                                            output_dir=self.build_temp,
383:                                            macros=macros,
384:                                            include_dirs=include_dirs,
385:                                            debug=self.debug,
386:                                            extra_postargs=extra_postargs,
387:                                            depends=ext.depends)
388: 
389:             if fcompiler.module_dir_switch is None:
390:                 for f in glob('*.mod'):
391:                     if f in existing_modules:
392:                         continue
393:                     t = os.path.join(module_build_dir, f)
394:                     if os.path.abspath(f)==os.path.abspath(t):
395:                         continue
396:                     if os.path.isfile(t):
397:                         os.remove(t)
398:                     try:
399:                         self.move_file(f, module_build_dir)
400:                     except DistutilsFileError:
401:                         log.warn('failed to move %r to %r' %
402:                                  (f, module_build_dir))
403:         if f_sources:
404:             log.info("compiling Fortran sources")
405:             f_objects += fcompiler.compile(f_sources,
406:                                            output_dir=self.build_temp,
407:                                            macros=macros,
408:                                            include_dirs=include_dirs,
409:                                            debug=self.debug,
410:                                            extra_postargs=extra_postargs,
411:                                            depends=ext.depends)
412: 
413:         objects = c_objects + f_objects
414: 
415:         if ext.extra_objects:
416:             objects.extend(ext.extra_objects)
417:         extra_args = ext.extra_link_args or []
418:         libraries = self.get_libraries(ext)[:]
419:         library_dirs = ext.library_dirs[:]
420: 
421:         linker = self.compiler.link_shared_object
422:         # Always use system linker when using MSVC compiler.
423:         if self.compiler.compiler_type in ('msvc', 'intelw', 'intelemw'):
424:             # expand libraries with fcompiler libraries as we are
425:             # not using fcompiler linker
426:             self._libs_with_msvc_and_fortran(fcompiler, libraries, library_dirs)
427: 
428:         elif ext.language in ['f77', 'f90'] and fcompiler is not None:
429:             linker = fcompiler.link_shared_object
430:         if ext.language=='c++' and cxx_compiler is not None:
431:             linker = cxx_compiler.link_shared_object
432: 
433:         linker(objects, ext_filename,
434:                libraries=libraries,
435:                library_dirs=library_dirs,
436:                runtime_library_dirs=ext.runtime_library_dirs,
437:                extra_postargs=extra_args,
438:                export_symbols=self.get_export_symbols(ext),
439:                debug=self.debug,
440:                build_temp=self.build_temp,
441:                target_lang=ext.language)
442: 
443:     def _add_dummy_mingwex_sym(self, c_sources):
444:         build_src = self.get_finalized_command("build_src").build_src
445:         build_clib = self.get_finalized_command("build_clib").build_clib
446:         objects = self.compiler.compile([os.path.join(build_src,
447:                 "gfortran_vs2003_hack.c")],
448:                 output_dir=self.build_temp)
449:         self.compiler.create_static_lib(objects, "_gfortran_workaround", output_dir=build_clib, debug=self.debug)
450: 
451:     def _libs_with_msvc_and_fortran(self, fcompiler, c_libraries,
452:                                     c_library_dirs):
453:         if fcompiler is None: return
454: 
455:         for libname in c_libraries:
456:             if libname.startswith('msvc'): continue
457:             fileexists = False
458:             for libdir in c_library_dirs or []:
459:                 libfile = os.path.join(libdir, '%s.lib' % (libname))
460:                 if os.path.isfile(libfile):
461:                     fileexists = True
462:                     break
463:             if fileexists: continue
464:             # make g77-compiled static libs available to MSVC
465:             fileexists = False
466:             for libdir in c_library_dirs:
467:                 libfile = os.path.join(libdir, 'lib%s.a' % (libname))
468:                 if os.path.isfile(libfile):
469:                     # copy libname.a file to name.lib so that MSVC linker
470:                     # can find it
471:                     libfile2 = os.path.join(self.build_temp, libname + '.lib')
472:                     copy_file(libfile, libfile2)
473:                     if self.build_temp not in c_library_dirs:
474:                         c_library_dirs.append(self.build_temp)
475:                     fileexists = True
476:                     break
477:             if fileexists: continue
478:             log.warn('could not find library %r in directories %s'
479:                      % (libname, c_library_dirs))
480: 
481:         # Always use system linker when using MSVC compiler.
482:         f_lib_dirs = []
483:         for dir in fcompiler.library_dirs:
484:             # correct path when compiling in Cygwin but with normal Win
485:             # Python
486:             if dir.startswith('/usr/lib'):
487:                 s, o = exec_command(['cygpath', '-w', dir], use_tee=False)
488:                 if not s:
489:                     dir = o
490:             f_lib_dirs.append(dir)
491:         c_library_dirs.extend(f_lib_dirs)
492: 
493:         # make g77-compiled static libs available to MSVC
494:         for lib in fcompiler.libraries:
495:             if not lib.startswith('msvc'):
496:                 c_libraries.append(lib)
497:                 p = combine_paths(f_lib_dirs, 'lib' + lib + '.a')
498:                 if p:
499:                     dst_name = os.path.join(self.build_temp, lib + '.lib')
500:                     if not os.path.isfile(dst_name):
501:                         copy_file(p[0], dst_name)
502:                     if self.build_temp not in c_library_dirs:
503:                         c_library_dirs.append(self.build_temp)
504: 
505:     def get_source_files (self):
506:         self.check_extensions_list(self.extensions)
507:         filenames = []
508:         for ext in self.extensions:
509:             filenames.extend(get_ext_source_files(ext))
510:         return filenames
511: 
512:     def get_outputs (self):
513:         self.check_extensions_list(self.extensions)
514: 
515:         outputs = []
516:         for ext in self.extensions:
517:             if not ext.sources:
518:                 continue
519:             fullname = self.get_ext_fullname(ext.name)
520:             outputs.append(os.path.join(self.build_lib,
521:                                         self.get_ext_filename(fullname)))
522:         return outputs
523: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_53472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', ' Modified version of build_ext that handles fortran source files.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import os' statement (line 6)
import os

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import sys' statement (line 7)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from glob import glob' statement (line 8)
from glob import glob

import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'glob', None, module_type_store, ['glob'], [glob])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from distutils.dep_util import newer_group' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_53473 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.dep_util')

if (type(import_53473) is not StypyTypeError):

    if (import_53473 != 'pyd_module'):
        __import__(import_53473)
        sys_modules_53474 = sys.modules[import_53473]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.dep_util', sys_modules_53474.module_type_store, module_type_store, ['newer_group'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_53474, sys_modules_53474.module_type_store, module_type_store)
    else:
        from distutils.dep_util import newer_group

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.dep_util', None, module_type_store, ['newer_group'], [newer_group])

else:
    # Assigning a type to the variable 'distutils.dep_util' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.dep_util', import_53473)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from distutils.command.build_ext import old_build_ext' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_53475 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.command.build_ext')

if (type(import_53475) is not StypyTypeError):

    if (import_53475 != 'pyd_module'):
        __import__(import_53475)
        sys_modules_53476 = sys.modules[import_53475]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.command.build_ext', sys_modules_53476.module_type_store, module_type_store, ['build_ext'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_53476, sys_modules_53476.module_type_store, module_type_store)
    else:
        from distutils.command.build_ext import build_ext as old_build_ext

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.command.build_ext', None, module_type_store, ['build_ext'], [old_build_ext])

else:
    # Assigning a type to the variable 'distutils.command.build_ext' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.command.build_ext', import_53475)

# Adding an alias
module_type_store.add_alias('old_build_ext', 'build_ext')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils.errors import DistutilsFileError, DistutilsSetupError, DistutilsError' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_53477 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors')

if (type(import_53477) is not StypyTypeError):

    if (import_53477 != 'pyd_module'):
        __import__(import_53477)
        sys_modules_53478 = sys.modules[import_53477]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors', sys_modules_53478.module_type_store, module_type_store, ['DistutilsFileError', 'DistutilsSetupError', 'DistutilsError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_53478, sys_modules_53478.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsFileError, DistutilsSetupError, DistutilsError

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors', None, module_type_store, ['DistutilsFileError', 'DistutilsSetupError', 'DistutilsError'], [DistutilsFileError, DistutilsSetupError, DistutilsError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors', import_53477)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from distutils.file_util import copy_file' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_53479 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.file_util')

if (type(import_53479) is not StypyTypeError):

    if (import_53479 != 'pyd_module'):
        __import__(import_53479)
        sys_modules_53480 = sys.modules[import_53479]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.file_util', sys_modules_53480.module_type_store, module_type_store, ['copy_file'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_53480, sys_modules_53480.module_type_store, module_type_store)
    else:
        from distutils.file_util import copy_file

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.file_util', None, module_type_store, ['copy_file'], [copy_file])

else:
    # Assigning a type to the variable 'distutils.file_util' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.file_util', import_53479)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from numpy.distutils import log' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_53481 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy.distutils')

if (type(import_53481) is not StypyTypeError):

    if (import_53481 != 'pyd_module'):
        __import__(import_53481)
        sys_modules_53482 = sys.modules[import_53481]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy.distutils', sys_modules_53482.module_type_store, module_type_store, ['log'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_53482, sys_modules_53482.module_type_store, module_type_store)
    else:
        from numpy.distutils import log

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy.distutils', None, module_type_store, ['log'], [log])

else:
    # Assigning a type to the variable 'numpy.distutils' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy.distutils', import_53481)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from numpy.distutils.exec_command import exec_command' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_53483 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy.distutils.exec_command')

if (type(import_53483) is not StypyTypeError):

    if (import_53483 != 'pyd_module'):
        __import__(import_53483)
        sys_modules_53484 = sys.modules[import_53483]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy.distutils.exec_command', sys_modules_53484.module_type_store, module_type_store, ['exec_command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_53484, sys_modules_53484.module_type_store, module_type_store)
    else:
        from numpy.distutils.exec_command import exec_command

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy.distutils.exec_command', None, module_type_store, ['exec_command'], [exec_command])

else:
    # Assigning a type to the variable 'numpy.distutils.exec_command' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy.distutils.exec_command', import_53483)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from numpy.distutils.system_info import combine_paths' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_53485 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.distutils.system_info')

if (type(import_53485) is not StypyTypeError):

    if (import_53485 != 'pyd_module'):
        __import__(import_53485)
        sys_modules_53486 = sys.modules[import_53485]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.distutils.system_info', sys_modules_53486.module_type_store, module_type_store, ['combine_paths'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_53486, sys_modules_53486.module_type_store, module_type_store)
    else:
        from numpy.distutils.system_info import combine_paths

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.distutils.system_info', None, module_type_store, ['combine_paths'], [combine_paths])

else:
    # Assigning a type to the variable 'numpy.distutils.system_info' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.distutils.system_info', import_53485)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from numpy.distutils.misc_util import filter_sources, has_f_sources, has_cxx_sources, get_ext_source_files, get_numpy_include_dirs, is_sequence, get_build_architecture, msvc_version' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_53487 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy.distutils.misc_util')

if (type(import_53487) is not StypyTypeError):

    if (import_53487 != 'pyd_module'):
        __import__(import_53487)
        sys_modules_53488 = sys.modules[import_53487]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy.distutils.misc_util', sys_modules_53488.module_type_store, module_type_store, ['filter_sources', 'has_f_sources', 'has_cxx_sources', 'get_ext_source_files', 'get_numpy_include_dirs', 'is_sequence', 'get_build_architecture', 'msvc_version'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_53488, sys_modules_53488.module_type_store, module_type_store)
    else:
        from numpy.distutils.misc_util import filter_sources, has_f_sources, has_cxx_sources, get_ext_source_files, get_numpy_include_dirs, is_sequence, get_build_architecture, msvc_version

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy.distutils.misc_util', None, module_type_store, ['filter_sources', 'has_f_sources', 'has_cxx_sources', 'get_ext_source_files', 'get_numpy_include_dirs', 'is_sequence', 'get_build_architecture', 'msvc_version'], [filter_sources, has_f_sources, has_cxx_sources, get_ext_source_files, get_numpy_include_dirs, is_sequence, get_build_architecture, msvc_version])

else:
    # Assigning a type to the variable 'numpy.distutils.misc_util' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy.distutils.misc_util', import_53487)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from numpy.distutils.command.config_compiler import show_fortran_compilers' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_53489 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.distutils.command.config_compiler')

if (type(import_53489) is not StypyTypeError):

    if (import_53489 != 'pyd_module'):
        __import__(import_53489)
        sys_modules_53490 = sys.modules[import_53489]
        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.distutils.command.config_compiler', sys_modules_53490.module_type_store, module_type_store, ['show_fortran_compilers'])
        nest_module(stypy.reporting.localization.Localization(__file__, 23, 0), __file__, sys_modules_53490, sys_modules_53490.module_type_store, module_type_store)
    else:
        from numpy.distutils.command.config_compiler import show_fortran_compilers

        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.distutils.command.config_compiler', None, module_type_store, ['show_fortran_compilers'], [show_fortran_compilers])

else:
    # Assigning a type to the variable 'numpy.distutils.command.config_compiler' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.distutils.command.config_compiler', import_53489)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')



# SSA begins for try-except statement (line 25)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
# Getting the type of 'set' (line 26)
set_53491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'set')
# SSA branch for the except part of a try statement (line 25)
# SSA branch for the except 'NameError' branch of a try statement (line 25)
module_type_store.open_ssa_branch('except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 4))

# 'from sets import set' statement (line 28)
from sets import Set as set

import_from_module(stypy.reporting.localization.Localization(__file__, 28, 4), 'sets', None, module_type_store, ['Set'], [set])
# Adding an alias
module_type_store.add_alias('set', 'Set')

# SSA join for try-except statement (line 25)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'build_ext' class
# Getting the type of 'old_build_ext' (line 30)
old_build_ext_53492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 17), 'old_build_ext')

class build_ext(old_build_ext_53492, ):
    
    # Assigning a Str to a Name (line 32):
    
    # Assigning a BinOp to a Name (line 34):
    
    # Assigning a BinOp to a Name (line 41):

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 46, 4, False)
        # Assigning a type to the variable 'self' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        build_ext.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext.initialize_options.__dict__.__setitem__('stypy_function_name', 'build_ext.initialize_options')
        build_ext.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        build_ext.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.initialize_options', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'initialize_options', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'initialize_options(...)' code ##################

        
        # Call to initialize_options(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'self' (line 47)
        self_53495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 41), 'self', False)
        # Processing the call keyword arguments (line 47)
        kwargs_53496 = {}
        # Getting the type of 'old_build_ext' (line 47)
        old_build_ext_53493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'old_build_ext', False)
        # Obtaining the member 'initialize_options' of a type (line 47)
        initialize_options_53494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), old_build_ext_53493, 'initialize_options')
        # Calling initialize_options(args, kwargs) (line 47)
        initialize_options_call_result_53497 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), initialize_options_53494, *[self_53495], **kwargs_53496)
        
        
        # Assigning a Name to a Attribute (line 48):
        
        # Assigning a Name to a Attribute (line 48):
        # Getting the type of 'None' (line 48)
        None_53498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 25), 'None')
        # Getting the type of 'self' (line 48)
        self_53499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'self')
        # Setting the type of the member 'fcompiler' of a type (line 48)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), self_53499, 'fcompiler', None_53498)
        
        # Assigning a Name to a Attribute (line 49):
        
        # Assigning a Name to a Attribute (line 49):
        # Getting the type of 'None' (line 49)
        None_53500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 24), 'None')
        # Getting the type of 'self' (line 49)
        self_53501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'self')
        # Setting the type of the member 'parallel' of a type (line 49)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), self_53501, 'parallel', None_53500)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 46)
        stypy_return_type_53502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_53502)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_53502


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 51, 4, False)
        # Assigning a type to the variable 'self' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        build_ext.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext.finalize_options.__dict__.__setitem__('stypy_function_name', 'build_ext.finalize_options')
        build_ext.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        build_ext.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.finalize_options', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'finalize_options', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'finalize_options(...)' code ##################

        
        # Getting the type of 'self' (line 52)
        self_53503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'self')
        # Obtaining the member 'parallel' of a type (line 52)
        parallel_53504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 11), self_53503, 'parallel')
        # Testing the type of an if condition (line 52)
        if_condition_53505 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 52, 8), parallel_53504)
        # Assigning a type to the variable 'if_condition_53505' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'if_condition_53505', if_condition_53505)
        # SSA begins for if statement (line 52)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # SSA begins for try-except statement (line 53)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Attribute (line 54):
        
        # Assigning a Call to a Attribute (line 54):
        
        # Call to int(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'self' (line 54)
        self_53507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 36), 'self', False)
        # Obtaining the member 'parallel' of a type (line 54)
        parallel_53508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 36), self_53507, 'parallel')
        # Processing the call keyword arguments (line 54)
        kwargs_53509 = {}
        # Getting the type of 'int' (line 54)
        int_53506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 32), 'int', False)
        # Calling int(args, kwargs) (line 54)
        int_call_result_53510 = invoke(stypy.reporting.localization.Localization(__file__, 54, 32), int_53506, *[parallel_53508], **kwargs_53509)
        
        # Getting the type of 'self' (line 54)
        self_53511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'self')
        # Setting the type of the member 'parallel' of a type (line 54)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 16), self_53511, 'parallel', int_call_result_53510)
        # SSA branch for the except part of a try statement (line 53)
        # SSA branch for the except 'ValueError' branch of a try statement (line 53)
        module_type_store.open_ssa_branch('except')
        
        # Call to ValueError(...): (line 56)
        # Processing the call arguments (line 56)
        str_53513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 33), 'str', '--parallel/-j argument must be an integer')
        # Processing the call keyword arguments (line 56)
        kwargs_53514 = {}
        # Getting the type of 'ValueError' (line 56)
        ValueError_53512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 56)
        ValueError_call_result_53515 = invoke(stypy.reporting.localization.Localization(__file__, 56, 22), ValueError_53512, *[str_53513], **kwargs_53514)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 56, 16), ValueError_call_result_53515, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 53)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 52)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 66)
        # Getting the type of 'str' (line 66)
        str_53516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 41), 'str')
        # Getting the type of 'self' (line 66)
        self_53517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 22), 'self')
        # Obtaining the member 'include_dirs' of a type (line 66)
        include_dirs_53518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 22), self_53517, 'include_dirs')
        
        (may_be_53519, more_types_in_union_53520) = may_be_subtype(str_53516, include_dirs_53518)

        if may_be_53519:

            if more_types_in_union_53520:
                # Runtime conditional SSA (line 66)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 66)
            self_53521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'self')
            # Obtaining the member 'include_dirs' of a type (line 66)
            include_dirs_53522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), self_53521, 'include_dirs')
            # Setting the type of the member 'include_dirs' of a type (line 66)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), self_53521, 'include_dirs', remove_not_subtype_from_union(include_dirs_53518, str))
            
            # Assigning a Call to a Attribute (line 67):
            
            # Assigning a Call to a Attribute (line 67):
            
            # Call to split(...): (line 67)
            # Processing the call arguments (line 67)
            # Getting the type of 'os' (line 67)
            os_53526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 56), 'os', False)
            # Obtaining the member 'pathsep' of a type (line 67)
            pathsep_53527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 56), os_53526, 'pathsep')
            # Processing the call keyword arguments (line 67)
            kwargs_53528 = {}
            # Getting the type of 'self' (line 67)
            self_53523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 32), 'self', False)
            # Obtaining the member 'include_dirs' of a type (line 67)
            include_dirs_53524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 32), self_53523, 'include_dirs')
            # Obtaining the member 'split' of a type (line 67)
            split_53525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 32), include_dirs_53524, 'split')
            # Calling split(args, kwargs) (line 67)
            split_call_result_53529 = invoke(stypy.reporting.localization.Localization(__file__, 67, 32), split_53525, *[pathsep_53527], **kwargs_53528)
            
            # Getting the type of 'self' (line 67)
            self_53530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'self')
            # Setting the type of the member 'include_dirs' of a type (line 67)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 12), self_53530, 'include_dirs', split_call_result_53529)

            if more_types_in_union_53520:
                # SSA join for if statement (line 66)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BoolOp to a Name (line 68):
        
        # Assigning a BoolOp to a Name (line 68):
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 68)
        self_53531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 20), 'self')
        # Obtaining the member 'include_dirs' of a type (line 68)
        include_dirs_53532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 20), self_53531, 'include_dirs')
        
        # Obtaining an instance of the builtin type 'list' (line 68)
        list_53533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 68)
        
        # Applying the binary operator 'or' (line 68)
        result_or_keyword_53534 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 20), 'or', include_dirs_53532, list_53533)
        
        # Assigning a type to the variable 'incl_dirs' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'incl_dirs', result_or_keyword_53534)
        
        # Type idiom detected: calculating its left and rigth part (line 69)
        # Getting the type of 'self' (line 69)
        self_53535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 11), 'self')
        # Obtaining the member 'distribution' of a type (line 69)
        distribution_53536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 11), self_53535, 'distribution')
        # Obtaining the member 'include_dirs' of a type (line 69)
        include_dirs_53537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 11), distribution_53536, 'include_dirs')
        # Getting the type of 'None' (line 69)
        None_53538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 45), 'None')
        
        (may_be_53539, more_types_in_union_53540) = may_be_none(include_dirs_53537, None_53538)

        if may_be_53539:

            if more_types_in_union_53540:
                # Runtime conditional SSA (line 69)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a List to a Attribute (line 70):
            
            # Assigning a List to a Attribute (line 70):
            
            # Obtaining an instance of the builtin type 'list' (line 70)
            list_53541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 45), 'list')
            # Adding type elements to the builtin type 'list' instance (line 70)
            
            # Getting the type of 'self' (line 70)
            self_53542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'self')
            # Obtaining the member 'distribution' of a type (line 70)
            distribution_53543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), self_53542, 'distribution')
            # Setting the type of the member 'include_dirs' of a type (line 70)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), distribution_53543, 'include_dirs', list_53541)

            if more_types_in_union_53540:
                # SSA join for if statement (line 69)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Attribute (line 71):
        
        # Assigning a Attribute to a Attribute (line 71):
        # Getting the type of 'self' (line 71)
        self_53544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 28), 'self')
        # Obtaining the member 'distribution' of a type (line 71)
        distribution_53545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 28), self_53544, 'distribution')
        # Obtaining the member 'include_dirs' of a type (line 71)
        include_dirs_53546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 28), distribution_53545, 'include_dirs')
        # Getting the type of 'self' (line 71)
        self_53547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'self')
        # Setting the type of the member 'include_dirs' of a type (line 71)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), self_53547, 'include_dirs', include_dirs_53546)
        
        # Call to extend(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'incl_dirs' (line 72)
        incl_dirs_53551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 33), 'incl_dirs', False)
        # Processing the call keyword arguments (line 72)
        kwargs_53552 = {}
        # Getting the type of 'self' (line 72)
        self_53548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'self', False)
        # Obtaining the member 'include_dirs' of a type (line 72)
        include_dirs_53549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), self_53548, 'include_dirs')
        # Obtaining the member 'extend' of a type (line 72)
        extend_53550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), include_dirs_53549, 'extend')
        # Calling extend(args, kwargs) (line 72)
        extend_call_result_53553 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), extend_53550, *[incl_dirs_53551], **kwargs_53552)
        
        
        # Call to finalize_options(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'self' (line 74)
        self_53556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 39), 'self', False)
        # Processing the call keyword arguments (line 74)
        kwargs_53557 = {}
        # Getting the type of 'old_build_ext' (line 74)
        old_build_ext_53554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'old_build_ext', False)
        # Obtaining the member 'finalize_options' of a type (line 74)
        finalize_options_53555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), old_build_ext_53554, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 74)
        finalize_options_call_result_53558 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), finalize_options_53555, *[self_53556], **kwargs_53557)
        
        
        # Call to set_undefined_options(...): (line 75)
        # Processing the call arguments (line 75)
        str_53561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 35), 'str', 'build')
        
        # Obtaining an instance of the builtin type 'tuple' (line 75)
        tuple_53562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 75)
        # Adding element type (line 75)
        str_53563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 45), 'str', 'parallel')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 45), tuple_53562, str_53563)
        # Adding element type (line 75)
        str_53564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 57), 'str', 'parallel')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 45), tuple_53562, str_53564)
        
        # Processing the call keyword arguments (line 75)
        kwargs_53565 = {}
        # Getting the type of 'self' (line 75)
        self_53559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'self', False)
        # Obtaining the member 'set_undefined_options' of a type (line 75)
        set_undefined_options_53560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), self_53559, 'set_undefined_options')
        # Calling set_undefined_options(args, kwargs) (line 75)
        set_undefined_options_call_result_53566 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), set_undefined_options_53560, *[str_53561, tuple_53562], **kwargs_53565)
        
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 51)
        stypy_return_type_53567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_53567)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_53567


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 77, 4, False)
        # Assigning a type to the variable 'self' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext.run.__dict__.__setitem__('stypy_localization', localization)
        build_ext.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext.run.__dict__.__setitem__('stypy_function_name', 'build_ext.run')
        build_ext.run.__dict__.__setitem__('stypy_param_names_list', [])
        build_ext.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.run', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'run', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'run(...)' code ##################

        
        
        # Getting the type of 'self' (line 78)
        self_53568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 15), 'self')
        # Obtaining the member 'extensions' of a type (line 78)
        extensions_53569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 15), self_53568, 'extensions')
        # Applying the 'not' unary operator (line 78)
        result_not__53570 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 11), 'not', extensions_53569)
        
        # Testing the type of an if condition (line 78)
        if_condition_53571 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 8), result_not__53570)
        # Assigning a type to the variable 'if_condition_53571' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'if_condition_53571', if_condition_53571)
        # SSA begins for if statement (line 78)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 78)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to run_command(...): (line 82)
        # Processing the call arguments (line 82)
        str_53574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 25), 'str', 'build_src')
        # Processing the call keyword arguments (line 82)
        kwargs_53575 = {}
        # Getting the type of 'self' (line 82)
        self_53572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'self', False)
        # Obtaining the member 'run_command' of a type (line 82)
        run_command_53573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), self_53572, 'run_command')
        # Calling run_command(args, kwargs) (line 82)
        run_command_call_result_53576 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), run_command_53573, *[str_53574], **kwargs_53575)
        
        
        
        # Call to has_c_libraries(...): (line 84)
        # Processing the call keyword arguments (line 84)
        kwargs_53580 = {}
        # Getting the type of 'self' (line 84)
        self_53577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 11), 'self', False)
        # Obtaining the member 'distribution' of a type (line 84)
        distribution_53578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 11), self_53577, 'distribution')
        # Obtaining the member 'has_c_libraries' of a type (line 84)
        has_c_libraries_53579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 11), distribution_53578, 'has_c_libraries')
        # Calling has_c_libraries(args, kwargs) (line 84)
        has_c_libraries_call_result_53581 = invoke(stypy.reporting.localization.Localization(__file__, 84, 11), has_c_libraries_53579, *[], **kwargs_53580)
        
        # Testing the type of an if condition (line 84)
        if_condition_53582 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 8), has_c_libraries_call_result_53581)
        # Assigning a type to the variable 'if_condition_53582' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'if_condition_53582', if_condition_53582)
        # SSA begins for if statement (line 84)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 85)
        self_53583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 15), 'self')
        # Obtaining the member 'inplace' of a type (line 85)
        inplace_53584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 15), self_53583, 'inplace')
        # Testing the type of an if condition (line 85)
        if_condition_53585 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 12), inplace_53584)
        # Assigning a type to the variable 'if_condition_53585' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'if_condition_53585', if_condition_53585)
        # SSA begins for if statement (line 85)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to get(...): (line 86)
        # Processing the call arguments (line 86)
        str_53590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 50), 'str', 'build_clib')
        # Processing the call keyword arguments (line 86)
        kwargs_53591 = {}
        # Getting the type of 'self' (line 86)
        self_53586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'self', False)
        # Obtaining the member 'distribution' of a type (line 86)
        distribution_53587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 19), self_53586, 'distribution')
        # Obtaining the member 'have_run' of a type (line 86)
        have_run_53588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 19), distribution_53587, 'have_run')
        # Obtaining the member 'get' of a type (line 86)
        get_53589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 19), have_run_53588, 'get')
        # Calling get(args, kwargs) (line 86)
        get_call_result_53592 = invoke(stypy.reporting.localization.Localization(__file__, 86, 19), get_53589, *[str_53590], **kwargs_53591)
        
        # Testing the type of an if condition (line 86)
        if_condition_53593 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 16), get_call_result_53592)
        # Assigning a type to the variable 'if_condition_53593' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'if_condition_53593', if_condition_53593)
        # SSA begins for if statement (line 86)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 87)
        # Processing the call arguments (line 87)
        str_53596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 29), 'str', 'build_clib already run, it is too late to ensure in-place build of build_clib')
        # Processing the call keyword arguments (line 87)
        kwargs_53597 = {}
        # Getting the type of 'log' (line 87)
        log_53594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'log', False)
        # Obtaining the member 'warn' of a type (line 87)
        warn_53595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 20), log_53594, 'warn')
        # Calling warn(args, kwargs) (line 87)
        warn_call_result_53598 = invoke(stypy.reporting.localization.Localization(__file__, 87, 20), warn_53595, *[str_53596], **kwargs_53597)
        
        
        # Assigning a Call to a Name (line 89):
        
        # Assigning a Call to a Name (line 89):
        
        # Call to get_command_obj(...): (line 89)
        # Processing the call arguments (line 89)
        str_53602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 67), 'str', 'build_clib')
        # Processing the call keyword arguments (line 89)
        kwargs_53603 = {}
        # Getting the type of 'self' (line 89)
        self_53599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 33), 'self', False)
        # Obtaining the member 'distribution' of a type (line 89)
        distribution_53600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 33), self_53599, 'distribution')
        # Obtaining the member 'get_command_obj' of a type (line 89)
        get_command_obj_53601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 33), distribution_53600, 'get_command_obj')
        # Calling get_command_obj(args, kwargs) (line 89)
        get_command_obj_call_result_53604 = invoke(stypy.reporting.localization.Localization(__file__, 89, 33), get_command_obj_53601, *[str_53602], **kwargs_53603)
        
        # Assigning a type to the variable 'build_clib' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 20), 'build_clib', get_command_obj_call_result_53604)
        # SSA branch for the else part of an if statement (line 86)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 91):
        
        # Assigning a Call to a Name (line 91):
        
        # Call to get_command_obj(...): (line 91)
        # Processing the call arguments (line 91)
        str_53608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 67), 'str', 'build_clib')
        # Processing the call keyword arguments (line 91)
        kwargs_53609 = {}
        # Getting the type of 'self' (line 91)
        self_53605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 33), 'self', False)
        # Obtaining the member 'distribution' of a type (line 91)
        distribution_53606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 33), self_53605, 'distribution')
        # Obtaining the member 'get_command_obj' of a type (line 91)
        get_command_obj_53607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 33), distribution_53606, 'get_command_obj')
        # Calling get_command_obj(args, kwargs) (line 91)
        get_command_obj_call_result_53610 = invoke(stypy.reporting.localization.Localization(__file__, 91, 33), get_command_obj_53607, *[str_53608], **kwargs_53609)
        
        # Assigning a type to the variable 'build_clib' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'build_clib', get_command_obj_call_result_53610)
        
        # Assigning a Num to a Attribute (line 92):
        
        # Assigning a Num to a Attribute (line 92):
        int_53611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 41), 'int')
        # Getting the type of 'build_clib' (line 92)
        build_clib_53612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 20), 'build_clib')
        # Setting the type of the member 'inplace' of a type (line 92)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 20), build_clib_53612, 'inplace', int_53611)
        
        # Call to ensure_finalized(...): (line 93)
        # Processing the call keyword arguments (line 93)
        kwargs_53615 = {}
        # Getting the type of 'build_clib' (line 93)
        build_clib_53613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 20), 'build_clib', False)
        # Obtaining the member 'ensure_finalized' of a type (line 93)
        ensure_finalized_53614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 20), build_clib_53613, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 93)
        ensure_finalized_call_result_53616 = invoke(stypy.reporting.localization.Localization(__file__, 93, 20), ensure_finalized_53614, *[], **kwargs_53615)
        
        
        # Call to run(...): (line 94)
        # Processing the call keyword arguments (line 94)
        kwargs_53619 = {}
        # Getting the type of 'build_clib' (line 94)
        build_clib_53617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 20), 'build_clib', False)
        # Obtaining the member 'run' of a type (line 94)
        run_53618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 20), build_clib_53617, 'run')
        # Calling run(args, kwargs) (line 94)
        run_call_result_53620 = invoke(stypy.reporting.localization.Localization(__file__, 94, 20), run_53618, *[], **kwargs_53619)
        
        
        # Assigning a Num to a Subscript (line 95):
        
        # Assigning a Num to a Subscript (line 95):
        int_53621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 63), 'int')
        # Getting the type of 'self' (line 95)
        self_53622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 20), 'self')
        # Obtaining the member 'distribution' of a type (line 95)
        distribution_53623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 20), self_53622, 'distribution')
        # Obtaining the member 'have_run' of a type (line 95)
        have_run_53624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 20), distribution_53623, 'have_run')
        str_53625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 47), 'str', 'build_clib')
        # Storing an element on a container (line 95)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 20), have_run_53624, (str_53625, int_53621))
        # SSA join for if statement (line 86)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 85)
        module_type_store.open_ssa_branch('else')
        
        # Call to run_command(...): (line 98)
        # Processing the call arguments (line 98)
        str_53628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 33), 'str', 'build_clib')
        # Processing the call keyword arguments (line 98)
        kwargs_53629 = {}
        # Getting the type of 'self' (line 98)
        self_53626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 16), 'self', False)
        # Obtaining the member 'run_command' of a type (line 98)
        run_command_53627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 16), self_53626, 'run_command')
        # Calling run_command(args, kwargs) (line 98)
        run_command_call_result_53630 = invoke(stypy.reporting.localization.Localization(__file__, 98, 16), run_command_53627, *[str_53628], **kwargs_53629)
        
        
        # Assigning a Call to a Name (line 99):
        
        # Assigning a Call to a Name (line 99):
        
        # Call to get_finalized_command(...): (line 99)
        # Processing the call arguments (line 99)
        str_53633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 56), 'str', 'build_clib')
        # Processing the call keyword arguments (line 99)
        kwargs_53634 = {}
        # Getting the type of 'self' (line 99)
        self_53631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 29), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 99)
        get_finalized_command_53632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 29), self_53631, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 99)
        get_finalized_command_call_result_53635 = invoke(stypy.reporting.localization.Localization(__file__, 99, 29), get_finalized_command_53632, *[str_53633], **kwargs_53634)
        
        # Assigning a type to the variable 'build_clib' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'build_clib', get_finalized_command_call_result_53635)
        # SSA join for if statement (line 85)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'build_clib' (line 100)
        build_clib_53639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 37), 'build_clib', False)
        # Obtaining the member 'build_clib' of a type (line 100)
        build_clib_53640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 37), build_clib_53639, 'build_clib')
        # Processing the call keyword arguments (line 100)
        kwargs_53641 = {}
        # Getting the type of 'self' (line 100)
        self_53636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'self', False)
        # Obtaining the member 'library_dirs' of a type (line 100)
        library_dirs_53637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), self_53636, 'library_dirs')
        # Obtaining the member 'append' of a type (line 100)
        append_53638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), library_dirs_53637, 'append')
        # Calling append(args, kwargs) (line 100)
        append_call_result_53642 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), append_53638, *[build_clib_53640], **kwargs_53641)
        
        # SSA branch for the else part of an if statement (line 84)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 102):
        
        # Assigning a Name to a Name (line 102):
        # Getting the type of 'None' (line 102)
        None_53643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 25), 'None')
        # Assigning a type to the variable 'build_clib' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'build_clib', None_53643)
        # SSA join for if statement (line 84)
        module_type_store = module_type_store.join_ssa_context()
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 109, 8))
        
        # 'from distutils.ccompiler import new_compiler' statement (line 109)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
        import_53644 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 109, 8), 'distutils.ccompiler')

        if (type(import_53644) is not StypyTypeError):

            if (import_53644 != 'pyd_module'):
                __import__(import_53644)
                sys_modules_53645 = sys.modules[import_53644]
                import_from_module(stypy.reporting.localization.Localization(__file__, 109, 8), 'distutils.ccompiler', sys_modules_53645.module_type_store, module_type_store, ['new_compiler'])
                nest_module(stypy.reporting.localization.Localization(__file__, 109, 8), __file__, sys_modules_53645, sys_modules_53645.module_type_store, module_type_store)
            else:
                from distutils.ccompiler import new_compiler

                import_from_module(stypy.reporting.localization.Localization(__file__, 109, 8), 'distutils.ccompiler', None, module_type_store, ['new_compiler'], [new_compiler])

        else:
            # Assigning a type to the variable 'distutils.ccompiler' (line 109)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'distutils.ccompiler', import_53644)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 110, 8))
        
        # 'from numpy.distutils.fcompiler import new_fcompiler' statement (line 110)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
        import_53646 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 110, 8), 'numpy.distutils.fcompiler')

        if (type(import_53646) is not StypyTypeError):

            if (import_53646 != 'pyd_module'):
                __import__(import_53646)
                sys_modules_53647 = sys.modules[import_53646]
                import_from_module(stypy.reporting.localization.Localization(__file__, 110, 8), 'numpy.distutils.fcompiler', sys_modules_53647.module_type_store, module_type_store, ['new_fcompiler'])
                nest_module(stypy.reporting.localization.Localization(__file__, 110, 8), __file__, sys_modules_53647, sys_modules_53647.module_type_store, module_type_store)
            else:
                from numpy.distutils.fcompiler import new_fcompiler

                import_from_module(stypy.reporting.localization.Localization(__file__, 110, 8), 'numpy.distutils.fcompiler', None, module_type_store, ['new_fcompiler'], [new_fcompiler])

        else:
            # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 110)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'numpy.distutils.fcompiler', import_53646)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')
        
        
        # Assigning a Attribute to a Name (line 112):
        
        # Assigning a Attribute to a Name (line 112):
        # Getting the type of 'self' (line 112)
        self_53648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 24), 'self')
        # Obtaining the member 'compiler' of a type (line 112)
        compiler_53649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 24), self_53648, 'compiler')
        # Assigning a type to the variable 'compiler_type' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'compiler_type', compiler_53649)
        
        # Assigning a Call to a Attribute (line 114):
        
        # Assigning a Call to a Attribute (line 114):
        
        # Call to new_compiler(...): (line 114)
        # Processing the call keyword arguments (line 114)
        # Getting the type of 'compiler_type' (line 114)
        compiler_type_53651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 46), 'compiler_type', False)
        keyword_53652 = compiler_type_53651
        # Getting the type of 'self' (line 115)
        self_53653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 45), 'self', False)
        # Obtaining the member 'verbose' of a type (line 115)
        verbose_53654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 45), self_53653, 'verbose')
        keyword_53655 = verbose_53654
        # Getting the type of 'self' (line 116)
        self_53656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 45), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 116)
        dry_run_53657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 45), self_53656, 'dry_run')
        keyword_53658 = dry_run_53657
        # Getting the type of 'self' (line 117)
        self_53659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 43), 'self', False)
        # Obtaining the member 'force' of a type (line 117)
        force_53660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 43), self_53659, 'force')
        keyword_53661 = force_53660
        kwargs_53662 = {'force': keyword_53661, 'verbose': keyword_53655, 'dry_run': keyword_53658, 'compiler': keyword_53652}
        # Getting the type of 'new_compiler' (line 114)
        new_compiler_53650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 24), 'new_compiler', False)
        # Calling new_compiler(args, kwargs) (line 114)
        new_compiler_call_result_53663 = invoke(stypy.reporting.localization.Localization(__file__, 114, 24), new_compiler_53650, *[], **kwargs_53662)
        
        # Getting the type of 'self' (line 114)
        self_53664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'self')
        # Setting the type of the member 'compiler' of a type (line 114)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), self_53664, 'compiler', new_compiler_call_result_53663)
        
        # Call to customize(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'self' (line 118)
        self_53668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 32), 'self', False)
        # Obtaining the member 'distribution' of a type (line 118)
        distribution_53669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 32), self_53668, 'distribution')
        # Processing the call keyword arguments (line 118)
        kwargs_53670 = {}
        # Getting the type of 'self' (line 118)
        self_53665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'self', False)
        # Obtaining the member 'compiler' of a type (line 118)
        compiler_53666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), self_53665, 'compiler')
        # Obtaining the member 'customize' of a type (line 118)
        customize_53667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), compiler_53666, 'customize')
        # Calling customize(args, kwargs) (line 118)
        customize_call_result_53671 = invoke(stypy.reporting.localization.Localization(__file__, 118, 8), customize_53667, *[distribution_53669], **kwargs_53670)
        
        
        # Call to customize_cmd(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'self' (line 119)
        self_53675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 36), 'self', False)
        # Processing the call keyword arguments (line 119)
        kwargs_53676 = {}
        # Getting the type of 'self' (line 119)
        self_53672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'self', False)
        # Obtaining the member 'compiler' of a type (line 119)
        compiler_53673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), self_53672, 'compiler')
        # Obtaining the member 'customize_cmd' of a type (line 119)
        customize_cmd_53674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), compiler_53673, 'customize_cmd')
        # Calling customize_cmd(args, kwargs) (line 119)
        customize_cmd_call_result_53677 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), customize_cmd_53674, *[self_53675], **kwargs_53676)
        
        
        # Call to show_customization(...): (line 120)
        # Processing the call keyword arguments (line 120)
        kwargs_53681 = {}
        # Getting the type of 'self' (line 120)
        self_53678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'self', False)
        # Obtaining the member 'compiler' of a type (line 120)
        compiler_53679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), self_53678, 'compiler')
        # Obtaining the member 'show_customization' of a type (line 120)
        show_customization_53680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), compiler_53679, 'show_customization')
        # Calling show_customization(args, kwargs) (line 120)
        show_customization_call_result_53682 = invoke(stypy.reporting.localization.Localization(__file__, 120, 8), show_customization_53680, *[], **kwargs_53681)
        
        
        # Assigning a Dict to a Name (line 123):
        
        # Assigning a Dict to a Name (line 123):
        
        # Obtaining an instance of the builtin type 'dict' (line 123)
        dict_53683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 16), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 123)
        
        # Assigning a type to the variable 'clibs' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'clibs', dict_53683)
        
        # Type idiom detected: calculating its left and rigth part (line 124)
        # Getting the type of 'build_clib' (line 124)
        build_clib_53684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'build_clib')
        # Getting the type of 'None' (line 124)
        None_53685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 29), 'None')
        
        (may_be_53686, more_types_in_union_53687) = may_not_be_none(build_clib_53684, None_53685)

        if may_be_53686:

            if more_types_in_union_53687:
                # Runtime conditional SSA (line 124)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Evaluating a boolean operation
            # Getting the type of 'build_clib' (line 125)
            build_clib_53688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 39), 'build_clib')
            # Obtaining the member 'libraries' of a type (line 125)
            libraries_53689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 39), build_clib_53688, 'libraries')
            
            # Obtaining an instance of the builtin type 'list' (line 125)
            list_53690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 63), 'list')
            # Adding type elements to the builtin type 'list' instance (line 125)
            
            # Applying the binary operator 'or' (line 125)
            result_or_keyword_53691 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 39), 'or', libraries_53689, list_53690)
            
            # Testing the type of a for loop iterable (line 125)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 125, 12), result_or_keyword_53691)
            # Getting the type of the for loop variable (line 125)
            for_loop_var_53692 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 125, 12), result_or_keyword_53691)
            # Assigning a type to the variable 'libname' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'libname', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 12), for_loop_var_53692))
            # Assigning a type to the variable 'build_info' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'build_info', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 12), for_loop_var_53692))
            # SSA begins for a for statement (line 125)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'libname' (line 126)
            libname_53693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 19), 'libname')
            # Getting the type of 'clibs' (line 126)
            clibs_53694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 30), 'clibs')
            # Applying the binary operator 'in' (line 126)
            result_contains_53695 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 19), 'in', libname_53693, clibs_53694)
            
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'libname' (line 126)
            libname_53696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 46), 'libname')
            # Getting the type of 'clibs' (line 126)
            clibs_53697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 40), 'clibs')
            # Obtaining the member '__getitem__' of a type (line 126)
            getitem___53698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 40), clibs_53697, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 126)
            subscript_call_result_53699 = invoke(stypy.reporting.localization.Localization(__file__, 126, 40), getitem___53698, libname_53696)
            
            # Getting the type of 'build_info' (line 126)
            build_info_53700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 58), 'build_info')
            # Applying the binary operator '!=' (line 126)
            result_ne_53701 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 40), '!=', subscript_call_result_53699, build_info_53700)
            
            # Applying the binary operator 'and' (line 126)
            result_and_keyword_53702 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 19), 'and', result_contains_53695, result_ne_53701)
            
            # Testing the type of an if condition (line 126)
            if_condition_53703 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 126, 16), result_and_keyword_53702)
            # Assigning a type to the variable 'if_condition_53703' (line 126)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'if_condition_53703', if_condition_53703)
            # SSA begins for if statement (line 126)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to warn(...): (line 127)
            # Processing the call arguments (line 127)
            str_53706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 29), 'str', 'library %r defined more than once, overwriting build_info\n%s... \nwith\n%s...')
            
            # Obtaining an instance of the builtin type 'tuple' (line 129)
            tuple_53707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 32), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 129)
            # Adding element type (line 129)
            # Getting the type of 'libname' (line 129)
            libname_53708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 32), 'libname', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 32), tuple_53707, libname_53708)
            # Adding element type (line 129)
            
            # Obtaining the type of the subscript
            int_53709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 63), 'int')
            slice_53710 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 129, 41), None, int_53709, None)
            
            # Call to repr(...): (line 129)
            # Processing the call arguments (line 129)
            
            # Obtaining the type of the subscript
            # Getting the type of 'libname' (line 129)
            libname_53712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 52), 'libname', False)
            # Getting the type of 'clibs' (line 129)
            clibs_53713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 46), 'clibs', False)
            # Obtaining the member '__getitem__' of a type (line 129)
            getitem___53714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 46), clibs_53713, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 129)
            subscript_call_result_53715 = invoke(stypy.reporting.localization.Localization(__file__, 129, 46), getitem___53714, libname_53712)
            
            # Processing the call keyword arguments (line 129)
            kwargs_53716 = {}
            # Getting the type of 'repr' (line 129)
            repr_53711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 41), 'repr', False)
            # Calling repr(args, kwargs) (line 129)
            repr_call_result_53717 = invoke(stypy.reporting.localization.Localization(__file__, 129, 41), repr_53711, *[subscript_call_result_53715], **kwargs_53716)
            
            # Obtaining the member '__getitem__' of a type (line 129)
            getitem___53718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 41), repr_call_result_53717, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 129)
            subscript_call_result_53719 = invoke(stypy.reporting.localization.Localization(__file__, 129, 41), getitem___53718, slice_53710)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 32), tuple_53707, subscript_call_result_53719)
            # Adding element type (line 129)
            
            # Obtaining the type of the subscript
            int_53720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 87), 'int')
            slice_53721 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 129, 69), None, int_53720, None)
            
            # Call to repr(...): (line 129)
            # Processing the call arguments (line 129)
            # Getting the type of 'build_info' (line 129)
            build_info_53723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 74), 'build_info', False)
            # Processing the call keyword arguments (line 129)
            kwargs_53724 = {}
            # Getting the type of 'repr' (line 129)
            repr_53722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 69), 'repr', False)
            # Calling repr(args, kwargs) (line 129)
            repr_call_result_53725 = invoke(stypy.reporting.localization.Localization(__file__, 129, 69), repr_53722, *[build_info_53723], **kwargs_53724)
            
            # Obtaining the member '__getitem__' of a type (line 129)
            getitem___53726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 69), repr_call_result_53725, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 129)
            subscript_call_result_53727 = invoke(stypy.reporting.localization.Localization(__file__, 129, 69), getitem___53726, slice_53721)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 32), tuple_53707, subscript_call_result_53727)
            
            # Applying the binary operator '%' (line 127)
            result_mod_53728 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 29), '%', str_53706, tuple_53707)
            
            # Processing the call keyword arguments (line 127)
            kwargs_53729 = {}
            # Getting the type of 'log' (line 127)
            log_53704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 20), 'log', False)
            # Obtaining the member 'warn' of a type (line 127)
            warn_53705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 20), log_53704, 'warn')
            # Calling warn(args, kwargs) (line 127)
            warn_call_result_53730 = invoke(stypy.reporting.localization.Localization(__file__, 127, 20), warn_53705, *[result_mod_53728], **kwargs_53729)
            
            # SSA join for if statement (line 126)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Name to a Subscript (line 130):
            
            # Assigning a Name to a Subscript (line 130):
            # Getting the type of 'build_info' (line 130)
            build_info_53731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 33), 'build_info')
            # Getting the type of 'clibs' (line 130)
            clibs_53732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'clibs')
            # Getting the type of 'libname' (line 130)
            libname_53733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 22), 'libname')
            # Storing an element on a container (line 130)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 16), clibs_53732, (libname_53733, build_info_53731))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_53687:
                # SSA join for if statement (line 124)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 132)
        self_53734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 35), 'self')
        # Obtaining the member 'distribution' of a type (line 132)
        distribution_53735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 35), self_53734, 'distribution')
        # Obtaining the member 'libraries' of a type (line 132)
        libraries_53736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 35), distribution_53735, 'libraries')
        
        # Obtaining an instance of the builtin type 'list' (line 132)
        list_53737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 66), 'list')
        # Adding type elements to the builtin type 'list' instance (line 132)
        
        # Applying the binary operator 'or' (line 132)
        result_or_keyword_53738 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 35), 'or', libraries_53736, list_53737)
        
        # Testing the type of a for loop iterable (line 132)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 132, 8), result_or_keyword_53738)
        # Getting the type of the for loop variable (line 132)
        for_loop_var_53739 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 132, 8), result_or_keyword_53738)
        # Assigning a type to the variable 'libname' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'libname', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 8), for_loop_var_53739))
        # Assigning a type to the variable 'build_info' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'build_info', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 8), for_loop_var_53739))
        # SSA begins for a for statement (line 132)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'libname' (line 133)
        libname_53740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), 'libname')
        # Getting the type of 'clibs' (line 133)
        clibs_53741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 26), 'clibs')
        # Applying the binary operator 'in' (line 133)
        result_contains_53742 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 15), 'in', libname_53740, clibs_53741)
        
        # Testing the type of an if condition (line 133)
        if_condition_53743 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 133, 12), result_contains_53742)
        # Assigning a type to the variable 'if_condition_53743' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'if_condition_53743', if_condition_53743)
        # SSA begins for if statement (line 133)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 133)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 136):
        
        # Assigning a Name to a Subscript (line 136):
        # Getting the type of 'build_info' (line 136)
        build_info_53744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 29), 'build_info')
        # Getting the type of 'clibs' (line 136)
        clibs_53745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'clibs')
        # Getting the type of 'libname' (line 136)
        libname_53746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 18), 'libname')
        # Storing an element on a container (line 136)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 12), clibs_53745, (libname_53746, build_info_53744))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 140):
        
        # Assigning a Call to a Name (line 140):
        
        # Call to set(...): (line 140)
        # Processing the call keyword arguments (line 140)
        kwargs_53748 = {}
        # Getting the type of 'set' (line 140)
        set_53747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 24), 'set', False)
        # Calling set(args, kwargs) (line 140)
        set_call_result_53749 = invoke(stypy.reporting.localization.Localization(__file__, 140, 24), set_53747, *[], **kwargs_53748)
        
        # Assigning a type to the variable 'all_languages' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'all_languages', set_call_result_53749)
        
        # Getting the type of 'self' (line 141)
        self_53750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 19), 'self')
        # Obtaining the member 'extensions' of a type (line 141)
        extensions_53751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 19), self_53750, 'extensions')
        # Testing the type of a for loop iterable (line 141)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 141, 8), extensions_53751)
        # Getting the type of the for loop variable (line 141)
        for_loop_var_53752 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 141, 8), extensions_53751)
        # Assigning a type to the variable 'ext' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'ext', for_loop_var_53752)
        # SSA begins for a for statement (line 141)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 142):
        
        # Assigning a Call to a Name (line 142):
        
        # Call to set(...): (line 142)
        # Processing the call keyword arguments (line 142)
        kwargs_53754 = {}
        # Getting the type of 'set' (line 142)
        set_53753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 28), 'set', False)
        # Calling set(args, kwargs) (line 142)
        set_call_result_53755 = invoke(stypy.reporting.localization.Localization(__file__, 142, 28), set_53753, *[], **kwargs_53754)
        
        # Assigning a type to the variable 'ext_languages' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'ext_languages', set_call_result_53755)
        
        # Assigning a List to a Name (line 143):
        
        # Assigning a List to a Name (line 143):
        
        # Obtaining an instance of the builtin type 'list' (line 143)
        list_53756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 143)
        
        # Assigning a type to the variable 'c_libs' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'c_libs', list_53756)
        
        # Assigning a List to a Name (line 144):
        
        # Assigning a List to a Name (line 144):
        
        # Obtaining an instance of the builtin type 'list' (line 144)
        list_53757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 144)
        
        # Assigning a type to the variable 'c_lib_dirs' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'c_lib_dirs', list_53757)
        
        # Assigning a List to a Name (line 145):
        
        # Assigning a List to a Name (line 145):
        
        # Obtaining an instance of the builtin type 'list' (line 145)
        list_53758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 145)
        
        # Assigning a type to the variable 'macros' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'macros', list_53758)
        
        # Getting the type of 'ext' (line 146)
        ext_53759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 27), 'ext')
        # Obtaining the member 'libraries' of a type (line 146)
        libraries_53760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 27), ext_53759, 'libraries')
        # Testing the type of a for loop iterable (line 146)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 146, 12), libraries_53760)
        # Getting the type of the for loop variable (line 146)
        for_loop_var_53761 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 146, 12), libraries_53760)
        # Assigning a type to the variable 'libname' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'libname', for_loop_var_53761)
        # SSA begins for a for statement (line 146)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'libname' (line 147)
        libname_53762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 19), 'libname')
        # Getting the type of 'clibs' (line 147)
        clibs_53763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 30), 'clibs')
        # Applying the binary operator 'in' (line 147)
        result_contains_53764 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 19), 'in', libname_53762, clibs_53763)
        
        # Testing the type of an if condition (line 147)
        if_condition_53765 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 16), result_contains_53764)
        # Assigning a type to the variable 'if_condition_53765' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'if_condition_53765', if_condition_53765)
        # SSA begins for if statement (line 147)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 148):
        
        # Assigning a Subscript to a Name (line 148):
        
        # Obtaining the type of the subscript
        # Getting the type of 'libname' (line 148)
        libname_53766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 34), 'libname')
        # Getting the type of 'clibs' (line 148)
        clibs_53767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 28), 'clibs')
        # Obtaining the member '__getitem__' of a type (line 148)
        getitem___53768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 28), clibs_53767, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 148)
        subscript_call_result_53769 = invoke(stypy.reporting.localization.Localization(__file__, 148, 28), getitem___53768, libname_53766)
        
        # Assigning a type to the variable 'binfo' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 20), 'binfo', subscript_call_result_53769)
        
        # Getting the type of 'c_libs' (line 149)
        c_libs_53770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 20), 'c_libs')
        
        # Call to get(...): (line 149)
        # Processing the call arguments (line 149)
        str_53773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 40), 'str', 'libraries')
        
        # Obtaining an instance of the builtin type 'list' (line 149)
        list_53774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 149)
        
        # Processing the call keyword arguments (line 149)
        kwargs_53775 = {}
        # Getting the type of 'binfo' (line 149)
        binfo_53771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 30), 'binfo', False)
        # Obtaining the member 'get' of a type (line 149)
        get_53772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 30), binfo_53771, 'get')
        # Calling get(args, kwargs) (line 149)
        get_call_result_53776 = invoke(stypy.reporting.localization.Localization(__file__, 149, 30), get_53772, *[str_53773, list_53774], **kwargs_53775)
        
        # Applying the binary operator '+=' (line 149)
        result_iadd_53777 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 20), '+=', c_libs_53770, get_call_result_53776)
        # Assigning a type to the variable 'c_libs' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 20), 'c_libs', result_iadd_53777)
        
        
        # Getting the type of 'c_lib_dirs' (line 150)
        c_lib_dirs_53778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 20), 'c_lib_dirs')
        
        # Call to get(...): (line 150)
        # Processing the call arguments (line 150)
        str_53781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 44), 'str', 'library_dirs')
        
        # Obtaining an instance of the builtin type 'list' (line 150)
        list_53782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 60), 'list')
        # Adding type elements to the builtin type 'list' instance (line 150)
        
        # Processing the call keyword arguments (line 150)
        kwargs_53783 = {}
        # Getting the type of 'binfo' (line 150)
        binfo_53779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 34), 'binfo', False)
        # Obtaining the member 'get' of a type (line 150)
        get_53780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 34), binfo_53779, 'get')
        # Calling get(args, kwargs) (line 150)
        get_call_result_53784 = invoke(stypy.reporting.localization.Localization(__file__, 150, 34), get_53780, *[str_53781, list_53782], **kwargs_53783)
        
        # Applying the binary operator '+=' (line 150)
        result_iadd_53785 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 20), '+=', c_lib_dirs_53778, get_call_result_53784)
        # Assigning a type to the variable 'c_lib_dirs' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 20), 'c_lib_dirs', result_iadd_53785)
        
        
        
        # Call to get(...): (line 151)
        # Processing the call arguments (line 151)
        str_53788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 39), 'str', 'macros')
        
        # Obtaining an instance of the builtin type 'list' (line 151)
        list_53789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 151)
        
        # Processing the call keyword arguments (line 151)
        kwargs_53790 = {}
        # Getting the type of 'binfo' (line 151)
        binfo_53786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 29), 'binfo', False)
        # Obtaining the member 'get' of a type (line 151)
        get_53787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 29), binfo_53786, 'get')
        # Calling get(args, kwargs) (line 151)
        get_call_result_53791 = invoke(stypy.reporting.localization.Localization(__file__, 151, 29), get_53787, *[str_53788, list_53789], **kwargs_53790)
        
        # Testing the type of a for loop iterable (line 151)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 151, 20), get_call_result_53791)
        # Getting the type of the for loop variable (line 151)
        for_loop_var_53792 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 151, 20), get_call_result_53791)
        # Assigning a type to the variable 'm' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 20), 'm', for_loop_var_53792)
        # SSA begins for a for statement (line 151)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'm' (line 152)
        m_53793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 27), 'm')
        # Getting the type of 'macros' (line 152)
        macros_53794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 36), 'macros')
        # Applying the binary operator 'notin' (line 152)
        result_contains_53795 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 27), 'notin', m_53793, macros_53794)
        
        # Testing the type of an if condition (line 152)
        if_condition_53796 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 24), result_contains_53795)
        # Assigning a type to the variable 'if_condition_53796' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 24), 'if_condition_53796', if_condition_53796)
        # SSA begins for if statement (line 152)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'm' (line 153)
        m_53799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 42), 'm', False)
        # Processing the call keyword arguments (line 153)
        kwargs_53800 = {}
        # Getting the type of 'macros' (line 153)
        macros_53797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 28), 'macros', False)
        # Obtaining the member 'append' of a type (line 153)
        append_53798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 28), macros_53797, 'append')
        # Calling append(args, kwargs) (line 153)
        append_call_result_53801 = invoke(stypy.reporting.localization.Localization(__file__, 153, 28), append_53798, *[m_53799], **kwargs_53800)
        
        # SSA join for if statement (line 152)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 147)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to get(...): (line 155)
        # Processing the call arguments (line 155)
        str_53809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 52), 'str', 'source_languages')
        
        # Obtaining an instance of the builtin type 'list' (line 155)
        list_53810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 72), 'list')
        # Adding type elements to the builtin type 'list' instance (line 155)
        
        # Processing the call keyword arguments (line 155)
        kwargs_53811 = {}
        
        # Call to get(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'libname' (line 155)
        libname_53804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 35), 'libname', False)
        
        # Obtaining an instance of the builtin type 'dict' (line 155)
        dict_53805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 44), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 155)
        
        # Processing the call keyword arguments (line 155)
        kwargs_53806 = {}
        # Getting the type of 'clibs' (line 155)
        clibs_53802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 25), 'clibs', False)
        # Obtaining the member 'get' of a type (line 155)
        get_53803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 25), clibs_53802, 'get')
        # Calling get(args, kwargs) (line 155)
        get_call_result_53807 = invoke(stypy.reporting.localization.Localization(__file__, 155, 25), get_53803, *[libname_53804, dict_53805], **kwargs_53806)
        
        # Obtaining the member 'get' of a type (line 155)
        get_53808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 25), get_call_result_53807, 'get')
        # Calling get(args, kwargs) (line 155)
        get_call_result_53812 = invoke(stypy.reporting.localization.Localization(__file__, 155, 25), get_53808, *[str_53809, list_53810], **kwargs_53811)
        
        # Testing the type of a for loop iterable (line 155)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 155, 16), get_call_result_53812)
        # Getting the type of the for loop variable (line 155)
        for_loop_var_53813 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 155, 16), get_call_result_53812)
        # Assigning a type to the variable 'l' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'l', for_loop_var_53813)
        # SSA begins for a for statement (line 155)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to add(...): (line 156)
        # Processing the call arguments (line 156)
        # Getting the type of 'l' (line 156)
        l_53816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 38), 'l', False)
        # Processing the call keyword arguments (line 156)
        kwargs_53817 = {}
        # Getting the type of 'ext_languages' (line 156)
        ext_languages_53814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 20), 'ext_languages', False)
        # Obtaining the member 'add' of a type (line 156)
        add_53815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 20), ext_languages_53814, 'add')
        # Calling add(args, kwargs) (line 156)
        add_call_result_53818 = invoke(stypy.reporting.localization.Localization(__file__, 156, 20), add_53815, *[l_53816], **kwargs_53817)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'c_libs' (line 157)
        c_libs_53819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 15), 'c_libs')
        # Testing the type of an if condition (line 157)
        if_condition_53820 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 157, 12), c_libs_53819)
        # Assigning a type to the variable 'if_condition_53820' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'if_condition_53820', if_condition_53820)
        # SSA begins for if statement (line 157)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 158):
        
        # Assigning a BinOp to a Name (line 158):
        # Getting the type of 'ext' (line 158)
        ext_53821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 29), 'ext')
        # Obtaining the member 'libraries' of a type (line 158)
        libraries_53822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 29), ext_53821, 'libraries')
        # Getting the type of 'c_libs' (line 158)
        c_libs_53823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 45), 'c_libs')
        # Applying the binary operator '+' (line 158)
        result_add_53824 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 29), '+', libraries_53822, c_libs_53823)
        
        # Assigning a type to the variable 'new_c_libs' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'new_c_libs', result_add_53824)
        
        # Call to info(...): (line 159)
        # Processing the call arguments (line 159)
        str_53827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 25), 'str', 'updating extension %r libraries from %r to %r')
        
        # Obtaining an instance of the builtin type 'tuple' (line 160)
        tuple_53828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 160)
        # Adding element type (line 160)
        # Getting the type of 'ext' (line 160)
        ext_53829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 28), 'ext', False)
        # Obtaining the member 'name' of a type (line 160)
        name_53830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 28), ext_53829, 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 28), tuple_53828, name_53830)
        # Adding element type (line 160)
        # Getting the type of 'ext' (line 160)
        ext_53831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 38), 'ext', False)
        # Obtaining the member 'libraries' of a type (line 160)
        libraries_53832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 38), ext_53831, 'libraries')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 28), tuple_53828, libraries_53832)
        # Adding element type (line 160)
        # Getting the type of 'new_c_libs' (line 160)
        new_c_libs_53833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 53), 'new_c_libs', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 28), tuple_53828, new_c_libs_53833)
        
        # Applying the binary operator '%' (line 159)
        result_mod_53834 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 25), '%', str_53827, tuple_53828)
        
        # Processing the call keyword arguments (line 159)
        kwargs_53835 = {}
        # Getting the type of 'log' (line 159)
        log_53825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'log', False)
        # Obtaining the member 'info' of a type (line 159)
        info_53826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 16), log_53825, 'info')
        # Calling info(args, kwargs) (line 159)
        info_call_result_53836 = invoke(stypy.reporting.localization.Localization(__file__, 159, 16), info_53826, *[result_mod_53834], **kwargs_53835)
        
        
        # Assigning a Name to a Attribute (line 161):
        
        # Assigning a Name to a Attribute (line 161):
        # Getting the type of 'new_c_libs' (line 161)
        new_c_libs_53837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 32), 'new_c_libs')
        # Getting the type of 'ext' (line 161)
        ext_53838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'ext')
        # Setting the type of the member 'libraries' of a type (line 161)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 16), ext_53838, 'libraries', new_c_libs_53837)
        
        # Assigning a BinOp to a Attribute (line 162):
        
        # Assigning a BinOp to a Attribute (line 162):
        # Getting the type of 'ext' (line 162)
        ext_53839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 35), 'ext')
        # Obtaining the member 'library_dirs' of a type (line 162)
        library_dirs_53840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 35), ext_53839, 'library_dirs')
        # Getting the type of 'c_lib_dirs' (line 162)
        c_lib_dirs_53841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 54), 'c_lib_dirs')
        # Applying the binary operator '+' (line 162)
        result_add_53842 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 35), '+', library_dirs_53840, c_lib_dirs_53841)
        
        # Getting the type of 'ext' (line 162)
        ext_53843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'ext')
        # Setting the type of the member 'library_dirs' of a type (line 162)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 16), ext_53843, 'library_dirs', result_add_53842)
        # SSA join for if statement (line 157)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'macros' (line 163)
        macros_53844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 15), 'macros')
        # Testing the type of an if condition (line 163)
        if_condition_53845 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 12), macros_53844)
        # Assigning a type to the variable 'if_condition_53845' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'if_condition_53845', if_condition_53845)
        # SSA begins for if statement (line 163)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 164)
        # Processing the call arguments (line 164)
        str_53848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 25), 'str', 'extending extension %r defined_macros with %r')
        
        # Obtaining an instance of the builtin type 'tuple' (line 165)
        tuple_53849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 165)
        # Adding element type (line 165)
        # Getting the type of 'ext' (line 165)
        ext_53850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 28), 'ext', False)
        # Obtaining the member 'name' of a type (line 165)
        name_53851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 28), ext_53850, 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 28), tuple_53849, name_53851)
        # Adding element type (line 165)
        # Getting the type of 'macros' (line 165)
        macros_53852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 38), 'macros', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 28), tuple_53849, macros_53852)
        
        # Applying the binary operator '%' (line 164)
        result_mod_53853 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 25), '%', str_53848, tuple_53849)
        
        # Processing the call keyword arguments (line 164)
        kwargs_53854 = {}
        # Getting the type of 'log' (line 164)
        log_53846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'log', False)
        # Obtaining the member 'info' of a type (line 164)
        info_53847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 16), log_53846, 'info')
        # Calling info(args, kwargs) (line 164)
        info_call_result_53855 = invoke(stypy.reporting.localization.Localization(__file__, 164, 16), info_53847, *[result_mod_53853], **kwargs_53854)
        
        
        # Assigning a BinOp to a Attribute (line 166):
        
        # Assigning a BinOp to a Attribute (line 166):
        # Getting the type of 'ext' (line 166)
        ext_53856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 36), 'ext')
        # Obtaining the member 'define_macros' of a type (line 166)
        define_macros_53857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 36), ext_53856, 'define_macros')
        # Getting the type of 'macros' (line 166)
        macros_53858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 56), 'macros')
        # Applying the binary operator '+' (line 166)
        result_add_53859 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 36), '+', define_macros_53857, macros_53858)
        
        # Getting the type of 'ext' (line 166)
        ext_53860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'ext')
        # Setting the type of the member 'define_macros' of a type (line 166)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 16), ext_53860, 'define_macros', result_add_53859)
        # SSA join for if statement (line 163)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to has_f_sources(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'ext' (line 169)
        ext_53862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 29), 'ext', False)
        # Obtaining the member 'sources' of a type (line 169)
        sources_53863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 29), ext_53862, 'sources')
        # Processing the call keyword arguments (line 169)
        kwargs_53864 = {}
        # Getting the type of 'has_f_sources' (line 169)
        has_f_sources_53861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'has_f_sources', False)
        # Calling has_f_sources(args, kwargs) (line 169)
        has_f_sources_call_result_53865 = invoke(stypy.reporting.localization.Localization(__file__, 169, 15), has_f_sources_53861, *[sources_53863], **kwargs_53864)
        
        # Testing the type of an if condition (line 169)
        if_condition_53866 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 12), has_f_sources_call_result_53865)
        # Assigning a type to the variable 'if_condition_53866' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'if_condition_53866', if_condition_53866)
        # SSA begins for if statement (line 169)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to add(...): (line 170)
        # Processing the call arguments (line 170)
        str_53869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 34), 'str', 'f77')
        # Processing the call keyword arguments (line 170)
        kwargs_53870 = {}
        # Getting the type of 'ext_languages' (line 170)
        ext_languages_53867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'ext_languages', False)
        # Obtaining the member 'add' of a type (line 170)
        add_53868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 16), ext_languages_53867, 'add')
        # Calling add(args, kwargs) (line 170)
        add_call_result_53871 = invoke(stypy.reporting.localization.Localization(__file__, 170, 16), add_53868, *[str_53869], **kwargs_53870)
        
        # SSA join for if statement (line 169)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to has_cxx_sources(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'ext' (line 171)
        ext_53873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 31), 'ext', False)
        # Obtaining the member 'sources' of a type (line 171)
        sources_53874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 31), ext_53873, 'sources')
        # Processing the call keyword arguments (line 171)
        kwargs_53875 = {}
        # Getting the type of 'has_cxx_sources' (line 171)
        has_cxx_sources_53872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 15), 'has_cxx_sources', False)
        # Calling has_cxx_sources(args, kwargs) (line 171)
        has_cxx_sources_call_result_53876 = invoke(stypy.reporting.localization.Localization(__file__, 171, 15), has_cxx_sources_53872, *[sources_53874], **kwargs_53875)
        
        # Testing the type of an if condition (line 171)
        if_condition_53877 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 12), has_cxx_sources_call_result_53876)
        # Assigning a type to the variable 'if_condition_53877' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'if_condition_53877', if_condition_53877)
        # SSA begins for if statement (line 171)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to add(...): (line 172)
        # Processing the call arguments (line 172)
        str_53880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 34), 'str', 'c++')
        # Processing the call keyword arguments (line 172)
        kwargs_53881 = {}
        # Getting the type of 'ext_languages' (line 172)
        ext_languages_53878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), 'ext_languages', False)
        # Obtaining the member 'add' of a type (line 172)
        add_53879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 16), ext_languages_53878, 'add')
        # Calling add(args, kwargs) (line 172)
        add_call_result_53882 = invoke(stypy.reporting.localization.Localization(__file__, 172, 16), add_53879, *[str_53880], **kwargs_53881)
        
        # SSA join for if statement (line 171)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BoolOp to a Name (line 173):
        
        # Assigning a BoolOp to a Name (line 173):
        
        # Evaluating a boolean operation
        # Getting the type of 'ext' (line 173)
        ext_53883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'ext')
        # Obtaining the member 'language' of a type (line 173)
        language_53884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 16), ext_53883, 'language')
        
        # Call to detect_language(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'ext' (line 173)
        ext_53888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 62), 'ext', False)
        # Obtaining the member 'sources' of a type (line 173)
        sources_53889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 62), ext_53888, 'sources')
        # Processing the call keyword arguments (line 173)
        kwargs_53890 = {}
        # Getting the type of 'self' (line 173)
        self_53885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 32), 'self', False)
        # Obtaining the member 'compiler' of a type (line 173)
        compiler_53886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 32), self_53885, 'compiler')
        # Obtaining the member 'detect_language' of a type (line 173)
        detect_language_53887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 32), compiler_53886, 'detect_language')
        # Calling detect_language(args, kwargs) (line 173)
        detect_language_call_result_53891 = invoke(stypy.reporting.localization.Localization(__file__, 173, 32), detect_language_53887, *[sources_53889], **kwargs_53890)
        
        # Applying the binary operator 'or' (line 173)
        result_or_keyword_53892 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 16), 'or', language_53884, detect_language_call_result_53891)
        
        # Assigning a type to the variable 'l' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'l', result_or_keyword_53892)
        
        # Getting the type of 'l' (line 174)
        l_53893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 15), 'l')
        # Testing the type of an if condition (line 174)
        if_condition_53894 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 174, 12), l_53893)
        # Assigning a type to the variable 'if_condition_53894' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'if_condition_53894', if_condition_53894)
        # SSA begins for if statement (line 174)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to add(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'l' (line 175)
        l_53897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 34), 'l', False)
        # Processing the call keyword arguments (line 175)
        kwargs_53898 = {}
        # Getting the type of 'ext_languages' (line 175)
        ext_languages_53895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 16), 'ext_languages', False)
        # Obtaining the member 'add' of a type (line 175)
        add_53896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 16), ext_languages_53895, 'add')
        # Calling add(args, kwargs) (line 175)
        add_call_result_53899 = invoke(stypy.reporting.localization.Localization(__file__, 175, 16), add_53896, *[l_53897], **kwargs_53898)
        
        # SSA join for if statement (line 174)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        str_53900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 15), 'str', 'c++')
        # Getting the type of 'ext_languages' (line 177)
        ext_languages_53901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 24), 'ext_languages')
        # Applying the binary operator 'in' (line 177)
        result_contains_53902 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 15), 'in', str_53900, ext_languages_53901)
        
        # Testing the type of an if condition (line 177)
        if_condition_53903 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 177, 12), result_contains_53902)
        # Assigning a type to the variable 'if_condition_53903' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'if_condition_53903', if_condition_53903)
        # SSA begins for if statement (line 177)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 178):
        
        # Assigning a Str to a Name (line 178):
        str_53904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 31), 'str', 'c++')
        # Assigning a type to the variable 'ext_language' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 16), 'ext_language', str_53904)
        # SSA branch for the else part of an if statement (line 177)
        module_type_store.open_ssa_branch('else')
        
        
        str_53905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 17), 'str', 'f90')
        # Getting the type of 'ext_languages' (line 179)
        ext_languages_53906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 26), 'ext_languages')
        # Applying the binary operator 'in' (line 179)
        result_contains_53907 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 17), 'in', str_53905, ext_languages_53906)
        
        # Testing the type of an if condition (line 179)
        if_condition_53908 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 179, 17), result_contains_53907)
        # Assigning a type to the variable 'if_condition_53908' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 17), 'if_condition_53908', if_condition_53908)
        # SSA begins for if statement (line 179)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 180):
        
        # Assigning a Str to a Name (line 180):
        str_53909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 31), 'str', 'f90')
        # Assigning a type to the variable 'ext_language' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 16), 'ext_language', str_53909)
        # SSA branch for the else part of an if statement (line 179)
        module_type_store.open_ssa_branch('else')
        
        
        str_53910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 17), 'str', 'f77')
        # Getting the type of 'ext_languages' (line 181)
        ext_languages_53911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 26), 'ext_languages')
        # Applying the binary operator 'in' (line 181)
        result_contains_53912 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 17), 'in', str_53910, ext_languages_53911)
        
        # Testing the type of an if condition (line 181)
        if_condition_53913 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 181, 17), result_contains_53912)
        # Assigning a type to the variable 'if_condition_53913' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 17), 'if_condition_53913', if_condition_53913)
        # SSA begins for if statement (line 181)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 182):
        
        # Assigning a Str to a Name (line 182):
        str_53914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 31), 'str', 'f77')
        # Assigning a type to the variable 'ext_language' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 16), 'ext_language', str_53914)
        # SSA branch for the else part of an if statement (line 181)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 184):
        
        # Assigning a Str to a Name (line 184):
        str_53915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 31), 'str', 'c')
        # Assigning a type to the variable 'ext_language' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'ext_language', str_53915)
        # SSA join for if statement (line 181)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 179)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 177)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'l' (line 185)
        l_53916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 15), 'l')
        
        # Getting the type of 'l' (line 185)
        l_53917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 21), 'l')
        # Getting the type of 'ext_language' (line 185)
        ext_language_53918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 26), 'ext_language')
        # Applying the binary operator '!=' (line 185)
        result_ne_53919 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 21), '!=', l_53917, ext_language_53918)
        
        # Applying the binary operator 'and' (line 185)
        result_and_keyword_53920 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 15), 'and', l_53916, result_ne_53919)
        # Getting the type of 'ext' (line 185)
        ext_53921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 43), 'ext')
        # Obtaining the member 'language' of a type (line 185)
        language_53922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 43), ext_53921, 'language')
        # Applying the binary operator 'and' (line 185)
        result_and_keyword_53923 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 15), 'and', result_and_keyword_53920, language_53922)
        
        # Testing the type of an if condition (line 185)
        if_condition_53924 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 185, 12), result_and_keyword_53923)
        # Assigning a type to the variable 'if_condition_53924' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'if_condition_53924', if_condition_53924)
        # SSA begins for if statement (line 185)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 186)
        # Processing the call arguments (line 186)
        str_53927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 25), 'str', 'resetting extension %r language from %r to %r.')
        
        # Obtaining an instance of the builtin type 'tuple' (line 187)
        tuple_53928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 187)
        # Adding element type (line 187)
        # Getting the type of 'ext' (line 187)
        ext_53929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 26), 'ext', False)
        # Obtaining the member 'name' of a type (line 187)
        name_53930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 26), ext_53929, 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 26), tuple_53928, name_53930)
        # Adding element type (line 187)
        # Getting the type of 'l' (line 187)
        l_53931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 36), 'l', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 26), tuple_53928, l_53931)
        # Adding element type (line 187)
        # Getting the type of 'ext_language' (line 187)
        ext_language_53932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 39), 'ext_language', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 26), tuple_53928, ext_language_53932)
        
        # Applying the binary operator '%' (line 186)
        result_mod_53933 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 25), '%', str_53927, tuple_53928)
        
        # Processing the call keyword arguments (line 186)
        kwargs_53934 = {}
        # Getting the type of 'log' (line 186)
        log_53925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 16), 'log', False)
        # Obtaining the member 'warn' of a type (line 186)
        warn_53926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 16), log_53925, 'warn')
        # Calling warn(args, kwargs) (line 186)
        warn_call_result_53935 = invoke(stypy.reporting.localization.Localization(__file__, 186, 16), warn_53926, *[result_mod_53933], **kwargs_53934)
        
        # SSA join for if statement (line 185)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 188):
        
        # Assigning a Name to a Attribute (line 188):
        # Getting the type of 'ext_language' (line 188)
        ext_language_53936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 27), 'ext_language')
        # Getting the type of 'ext' (line 188)
        ext_53937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'ext')
        # Setting the type of the member 'language' of a type (line 188)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 12), ext_53937, 'language', ext_language_53936)
        
        # Call to update(...): (line 190)
        # Processing the call arguments (line 190)
        # Getting the type of 'ext_languages' (line 190)
        ext_languages_53940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 33), 'ext_languages', False)
        # Processing the call keyword arguments (line 190)
        kwargs_53941 = {}
        # Getting the type of 'all_languages' (line 190)
        all_languages_53938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'all_languages', False)
        # Obtaining the member 'update' of a type (line 190)
        update_53939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 12), all_languages_53938, 'update')
        # Calling update(args, kwargs) (line 190)
        update_call_result_53942 = invoke(stypy.reporting.localization.Localization(__file__, 190, 12), update_53939, *[ext_languages_53940], **kwargs_53941)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Compare to a Name (line 192):
        
        # Assigning a Compare to a Name (line 192):
        
        str_53943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 28), 'str', 'f90')
        # Getting the type of 'all_languages' (line 192)
        all_languages_53944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 37), 'all_languages')
        # Applying the binary operator 'in' (line 192)
        result_contains_53945 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 28), 'in', str_53943, all_languages_53944)
        
        # Assigning a type to the variable 'need_f90_compiler' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'need_f90_compiler', result_contains_53945)
        
        # Assigning a Compare to a Name (line 193):
        
        # Assigning a Compare to a Name (line 193):
        
        str_53946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 28), 'str', 'f77')
        # Getting the type of 'all_languages' (line 193)
        all_languages_53947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 37), 'all_languages')
        # Applying the binary operator 'in' (line 193)
        result_contains_53948 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 28), 'in', str_53946, all_languages_53947)
        
        # Assigning a type to the variable 'need_f77_compiler' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'need_f77_compiler', result_contains_53948)
        
        # Assigning a Compare to a Name (line 194):
        
        # Assigning a Compare to a Name (line 194):
        
        str_53949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 28), 'str', 'c++')
        # Getting the type of 'all_languages' (line 194)
        all_languages_53950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 37), 'all_languages')
        # Applying the binary operator 'in' (line 194)
        result_contains_53951 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 28), 'in', str_53949, all_languages_53950)
        
        # Assigning a type to the variable 'need_cxx_compiler' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'need_cxx_compiler', result_contains_53951)
        
        # Getting the type of 'need_cxx_compiler' (line 197)
        need_cxx_compiler_53952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 11), 'need_cxx_compiler')
        # Testing the type of an if condition (line 197)
        if_condition_53953 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 197, 8), need_cxx_compiler_53952)
        # Assigning a type to the variable 'if_condition_53953' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'if_condition_53953', if_condition_53953)
        # SSA begins for if statement (line 197)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 198):
        
        # Assigning a Call to a Attribute (line 198):
        
        # Call to new_compiler(...): (line 198)
        # Processing the call keyword arguments (line 198)
        # Getting the type of 'compiler_type' (line 198)
        compiler_type_53955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 55), 'compiler_type', False)
        keyword_53956 = compiler_type_53955
        # Getting the type of 'self' (line 199)
        self_53957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 53), 'self', False)
        # Obtaining the member 'verbose' of a type (line 199)
        verbose_53958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 53), self_53957, 'verbose')
        keyword_53959 = verbose_53958
        # Getting the type of 'self' (line 200)
        self_53960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 53), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 200)
        dry_run_53961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 53), self_53960, 'dry_run')
        keyword_53962 = dry_run_53961
        # Getting the type of 'self' (line 201)
        self_53963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 51), 'self', False)
        # Obtaining the member 'force' of a type (line 201)
        force_53964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 51), self_53963, 'force')
        keyword_53965 = force_53964
        kwargs_53966 = {'force': keyword_53965, 'verbose': keyword_53959, 'dry_run': keyword_53962, 'compiler': keyword_53956}
        # Getting the type of 'new_compiler' (line 198)
        new_compiler_53954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 33), 'new_compiler', False)
        # Calling new_compiler(args, kwargs) (line 198)
        new_compiler_call_result_53967 = invoke(stypy.reporting.localization.Localization(__file__, 198, 33), new_compiler_53954, *[], **kwargs_53966)
        
        # Getting the type of 'self' (line 198)
        self_53968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'self')
        # Setting the type of the member '_cxx_compiler' of a type (line 198)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 12), self_53968, '_cxx_compiler', new_compiler_call_result_53967)
        
        # Assigning a Attribute to a Name (line 202):
        
        # Assigning a Attribute to a Name (line 202):
        # Getting the type of 'self' (line 202)
        self_53969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 23), 'self')
        # Obtaining the member '_cxx_compiler' of a type (line 202)
        _cxx_compiler_53970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 23), self_53969, '_cxx_compiler')
        # Assigning a type to the variable 'compiler' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'compiler', _cxx_compiler_53970)
        
        # Call to customize(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'self' (line 203)
        self_53973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 31), 'self', False)
        # Obtaining the member 'distribution' of a type (line 203)
        distribution_53974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 31), self_53973, 'distribution')
        # Processing the call keyword arguments (line 203)
        # Getting the type of 'need_cxx_compiler' (line 203)
        need_cxx_compiler_53975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 59), 'need_cxx_compiler', False)
        keyword_53976 = need_cxx_compiler_53975
        kwargs_53977 = {'need_cxx': keyword_53976}
        # Getting the type of 'compiler' (line 203)
        compiler_53971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'compiler', False)
        # Obtaining the member 'customize' of a type (line 203)
        customize_53972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 12), compiler_53971, 'customize')
        # Calling customize(args, kwargs) (line 203)
        customize_call_result_53978 = invoke(stypy.reporting.localization.Localization(__file__, 203, 12), customize_53972, *[distribution_53974], **kwargs_53977)
        
        
        # Call to customize_cmd(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'self' (line 204)
        self_53981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 35), 'self', False)
        # Processing the call keyword arguments (line 204)
        kwargs_53982 = {}
        # Getting the type of 'compiler' (line 204)
        compiler_53979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'compiler', False)
        # Obtaining the member 'customize_cmd' of a type (line 204)
        customize_cmd_53980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 12), compiler_53979, 'customize_cmd')
        # Calling customize_cmd(args, kwargs) (line 204)
        customize_cmd_call_result_53983 = invoke(stypy.reporting.localization.Localization(__file__, 204, 12), customize_cmd_53980, *[self_53981], **kwargs_53982)
        
        
        # Call to show_customization(...): (line 205)
        # Processing the call keyword arguments (line 205)
        kwargs_53986 = {}
        # Getting the type of 'compiler' (line 205)
        compiler_53984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'compiler', False)
        # Obtaining the member 'show_customization' of a type (line 205)
        show_customization_53985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 12), compiler_53984, 'show_customization')
        # Calling show_customization(args, kwargs) (line 205)
        show_customization_call_result_53987 = invoke(stypy.reporting.localization.Localization(__file__, 205, 12), show_customization_53985, *[], **kwargs_53986)
        
        
        # Assigning a Call to a Attribute (line 206):
        
        # Assigning a Call to a Attribute (line 206):
        
        # Call to cxx_compiler(...): (line 206)
        # Processing the call keyword arguments (line 206)
        kwargs_53990 = {}
        # Getting the type of 'compiler' (line 206)
        compiler_53988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 33), 'compiler', False)
        # Obtaining the member 'cxx_compiler' of a type (line 206)
        cxx_compiler_53989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 33), compiler_53988, 'cxx_compiler')
        # Calling cxx_compiler(args, kwargs) (line 206)
        cxx_compiler_call_result_53991 = invoke(stypy.reporting.localization.Localization(__file__, 206, 33), cxx_compiler_53989, *[], **kwargs_53990)
        
        # Getting the type of 'self' (line 206)
        self_53992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'self')
        # Setting the type of the member '_cxx_compiler' of a type (line 206)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 12), self_53992, '_cxx_compiler', cxx_compiler_call_result_53991)
        # SSA branch for the else part of an if statement (line 197)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 208):
        
        # Assigning a Name to a Attribute (line 208):
        # Getting the type of 'None' (line 208)
        None_53993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 33), 'None')
        # Getting the type of 'self' (line 208)
        self_53994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'self')
        # Setting the type of the member '_cxx_compiler' of a type (line 208)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 12), self_53994, '_cxx_compiler', None_53993)
        # SSA join for if statement (line 197)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'need_f77_compiler' (line 211)
        need_f77_compiler_53995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 11), 'need_f77_compiler')
        # Testing the type of an if condition (line 211)
        if_condition_53996 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 211, 8), need_f77_compiler_53995)
        # Assigning a type to the variable 'if_condition_53996' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'if_condition_53996', if_condition_53996)
        # SSA begins for if statement (line 211)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 212):
        
        # Assigning a Attribute to a Name (line 212):
        # Getting the type of 'self' (line 212)
        self_53997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 20), 'self')
        # Obtaining the member 'fcompiler' of a type (line 212)
        fcompiler_53998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 20), self_53997, 'fcompiler')
        # Assigning a type to the variable 'ctype' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'ctype', fcompiler_53998)
        
        # Assigning a Call to a Attribute (line 213):
        
        # Assigning a Call to a Attribute (line 213):
        
        # Call to new_fcompiler(...): (line 213)
        # Processing the call keyword arguments (line 213)
        # Getting the type of 'self' (line 213)
        self_54000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 56), 'self', False)
        # Obtaining the member 'fcompiler' of a type (line 213)
        fcompiler_54001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 56), self_54000, 'fcompiler')
        keyword_54002 = fcompiler_54001
        # Getting the type of 'self' (line 214)
        self_54003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 55), 'self', False)
        # Obtaining the member 'verbose' of a type (line 214)
        verbose_54004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 55), self_54003, 'verbose')
        keyword_54005 = verbose_54004
        # Getting the type of 'self' (line 215)
        self_54006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 55), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 215)
        dry_run_54007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 55), self_54006, 'dry_run')
        keyword_54008 = dry_run_54007
        # Getting the type of 'self' (line 216)
        self_54009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 53), 'self', False)
        # Obtaining the member 'force' of a type (line 216)
        force_54010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 53), self_54009, 'force')
        keyword_54011 = force_54010
        # Getting the type of 'False' (line 217)
        False_54012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 58), 'False', False)
        keyword_54013 = False_54012
        # Getting the type of 'self' (line 218)
        self_54014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 58), 'self', False)
        # Obtaining the member 'compiler' of a type (line 218)
        compiler_54015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 58), self_54014, 'compiler')
        keyword_54016 = compiler_54015
        kwargs_54017 = {'force': keyword_54011, 'verbose': keyword_54005, 'dry_run': keyword_54008, 'c_compiler': keyword_54016, 'requiref90': keyword_54013, 'compiler': keyword_54002}
        # Getting the type of 'new_fcompiler' (line 213)
        new_fcompiler_53999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 33), 'new_fcompiler', False)
        # Calling new_fcompiler(args, kwargs) (line 213)
        new_fcompiler_call_result_54018 = invoke(stypy.reporting.localization.Localization(__file__, 213, 33), new_fcompiler_53999, *[], **kwargs_54017)
        
        # Getting the type of 'self' (line 213)
        self_54019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'self')
        # Setting the type of the member '_f77_compiler' of a type (line 213)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 12), self_54019, '_f77_compiler', new_fcompiler_call_result_54018)
        
        # Assigning a Attribute to a Name (line 219):
        
        # Assigning a Attribute to a Name (line 219):
        # Getting the type of 'self' (line 219)
        self_54020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 24), 'self')
        # Obtaining the member '_f77_compiler' of a type (line 219)
        _f77_compiler_54021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 24), self_54020, '_f77_compiler')
        # Assigning a type to the variable 'fcompiler' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'fcompiler', _f77_compiler_54021)
        
        # Getting the type of 'fcompiler' (line 220)
        fcompiler_54022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 15), 'fcompiler')
        # Testing the type of an if condition (line 220)
        if_condition_54023 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 220, 12), fcompiler_54022)
        # Assigning a type to the variable 'if_condition_54023' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'if_condition_54023', if_condition_54023)
        # SSA begins for if statement (line 220)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 221):
        
        # Assigning a Attribute to a Name (line 221):
        # Getting the type of 'fcompiler' (line 221)
        fcompiler_54024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 24), 'fcompiler')
        # Obtaining the member 'compiler_type' of a type (line 221)
        compiler_type_54025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 24), fcompiler_54024, 'compiler_type')
        # Assigning a type to the variable 'ctype' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'ctype', compiler_type_54025)
        
        # Call to customize(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'self' (line 222)
        self_54028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 36), 'self', False)
        # Obtaining the member 'distribution' of a type (line 222)
        distribution_54029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 36), self_54028, 'distribution')
        # Processing the call keyword arguments (line 222)
        kwargs_54030 = {}
        # Getting the type of 'fcompiler' (line 222)
        fcompiler_54026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'fcompiler', False)
        # Obtaining the member 'customize' of a type (line 222)
        customize_54027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 16), fcompiler_54026, 'customize')
        # Calling customize(args, kwargs) (line 222)
        customize_call_result_54031 = invoke(stypy.reporting.localization.Localization(__file__, 222, 16), customize_54027, *[distribution_54029], **kwargs_54030)
        
        # SSA join for if statement (line 220)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'fcompiler' (line 223)
        fcompiler_54032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 15), 'fcompiler')
        
        # Call to get_version(...): (line 223)
        # Processing the call keyword arguments (line 223)
        kwargs_54035 = {}
        # Getting the type of 'fcompiler' (line 223)
        fcompiler_54033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 29), 'fcompiler', False)
        # Obtaining the member 'get_version' of a type (line 223)
        get_version_54034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 29), fcompiler_54033, 'get_version')
        # Calling get_version(args, kwargs) (line 223)
        get_version_call_result_54036 = invoke(stypy.reporting.localization.Localization(__file__, 223, 29), get_version_54034, *[], **kwargs_54035)
        
        # Applying the binary operator 'and' (line 223)
        result_and_keyword_54037 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 15), 'and', fcompiler_54032, get_version_call_result_54036)
        
        # Testing the type of an if condition (line 223)
        if_condition_54038 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 12), result_and_keyword_54037)
        # Assigning a type to the variable 'if_condition_54038' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'if_condition_54038', if_condition_54038)
        # SSA begins for if statement (line 223)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to customize_cmd(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'self' (line 224)
        self_54041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 40), 'self', False)
        # Processing the call keyword arguments (line 224)
        kwargs_54042 = {}
        # Getting the type of 'fcompiler' (line 224)
        fcompiler_54039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'fcompiler', False)
        # Obtaining the member 'customize_cmd' of a type (line 224)
        customize_cmd_54040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 16), fcompiler_54039, 'customize_cmd')
        # Calling customize_cmd(args, kwargs) (line 224)
        customize_cmd_call_result_54043 = invoke(stypy.reporting.localization.Localization(__file__, 224, 16), customize_cmd_54040, *[self_54041], **kwargs_54042)
        
        
        # Call to show_customization(...): (line 225)
        # Processing the call keyword arguments (line 225)
        kwargs_54046 = {}
        # Getting the type of 'fcompiler' (line 225)
        fcompiler_54044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 16), 'fcompiler', False)
        # Obtaining the member 'show_customization' of a type (line 225)
        show_customization_54045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 16), fcompiler_54044, 'show_customization')
        # Calling show_customization(args, kwargs) (line 225)
        show_customization_call_result_54047 = invoke(stypy.reporting.localization.Localization(__file__, 225, 16), show_customization_54045, *[], **kwargs_54046)
        
        # SSA branch for the else part of an if statement (line 223)
        module_type_store.open_ssa_branch('else')
        
        # Call to warn(...): (line 227)
        # Processing the call arguments (line 227)
        str_54050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 26), 'str', 'f77_compiler=%s is not available.')
        # Getting the type of 'ctype' (line 228)
        ctype_54051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 27), 'ctype', False)
        # Applying the binary operator '%' (line 227)
        result_mod_54052 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 26), '%', str_54050, ctype_54051)
        
        # Processing the call keyword arguments (line 227)
        kwargs_54053 = {}
        # Getting the type of 'self' (line 227)
        self_54048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), 'self', False)
        # Obtaining the member 'warn' of a type (line 227)
        warn_54049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 16), self_54048, 'warn')
        # Calling warn(args, kwargs) (line 227)
        warn_call_result_54054 = invoke(stypy.reporting.localization.Localization(__file__, 227, 16), warn_54049, *[result_mod_54052], **kwargs_54053)
        
        
        # Assigning a Name to a Attribute (line 229):
        
        # Assigning a Name to a Attribute (line 229):
        # Getting the type of 'None' (line 229)
        None_54055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 37), 'None')
        # Getting the type of 'self' (line 229)
        self_54056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 16), 'self')
        # Setting the type of the member '_f77_compiler' of a type (line 229)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 16), self_54056, '_f77_compiler', None_54055)
        # SSA join for if statement (line 223)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 211)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 231):
        
        # Assigning a Name to a Attribute (line 231):
        # Getting the type of 'None' (line 231)
        None_54057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 33), 'None')
        # Getting the type of 'self' (line 231)
        self_54058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'self')
        # Setting the type of the member '_f77_compiler' of a type (line 231)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 12), self_54058, '_f77_compiler', None_54057)
        # SSA join for if statement (line 211)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'need_f90_compiler' (line 234)
        need_f90_compiler_54059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 11), 'need_f90_compiler')
        # Testing the type of an if condition (line 234)
        if_condition_54060 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 234, 8), need_f90_compiler_54059)
        # Assigning a type to the variable 'if_condition_54060' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'if_condition_54060', if_condition_54060)
        # SSA begins for if statement (line 234)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 235):
        
        # Assigning a Attribute to a Name (line 235):
        # Getting the type of 'self' (line 235)
        self_54061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'self')
        # Obtaining the member 'fcompiler' of a type (line 235)
        fcompiler_54062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 20), self_54061, 'fcompiler')
        # Assigning a type to the variable 'ctype' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'ctype', fcompiler_54062)
        
        # Assigning a Call to a Attribute (line 236):
        
        # Assigning a Call to a Attribute (line 236):
        
        # Call to new_fcompiler(...): (line 236)
        # Processing the call keyword arguments (line 236)
        # Getting the type of 'self' (line 236)
        self_54064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 56), 'self', False)
        # Obtaining the member 'fcompiler' of a type (line 236)
        fcompiler_54065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 56), self_54064, 'fcompiler')
        keyword_54066 = fcompiler_54065
        # Getting the type of 'self' (line 237)
        self_54067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 55), 'self', False)
        # Obtaining the member 'verbose' of a type (line 237)
        verbose_54068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 55), self_54067, 'verbose')
        keyword_54069 = verbose_54068
        # Getting the type of 'self' (line 238)
        self_54070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 55), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 238)
        dry_run_54071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 55), self_54070, 'dry_run')
        keyword_54072 = dry_run_54071
        # Getting the type of 'self' (line 239)
        self_54073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 53), 'self', False)
        # Obtaining the member 'force' of a type (line 239)
        force_54074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 53), self_54073, 'force')
        keyword_54075 = force_54074
        # Getting the type of 'True' (line 240)
        True_54076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 58), 'True', False)
        keyword_54077 = True_54076
        # Getting the type of 'self' (line 241)
        self_54078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 60), 'self', False)
        # Obtaining the member 'compiler' of a type (line 241)
        compiler_54079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 60), self_54078, 'compiler')
        keyword_54080 = compiler_54079
        kwargs_54081 = {'force': keyword_54075, 'verbose': keyword_54069, 'dry_run': keyword_54072, 'c_compiler': keyword_54080, 'requiref90': keyword_54077, 'compiler': keyword_54066}
        # Getting the type of 'new_fcompiler' (line 236)
        new_fcompiler_54063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 33), 'new_fcompiler', False)
        # Calling new_fcompiler(args, kwargs) (line 236)
        new_fcompiler_call_result_54082 = invoke(stypy.reporting.localization.Localization(__file__, 236, 33), new_fcompiler_54063, *[], **kwargs_54081)
        
        # Getting the type of 'self' (line 236)
        self_54083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'self')
        # Setting the type of the member '_f90_compiler' of a type (line 236)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 12), self_54083, '_f90_compiler', new_fcompiler_call_result_54082)
        
        # Assigning a Attribute to a Name (line 242):
        
        # Assigning a Attribute to a Name (line 242):
        # Getting the type of 'self' (line 242)
        self_54084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 24), 'self')
        # Obtaining the member '_f90_compiler' of a type (line 242)
        _f90_compiler_54085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 24), self_54084, '_f90_compiler')
        # Assigning a type to the variable 'fcompiler' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'fcompiler', _f90_compiler_54085)
        
        # Getting the type of 'fcompiler' (line 243)
        fcompiler_54086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 15), 'fcompiler')
        # Testing the type of an if condition (line 243)
        if_condition_54087 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 243, 12), fcompiler_54086)
        # Assigning a type to the variable 'if_condition_54087' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'if_condition_54087', if_condition_54087)
        # SSA begins for if statement (line 243)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 244):
        
        # Assigning a Attribute to a Name (line 244):
        # Getting the type of 'fcompiler' (line 244)
        fcompiler_54088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'fcompiler')
        # Obtaining the member 'compiler_type' of a type (line 244)
        compiler_type_54089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 24), fcompiler_54088, 'compiler_type')
        # Assigning a type to the variable 'ctype' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 16), 'ctype', compiler_type_54089)
        
        # Call to customize(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 'self' (line 245)
        self_54092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 36), 'self', False)
        # Obtaining the member 'distribution' of a type (line 245)
        distribution_54093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 36), self_54092, 'distribution')
        # Processing the call keyword arguments (line 245)
        kwargs_54094 = {}
        # Getting the type of 'fcompiler' (line 245)
        fcompiler_54090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 16), 'fcompiler', False)
        # Obtaining the member 'customize' of a type (line 245)
        customize_54091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 16), fcompiler_54090, 'customize')
        # Calling customize(args, kwargs) (line 245)
        customize_call_result_54095 = invoke(stypy.reporting.localization.Localization(__file__, 245, 16), customize_54091, *[distribution_54093], **kwargs_54094)
        
        # SSA join for if statement (line 243)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'fcompiler' (line 246)
        fcompiler_54096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 15), 'fcompiler')
        
        # Call to get_version(...): (line 246)
        # Processing the call keyword arguments (line 246)
        kwargs_54099 = {}
        # Getting the type of 'fcompiler' (line 246)
        fcompiler_54097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 29), 'fcompiler', False)
        # Obtaining the member 'get_version' of a type (line 246)
        get_version_54098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 29), fcompiler_54097, 'get_version')
        # Calling get_version(args, kwargs) (line 246)
        get_version_call_result_54100 = invoke(stypy.reporting.localization.Localization(__file__, 246, 29), get_version_54098, *[], **kwargs_54099)
        
        # Applying the binary operator 'and' (line 246)
        result_and_keyword_54101 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 15), 'and', fcompiler_54096, get_version_call_result_54100)
        
        # Testing the type of an if condition (line 246)
        if_condition_54102 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 246, 12), result_and_keyword_54101)
        # Assigning a type to the variable 'if_condition_54102' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'if_condition_54102', if_condition_54102)
        # SSA begins for if statement (line 246)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to customize_cmd(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'self' (line 247)
        self_54105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 40), 'self', False)
        # Processing the call keyword arguments (line 247)
        kwargs_54106 = {}
        # Getting the type of 'fcompiler' (line 247)
        fcompiler_54103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 16), 'fcompiler', False)
        # Obtaining the member 'customize_cmd' of a type (line 247)
        customize_cmd_54104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 16), fcompiler_54103, 'customize_cmd')
        # Calling customize_cmd(args, kwargs) (line 247)
        customize_cmd_call_result_54107 = invoke(stypy.reporting.localization.Localization(__file__, 247, 16), customize_cmd_54104, *[self_54105], **kwargs_54106)
        
        
        # Call to show_customization(...): (line 248)
        # Processing the call keyword arguments (line 248)
        kwargs_54110 = {}
        # Getting the type of 'fcompiler' (line 248)
        fcompiler_54108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'fcompiler', False)
        # Obtaining the member 'show_customization' of a type (line 248)
        show_customization_54109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 16), fcompiler_54108, 'show_customization')
        # Calling show_customization(args, kwargs) (line 248)
        show_customization_call_result_54111 = invoke(stypy.reporting.localization.Localization(__file__, 248, 16), show_customization_54109, *[], **kwargs_54110)
        
        # SSA branch for the else part of an if statement (line 246)
        module_type_store.open_ssa_branch('else')
        
        # Call to warn(...): (line 250)
        # Processing the call arguments (line 250)
        str_54114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 26), 'str', 'f90_compiler=%s is not available.')
        # Getting the type of 'ctype' (line 251)
        ctype_54115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 27), 'ctype', False)
        # Applying the binary operator '%' (line 250)
        result_mod_54116 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 26), '%', str_54114, ctype_54115)
        
        # Processing the call keyword arguments (line 250)
        kwargs_54117 = {}
        # Getting the type of 'self' (line 250)
        self_54112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'self', False)
        # Obtaining the member 'warn' of a type (line 250)
        warn_54113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 16), self_54112, 'warn')
        # Calling warn(args, kwargs) (line 250)
        warn_call_result_54118 = invoke(stypy.reporting.localization.Localization(__file__, 250, 16), warn_54113, *[result_mod_54116], **kwargs_54117)
        
        
        # Assigning a Name to a Attribute (line 252):
        
        # Assigning a Name to a Attribute (line 252):
        # Getting the type of 'None' (line 252)
        None_54119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 37), 'None')
        # Getting the type of 'self' (line 252)
        self_54120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 16), 'self')
        # Setting the type of the member '_f90_compiler' of a type (line 252)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 16), self_54120, '_f90_compiler', None_54119)
        # SSA join for if statement (line 246)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 234)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 254):
        
        # Assigning a Name to a Attribute (line 254):
        # Getting the type of 'None' (line 254)
        None_54121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 33), 'None')
        # Getting the type of 'self' (line 254)
        self_54122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'self')
        # Setting the type of the member '_f90_compiler' of a type (line 254)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 12), self_54122, '_f90_compiler', None_54121)
        # SSA join for if statement (line 234)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to build_extensions(...): (line 257)
        # Processing the call keyword arguments (line 257)
        kwargs_54125 = {}
        # Getting the type of 'self' (line 257)
        self_54123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'self', False)
        # Obtaining the member 'build_extensions' of a type (line 257)
        build_extensions_54124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 8), self_54123, 'build_extensions')
        # Calling build_extensions(args, kwargs) (line 257)
        build_extensions_call_result_54126 = invoke(stypy.reporting.localization.Localization(__file__, 257, 8), build_extensions_54124, *[], **kwargs_54125)
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 77)
        stypy_return_type_54127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_54127)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_54127


    @norecursion
    def swig_sources(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'swig_sources'
        module_type_store = module_type_store.open_function_context('swig_sources', 260, 4, False)
        # Assigning a type to the variable 'self' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext.swig_sources.__dict__.__setitem__('stypy_localization', localization)
        build_ext.swig_sources.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext.swig_sources.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext.swig_sources.__dict__.__setitem__('stypy_function_name', 'build_ext.swig_sources')
        build_ext.swig_sources.__dict__.__setitem__('stypy_param_names_list', ['sources'])
        build_ext.swig_sources.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext.swig_sources.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext.swig_sources.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext.swig_sources.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext.swig_sources.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext.swig_sources.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.swig_sources', ['sources'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'swig_sources', localization, ['sources'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'swig_sources(...)' code ##################

        # Getting the type of 'sources' (line 262)
        sources_54128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 15), 'sources')
        # Assigning a type to the variable 'stypy_return_type' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'stypy_return_type', sources_54128)
        
        # ################# End of 'swig_sources(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'swig_sources' in the type store
        # Getting the type of 'stypy_return_type' (line 260)
        stypy_return_type_54129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_54129)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'swig_sources'
        return stypy_return_type_54129


    @norecursion
    def build_extension(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'build_extension'
        module_type_store = module_type_store.open_function_context('build_extension', 264, 4, False)
        # Assigning a type to the variable 'self' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext.build_extension.__dict__.__setitem__('stypy_localization', localization)
        build_ext.build_extension.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext.build_extension.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext.build_extension.__dict__.__setitem__('stypy_function_name', 'build_ext.build_extension')
        build_ext.build_extension.__dict__.__setitem__('stypy_param_names_list', ['ext'])
        build_ext.build_extension.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext.build_extension.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext.build_extension.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext.build_extension.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext.build_extension.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext.build_extension.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.build_extension', ['ext'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'build_extension', localization, ['ext'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'build_extension(...)' code ##################

        
        # Assigning a Attribute to a Name (line 265):
        
        # Assigning a Attribute to a Name (line 265):
        # Getting the type of 'ext' (line 265)
        ext_54130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 18), 'ext')
        # Obtaining the member 'sources' of a type (line 265)
        sources_54131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 18), ext_54130, 'sources')
        # Assigning a type to the variable 'sources' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'sources', sources_54131)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'sources' (line 266)
        sources_54132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 11), 'sources')
        # Getting the type of 'None' (line 266)
        None_54133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 22), 'None')
        # Applying the binary operator 'is' (line 266)
        result_is__54134 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 11), 'is', sources_54132, None_54133)
        
        
        
        # Call to is_sequence(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'sources' (line 266)
        sources_54136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 46), 'sources', False)
        # Processing the call keyword arguments (line 266)
        kwargs_54137 = {}
        # Getting the type of 'is_sequence' (line 266)
        is_sequence_54135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 34), 'is_sequence', False)
        # Calling is_sequence(args, kwargs) (line 266)
        is_sequence_call_result_54138 = invoke(stypy.reporting.localization.Localization(__file__, 266, 34), is_sequence_54135, *[sources_54136], **kwargs_54137)
        
        # Applying the 'not' unary operator (line 266)
        result_not__54139 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 30), 'not', is_sequence_call_result_54138)
        
        # Applying the binary operator 'or' (line 266)
        result_or_keyword_54140 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 11), 'or', result_is__54134, result_not__54139)
        
        # Testing the type of an if condition (line 266)
        if_condition_54141 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 266, 8), result_or_keyword_54140)
        # Assigning a type to the variable 'if_condition_54141' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'if_condition_54141', if_condition_54141)
        # SSA begins for if statement (line 266)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to DistutilsSetupError(...): (line 267)
        # Processing the call arguments (line 267)
        str_54143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 17), 'str', "in 'ext_modules' option (extension '%s'), ")
        str_54144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 17), 'str', "'sources' must be present and must be ")
        # Applying the binary operator '+' (line 268)
        result_add_54145 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 17), '+', str_54143, str_54144)
        
        str_54146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 17), 'str', 'a list of source filenames')
        # Applying the binary operator '+' (line 269)
        result_add_54147 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 58), '+', result_add_54145, str_54146)
        
        # Getting the type of 'ext' (line 270)
        ext_54148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 49), 'ext', False)
        # Obtaining the member 'name' of a type (line 270)
        name_54149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 49), ext_54148, 'name')
        # Applying the binary operator '%' (line 268)
        result_mod_54150 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 16), '%', result_add_54147, name_54149)
        
        # Processing the call keyword arguments (line 267)
        kwargs_54151 = {}
        # Getting the type of 'DistutilsSetupError' (line 267)
        DistutilsSetupError_54142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 18), 'DistutilsSetupError', False)
        # Calling DistutilsSetupError(args, kwargs) (line 267)
        DistutilsSetupError_call_result_54152 = invoke(stypy.reporting.localization.Localization(__file__, 267, 18), DistutilsSetupError_54142, *[result_mod_54150], **kwargs_54151)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 267, 12), DistutilsSetupError_call_result_54152, 'raise parameter', BaseException)
        # SSA join for if statement (line 266)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 271):
        
        # Assigning a Call to a Name (line 271):
        
        # Call to list(...): (line 271)
        # Processing the call arguments (line 271)
        # Getting the type of 'sources' (line 271)
        sources_54154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 23), 'sources', False)
        # Processing the call keyword arguments (line 271)
        kwargs_54155 = {}
        # Getting the type of 'list' (line 271)
        list_54153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 18), 'list', False)
        # Calling list(args, kwargs) (line 271)
        list_call_result_54156 = invoke(stypy.reporting.localization.Localization(__file__, 271, 18), list_54153, *[sources_54154], **kwargs_54155)
        
        # Assigning a type to the variable 'sources' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'sources', list_call_result_54156)
        
        
        # Getting the type of 'sources' (line 273)
        sources_54157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 15), 'sources')
        # Applying the 'not' unary operator (line 273)
        result_not__54158 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 11), 'not', sources_54157)
        
        # Testing the type of an if condition (line 273)
        if_condition_54159 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 8), result_not__54158)
        # Assigning a type to the variable 'if_condition_54159' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'if_condition_54159', if_condition_54159)
        # SSA begins for if statement (line 273)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 273)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 276):
        
        # Assigning a Call to a Name (line 276):
        
        # Call to get_ext_fullname(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'ext' (line 276)
        ext_54162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 41), 'ext', False)
        # Obtaining the member 'name' of a type (line 276)
        name_54163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 41), ext_54162, 'name')
        # Processing the call keyword arguments (line 276)
        kwargs_54164 = {}
        # Getting the type of 'self' (line 276)
        self_54160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 19), 'self', False)
        # Obtaining the member 'get_ext_fullname' of a type (line 276)
        get_ext_fullname_54161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 19), self_54160, 'get_ext_fullname')
        # Calling get_ext_fullname(args, kwargs) (line 276)
        get_ext_fullname_call_result_54165 = invoke(stypy.reporting.localization.Localization(__file__, 276, 19), get_ext_fullname_54161, *[name_54163], **kwargs_54164)
        
        # Assigning a type to the variable 'fullname' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'fullname', get_ext_fullname_call_result_54165)
        
        # Getting the type of 'self' (line 277)
        self_54166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 11), 'self')
        # Obtaining the member 'inplace' of a type (line 277)
        inplace_54167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 11), self_54166, 'inplace')
        # Testing the type of an if condition (line 277)
        if_condition_54168 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 277, 8), inplace_54167)
        # Assigning a type to the variable 'if_condition_54168' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'if_condition_54168', if_condition_54168)
        # SSA begins for if statement (line 277)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 278):
        
        # Assigning a Call to a Name (line 278):
        
        # Call to split(...): (line 278)
        # Processing the call arguments (line 278)
        str_54171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 37), 'str', '.')
        # Processing the call keyword arguments (line 278)
        kwargs_54172 = {}
        # Getting the type of 'fullname' (line 278)
        fullname_54169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 22), 'fullname', False)
        # Obtaining the member 'split' of a type (line 278)
        split_54170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 22), fullname_54169, 'split')
        # Calling split(args, kwargs) (line 278)
        split_call_result_54173 = invoke(stypy.reporting.localization.Localization(__file__, 278, 22), split_54170, *[str_54171], **kwargs_54172)
        
        # Assigning a type to the variable 'modpath' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'modpath', split_call_result_54173)
        
        # Assigning a Call to a Name (line 279):
        
        # Assigning a Call to a Name (line 279):
        
        # Call to join(...): (line 279)
        # Processing the call arguments (line 279)
        
        # Obtaining the type of the subscript
        int_54176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 39), 'int')
        int_54177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 41), 'int')
        slice_54178 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 279, 31), int_54176, int_54177, None)
        # Getting the type of 'modpath' (line 279)
        modpath_54179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 31), 'modpath', False)
        # Obtaining the member '__getitem__' of a type (line 279)
        getitem___54180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 31), modpath_54179, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 279)
        subscript_call_result_54181 = invoke(stypy.reporting.localization.Localization(__file__, 279, 31), getitem___54180, slice_54178)
        
        # Processing the call keyword arguments (line 279)
        kwargs_54182 = {}
        str_54174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 22), 'str', '.')
        # Obtaining the member 'join' of a type (line 279)
        join_54175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 22), str_54174, 'join')
        # Calling join(args, kwargs) (line 279)
        join_call_result_54183 = invoke(stypy.reporting.localization.Localization(__file__, 279, 22), join_54175, *[subscript_call_result_54181], **kwargs_54182)
        
        # Assigning a type to the variable 'package' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'package', join_call_result_54183)
        
        # Assigning a Subscript to a Name (line 280):
        
        # Assigning a Subscript to a Name (line 280):
        
        # Obtaining the type of the subscript
        int_54184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 27), 'int')
        # Getting the type of 'modpath' (line 280)
        modpath_54185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 19), 'modpath')
        # Obtaining the member '__getitem__' of a type (line 280)
        getitem___54186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 19), modpath_54185, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 280)
        subscript_call_result_54187 = invoke(stypy.reporting.localization.Localization(__file__, 280, 19), getitem___54186, int_54184)
        
        # Assigning a type to the variable 'base' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'base', subscript_call_result_54187)
        
        # Assigning a Call to a Name (line 281):
        
        # Assigning a Call to a Name (line 281):
        
        # Call to get_finalized_command(...): (line 281)
        # Processing the call arguments (line 281)
        str_54190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 50), 'str', 'build_py')
        # Processing the call keyword arguments (line 281)
        kwargs_54191 = {}
        # Getting the type of 'self' (line 281)
        self_54188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 23), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 281)
        get_finalized_command_54189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 23), self_54188, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 281)
        get_finalized_command_call_result_54192 = invoke(stypy.reporting.localization.Localization(__file__, 281, 23), get_finalized_command_54189, *[str_54190], **kwargs_54191)
        
        # Assigning a type to the variable 'build_py' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'build_py', get_finalized_command_call_result_54192)
        
        # Assigning a Call to a Name (line 282):
        
        # Assigning a Call to a Name (line 282):
        
        # Call to get_package_dir(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'package' (line 282)
        package_54195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 51), 'package', False)
        # Processing the call keyword arguments (line 282)
        kwargs_54196 = {}
        # Getting the type of 'build_py' (line 282)
        build_py_54193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 26), 'build_py', False)
        # Obtaining the member 'get_package_dir' of a type (line 282)
        get_package_dir_54194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 26), build_py_54193, 'get_package_dir')
        # Calling get_package_dir(args, kwargs) (line 282)
        get_package_dir_call_result_54197 = invoke(stypy.reporting.localization.Localization(__file__, 282, 26), get_package_dir_54194, *[package_54195], **kwargs_54196)
        
        # Assigning a type to the variable 'package_dir' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'package_dir', get_package_dir_call_result_54197)
        
        # Assigning a Call to a Name (line 283):
        
        # Assigning a Call to a Name (line 283):
        
        # Call to join(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 'package_dir' (line 283)
        package_dir_54201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 40), 'package_dir', False)
        
        # Call to get_ext_filename(...): (line 284)
        # Processing the call arguments (line 284)
        # Getting the type of 'base' (line 284)
        base_54204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 62), 'base', False)
        # Processing the call keyword arguments (line 284)
        kwargs_54205 = {}
        # Getting the type of 'self' (line 284)
        self_54202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 40), 'self', False)
        # Obtaining the member 'get_ext_filename' of a type (line 284)
        get_ext_filename_54203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 40), self_54202, 'get_ext_filename')
        # Calling get_ext_filename(args, kwargs) (line 284)
        get_ext_filename_call_result_54206 = invoke(stypy.reporting.localization.Localization(__file__, 284, 40), get_ext_filename_54203, *[base_54204], **kwargs_54205)
        
        # Processing the call keyword arguments (line 283)
        kwargs_54207 = {}
        # Getting the type of 'os' (line 283)
        os_54198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 283)
        path_54199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 27), os_54198, 'path')
        # Obtaining the member 'join' of a type (line 283)
        join_54200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 27), path_54199, 'join')
        # Calling join(args, kwargs) (line 283)
        join_call_result_54208 = invoke(stypy.reporting.localization.Localization(__file__, 283, 27), join_54200, *[package_dir_54201, get_ext_filename_call_result_54206], **kwargs_54207)
        
        # Assigning a type to the variable 'ext_filename' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'ext_filename', join_call_result_54208)
        # SSA branch for the else part of an if statement (line 277)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 286):
        
        # Assigning a Call to a Name (line 286):
        
        # Call to join(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 'self' (line 286)
        self_54212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 40), 'self', False)
        # Obtaining the member 'build_lib' of a type (line 286)
        build_lib_54213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 40), self_54212, 'build_lib')
        
        # Call to get_ext_filename(...): (line 287)
        # Processing the call arguments (line 287)
        # Getting the type of 'fullname' (line 287)
        fullname_54216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 62), 'fullname', False)
        # Processing the call keyword arguments (line 287)
        kwargs_54217 = {}
        # Getting the type of 'self' (line 287)
        self_54214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 40), 'self', False)
        # Obtaining the member 'get_ext_filename' of a type (line 287)
        get_ext_filename_54215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 40), self_54214, 'get_ext_filename')
        # Calling get_ext_filename(args, kwargs) (line 287)
        get_ext_filename_call_result_54218 = invoke(stypy.reporting.localization.Localization(__file__, 287, 40), get_ext_filename_54215, *[fullname_54216], **kwargs_54217)
        
        # Processing the call keyword arguments (line 286)
        kwargs_54219 = {}
        # Getting the type of 'os' (line 286)
        os_54209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 286)
        path_54210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 27), os_54209, 'path')
        # Obtaining the member 'join' of a type (line 286)
        join_54211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 27), path_54210, 'join')
        # Calling join(args, kwargs) (line 286)
        join_call_result_54220 = invoke(stypy.reporting.localization.Localization(__file__, 286, 27), join_54211, *[build_lib_54213, get_ext_filename_call_result_54218], **kwargs_54219)
        
        # Assigning a type to the variable 'ext_filename' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'ext_filename', join_call_result_54220)
        # SSA join for if statement (line 277)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 288):
        
        # Assigning a BinOp to a Name (line 288):
        # Getting the type of 'sources' (line 288)
        sources_54221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 18), 'sources')
        # Getting the type of 'ext' (line 288)
        ext_54222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 28), 'ext')
        # Obtaining the member 'depends' of a type (line 288)
        depends_54223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 28), ext_54222, 'depends')
        # Applying the binary operator '+' (line 288)
        result_add_54224 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 18), '+', sources_54221, depends_54223)
        
        # Assigning a type to the variable 'depends' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'depends', result_add_54224)
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 290)
        self_54225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 16), 'self')
        # Obtaining the member 'force' of a type (line 290)
        force_54226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 16), self_54225, 'force')
        
        # Call to newer_group(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'depends' (line 290)
        depends_54228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 42), 'depends', False)
        # Getting the type of 'ext_filename' (line 290)
        ext_filename_54229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 51), 'ext_filename', False)
        str_54230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 65), 'str', 'newer')
        # Processing the call keyword arguments (line 290)
        kwargs_54231 = {}
        # Getting the type of 'newer_group' (line 290)
        newer_group_54227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 30), 'newer_group', False)
        # Calling newer_group(args, kwargs) (line 290)
        newer_group_call_result_54232 = invoke(stypy.reporting.localization.Localization(__file__, 290, 30), newer_group_54227, *[depends_54228, ext_filename_54229, str_54230], **kwargs_54231)
        
        # Applying the binary operator 'or' (line 290)
        result_or_keyword_54233 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 16), 'or', force_54226, newer_group_call_result_54232)
        
        # Applying the 'not' unary operator (line 290)
        result_not__54234 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 11), 'not', result_or_keyword_54233)
        
        # Testing the type of an if condition (line 290)
        if_condition_54235 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 290, 8), result_not__54234)
        # Assigning a type to the variable 'if_condition_54235' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'if_condition_54235', if_condition_54235)
        # SSA begins for if statement (line 290)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to debug(...): (line 291)
        # Processing the call arguments (line 291)
        str_54238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 22), 'str', "skipping '%s' extension (up-to-date)")
        # Getting the type of 'ext' (line 291)
        ext_54239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 62), 'ext', False)
        # Obtaining the member 'name' of a type (line 291)
        name_54240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 62), ext_54239, 'name')
        # Processing the call keyword arguments (line 291)
        kwargs_54241 = {}
        # Getting the type of 'log' (line 291)
        log_54236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'log', False)
        # Obtaining the member 'debug' of a type (line 291)
        debug_54237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 12), log_54236, 'debug')
        # Calling debug(args, kwargs) (line 291)
        debug_call_result_54242 = invoke(stypy.reporting.localization.Localization(__file__, 291, 12), debug_54237, *[str_54238, name_54240], **kwargs_54241)
        
        # Assigning a type to the variable 'stypy_return_type' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'stypy_return_type', types.NoneType)
        # SSA branch for the else part of an if statement (line 290)
        module_type_store.open_ssa_branch('else')
        
        # Call to info(...): (line 294)
        # Processing the call arguments (line 294)
        str_54245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 21), 'str', "building '%s' extension")
        # Getting the type of 'ext' (line 294)
        ext_54246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 48), 'ext', False)
        # Obtaining the member 'name' of a type (line 294)
        name_54247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 48), ext_54246, 'name')
        # Processing the call keyword arguments (line 294)
        kwargs_54248 = {}
        # Getting the type of 'log' (line 294)
        log_54243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'log', False)
        # Obtaining the member 'info' of a type (line 294)
        info_54244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 12), log_54243, 'info')
        # Calling info(args, kwargs) (line 294)
        info_call_result_54249 = invoke(stypy.reporting.localization.Localization(__file__, 294, 12), info_54244, *[str_54245, name_54247], **kwargs_54248)
        
        # SSA join for if statement (line 290)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BoolOp to a Name (line 296):
        
        # Assigning a BoolOp to a Name (line 296):
        
        # Evaluating a boolean operation
        # Getting the type of 'ext' (line 296)
        ext_54250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 21), 'ext')
        # Obtaining the member 'extra_compile_args' of a type (line 296)
        extra_compile_args_54251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 21), ext_54250, 'extra_compile_args')
        
        # Obtaining an instance of the builtin type 'list' (line 296)
        list_54252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 296)
        
        # Applying the binary operator 'or' (line 296)
        result_or_keyword_54253 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 21), 'or', extra_compile_args_54251, list_54252)
        
        # Assigning a type to the variable 'extra_args' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'extra_args', result_or_keyword_54253)
        
        # Assigning a Subscript to a Name (line 297):
        
        # Assigning a Subscript to a Name (line 297):
        
        # Obtaining the type of the subscript
        slice_54254 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 297, 17), None, None, None)
        # Getting the type of 'ext' (line 297)
        ext_54255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 17), 'ext')
        # Obtaining the member 'define_macros' of a type (line 297)
        define_macros_54256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 17), ext_54255, 'define_macros')
        # Obtaining the member '__getitem__' of a type (line 297)
        getitem___54257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 17), define_macros_54256, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 297)
        subscript_call_result_54258 = invoke(stypy.reporting.localization.Localization(__file__, 297, 17), getitem___54257, slice_54254)
        
        # Assigning a type to the variable 'macros' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'macros', subscript_call_result_54258)
        
        # Getting the type of 'ext' (line 298)
        ext_54259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 21), 'ext')
        # Obtaining the member 'undef_macros' of a type (line 298)
        undef_macros_54260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 21), ext_54259, 'undef_macros')
        # Testing the type of a for loop iterable (line 298)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 298, 8), undef_macros_54260)
        # Getting the type of the for loop variable (line 298)
        for_loop_var_54261 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 298, 8), undef_macros_54260)
        # Assigning a type to the variable 'undef' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'undef', for_loop_var_54261)
        # SSA begins for a for statement (line 298)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 299)
        # Processing the call arguments (line 299)
        
        # Obtaining an instance of the builtin type 'tuple' (line 299)
        tuple_54264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 299)
        # Adding element type (line 299)
        # Getting the type of 'undef' (line 299)
        undef_54265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 27), 'undef', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 27), tuple_54264, undef_54265)
        
        # Processing the call keyword arguments (line 299)
        kwargs_54266 = {}
        # Getting the type of 'macros' (line 299)
        macros_54262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'macros', False)
        # Obtaining the member 'append' of a type (line 299)
        append_54263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 12), macros_54262, 'append')
        # Calling append(args, kwargs) (line 299)
        append_call_result_54267 = invoke(stypy.reporting.localization.Localization(__file__, 299, 12), append_54263, *[tuple_54264], **kwargs_54266)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 301):
        
        # Assigning a Call to a Name:
        
        # Call to filter_sources(...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 'ext' (line 302)
        ext_54269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 34), 'ext', False)
        # Obtaining the member 'sources' of a type (line 302)
        sources_54270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 34), ext_54269, 'sources')
        # Processing the call keyword arguments (line 302)
        kwargs_54271 = {}
        # Getting the type of 'filter_sources' (line 302)
        filter_sources_54268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 19), 'filter_sources', False)
        # Calling filter_sources(args, kwargs) (line 302)
        filter_sources_call_result_54272 = invoke(stypy.reporting.localization.Localization(__file__, 302, 19), filter_sources_54268, *[sources_54270], **kwargs_54271)
        
        # Assigning a type to the variable 'call_assignment_53464' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'call_assignment_53464', filter_sources_call_result_54272)
        
        # Assigning a Call to a Name (line 301):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_54275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 8), 'int')
        # Processing the call keyword arguments
        kwargs_54276 = {}
        # Getting the type of 'call_assignment_53464' (line 301)
        call_assignment_53464_54273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'call_assignment_53464', False)
        # Obtaining the member '__getitem__' of a type (line 301)
        getitem___54274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 8), call_assignment_53464_54273, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_54277 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___54274, *[int_54275], **kwargs_54276)
        
        # Assigning a type to the variable 'call_assignment_53465' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'call_assignment_53465', getitem___call_result_54277)
        
        # Assigning a Name to a Name (line 301):
        # Getting the type of 'call_assignment_53465' (line 301)
        call_assignment_53465_54278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'call_assignment_53465')
        # Assigning a type to the variable 'c_sources' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'c_sources', call_assignment_53465_54278)
        
        # Assigning a Call to a Name (line 301):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_54281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 8), 'int')
        # Processing the call keyword arguments
        kwargs_54282 = {}
        # Getting the type of 'call_assignment_53464' (line 301)
        call_assignment_53464_54279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'call_assignment_53464', False)
        # Obtaining the member '__getitem__' of a type (line 301)
        getitem___54280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 8), call_assignment_53464_54279, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_54283 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___54280, *[int_54281], **kwargs_54282)
        
        # Assigning a type to the variable 'call_assignment_53466' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'call_assignment_53466', getitem___call_result_54283)
        
        # Assigning a Name to a Name (line 301):
        # Getting the type of 'call_assignment_53466' (line 301)
        call_assignment_53466_54284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'call_assignment_53466')
        # Assigning a type to the variable 'cxx_sources' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 19), 'cxx_sources', call_assignment_53466_54284)
        
        # Assigning a Call to a Name (line 301):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_54287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 8), 'int')
        # Processing the call keyword arguments
        kwargs_54288 = {}
        # Getting the type of 'call_assignment_53464' (line 301)
        call_assignment_53464_54285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'call_assignment_53464', False)
        # Obtaining the member '__getitem__' of a type (line 301)
        getitem___54286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 8), call_assignment_53464_54285, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_54289 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___54286, *[int_54287], **kwargs_54288)
        
        # Assigning a type to the variable 'call_assignment_53467' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'call_assignment_53467', getitem___call_result_54289)
        
        # Assigning a Name to a Name (line 301):
        # Getting the type of 'call_assignment_53467' (line 301)
        call_assignment_53467_54290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'call_assignment_53467')
        # Assigning a type to the variable 'f_sources' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 32), 'f_sources', call_assignment_53467_54290)
        
        # Assigning a Call to a Name (line 301):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_54293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 8), 'int')
        # Processing the call keyword arguments
        kwargs_54294 = {}
        # Getting the type of 'call_assignment_53464' (line 301)
        call_assignment_53464_54291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'call_assignment_53464', False)
        # Obtaining the member '__getitem__' of a type (line 301)
        getitem___54292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 8), call_assignment_53464_54291, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_54295 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___54292, *[int_54293], **kwargs_54294)
        
        # Assigning a type to the variable 'call_assignment_53468' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'call_assignment_53468', getitem___call_result_54295)
        
        # Assigning a Name to a Name (line 301):
        # Getting the type of 'call_assignment_53468' (line 301)
        call_assignment_53468_54296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'call_assignment_53468')
        # Assigning a type to the variable 'fmodule_sources' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 43), 'fmodule_sources', call_assignment_53468_54296)
        
        
        # Getting the type of 'self' (line 306)
        self_54297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 11), 'self')
        # Obtaining the member 'compiler' of a type (line 306)
        compiler_54298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 11), self_54297, 'compiler')
        # Obtaining the member 'compiler_type' of a type (line 306)
        compiler_type_54299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 11), compiler_54298, 'compiler_type')
        str_54300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 40), 'str', 'msvc')
        # Applying the binary operator '==' (line 306)
        result_eq_54301 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 11), '==', compiler_type_54299, str_54300)
        
        # Testing the type of an if condition (line 306)
        if_condition_54302 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 306, 8), result_eq_54301)
        # Assigning a type to the variable 'if_condition_54302' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'if_condition_54302', if_condition_54302)
        # SSA begins for if statement (line 306)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'cxx_sources' (line 307)
        cxx_sources_54303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 15), 'cxx_sources')
        # Testing the type of an if condition (line 307)
        if_condition_54304 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 307, 12), cxx_sources_54303)
        # Assigning a type to the variable 'if_condition_54304' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'if_condition_54304', if_condition_54304)
        # SSA begins for if statement (line 307)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 309)
        # Processing the call arguments (line 309)
        str_54307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 34), 'str', '/Zm1000')
        # Processing the call keyword arguments (line 309)
        kwargs_54308 = {}
        # Getting the type of 'extra_args' (line 309)
        extra_args_54305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 16), 'extra_args', False)
        # Obtaining the member 'append' of a type (line 309)
        append_54306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 16), extra_args_54305, 'append')
        # Calling append(args, kwargs) (line 309)
        append_call_result_54309 = invoke(stypy.reporting.localization.Localization(__file__, 309, 16), append_54306, *[str_54307], **kwargs_54308)
        
        # SSA join for if statement (line 307)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'c_sources' (line 312)
        c_sources_54310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'c_sources')
        # Getting the type of 'cxx_sources' (line 312)
        cxx_sources_54311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 25), 'cxx_sources')
        # Applying the binary operator '+=' (line 312)
        result_iadd_54312 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 12), '+=', c_sources_54310, cxx_sources_54311)
        # Assigning a type to the variable 'c_sources' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'c_sources', result_iadd_54312)
        
        
        # Assigning a List to a Name (line 313):
        
        # Assigning a List to a Name (line 313):
        
        # Obtaining an instance of the builtin type 'list' (line 313)
        list_54313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 313)
        
        # Assigning a type to the variable 'cxx_sources' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'cxx_sources', list_54313)
        # SSA join for if statement (line 306)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'ext' (line 316)
        ext_54314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 11), 'ext')
        # Obtaining the member 'language' of a type (line 316)
        language_54315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 11), ext_54314, 'language')
        str_54316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 25), 'str', 'f90')
        # Applying the binary operator '==' (line 316)
        result_eq_54317 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 11), '==', language_54315, str_54316)
        
        # Testing the type of an if condition (line 316)
        if_condition_54318 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 316, 8), result_eq_54317)
        # Assigning a type to the variable 'if_condition_54318' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'if_condition_54318', if_condition_54318)
        # SSA begins for if statement (line 316)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 317):
        
        # Assigning a Attribute to a Name (line 317):
        # Getting the type of 'self' (line 317)
        self_54319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 24), 'self')
        # Obtaining the member '_f90_compiler' of a type (line 317)
        _f90_compiler_54320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 24), self_54319, '_f90_compiler')
        # Assigning a type to the variable 'fcompiler' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'fcompiler', _f90_compiler_54320)
        # SSA branch for the else part of an if statement (line 316)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'ext' (line 318)
        ext_54321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 13), 'ext')
        # Obtaining the member 'language' of a type (line 318)
        language_54322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 13), ext_54321, 'language')
        str_54323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 27), 'str', 'f77')
        # Applying the binary operator '==' (line 318)
        result_eq_54324 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 13), '==', language_54322, str_54323)
        
        # Testing the type of an if condition (line 318)
        if_condition_54325 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 318, 13), result_eq_54324)
        # Assigning a type to the variable 'if_condition_54325' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 13), 'if_condition_54325', if_condition_54325)
        # SSA begins for if statement (line 318)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 319):
        
        # Assigning a Attribute to a Name (line 319):
        # Getting the type of 'self' (line 319)
        self_54326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 24), 'self')
        # Obtaining the member '_f77_compiler' of a type (line 319)
        _f77_compiler_54327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 24), self_54326, '_f77_compiler')
        # Assigning a type to the variable 'fcompiler' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'fcompiler', _f77_compiler_54327)
        # SSA branch for the else part of an if statement (line 318)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BoolOp to a Name (line 321):
        
        # Assigning a BoolOp to a Name (line 321):
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 321)
        self_54328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 24), 'self')
        # Obtaining the member '_f90_compiler' of a type (line 321)
        _f90_compiler_54329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 24), self_54328, '_f90_compiler')
        # Getting the type of 'self' (line 321)
        self_54330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 46), 'self')
        # Obtaining the member '_f77_compiler' of a type (line 321)
        _f77_compiler_54331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 46), self_54330, '_f77_compiler')
        # Applying the binary operator 'or' (line 321)
        result_or_keyword_54332 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 24), 'or', _f90_compiler_54329, _f77_compiler_54331)
        
        # Assigning a type to the variable 'fcompiler' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'fcompiler', result_or_keyword_54332)
        # SSA join for if statement (line 318)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 316)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 322)
        # Getting the type of 'fcompiler' (line 322)
        fcompiler_54333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'fcompiler')
        # Getting the type of 'None' (line 322)
        None_54334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 28), 'None')
        
        (may_be_54335, more_types_in_union_54336) = may_not_be_none(fcompiler_54333, None_54334)

        if may_be_54335:

            if more_types_in_union_54336:
                # Runtime conditional SSA (line 322)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a IfExp to a Attribute (line 323):
            
            # Assigning a IfExp to a Attribute (line 323):
            
            
            # Call to hasattr(...): (line 323)
            # Processing the call arguments (line 323)
            # Getting the type of 'ext' (line 323)
            ext_54338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 93), 'ext', False)
            str_54339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 98), 'str', 'extra_f77_compile_args')
            # Processing the call keyword arguments (line 323)
            kwargs_54340 = {}
            # Getting the type of 'hasattr' (line 323)
            hasattr_54337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 85), 'hasattr', False)
            # Calling hasattr(args, kwargs) (line 323)
            hasattr_call_result_54341 = invoke(stypy.reporting.localization.Localization(__file__, 323, 85), hasattr_54337, *[ext_54338, str_54339], **kwargs_54340)
            
            # Testing the type of an if expression (line 323)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 323, 47), hasattr_call_result_54341)
            # SSA begins for if expression (line 323)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
            
            # Evaluating a boolean operation
            # Getting the type of 'ext' (line 323)
            ext_54342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 48), 'ext')
            # Obtaining the member 'extra_f77_compile_args' of a type (line 323)
            extra_f77_compile_args_54343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 48), ext_54342, 'extra_f77_compile_args')
            
            # Obtaining an instance of the builtin type 'list' (line 323)
            list_54344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 78), 'list')
            # Adding type elements to the builtin type 'list' instance (line 323)
            
            # Applying the binary operator 'or' (line 323)
            result_or_keyword_54345 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 48), 'or', extra_f77_compile_args_54343, list_54344)
            
            # SSA branch for the else part of an if expression (line 323)
            module_type_store.open_ssa_branch('if expression else')
            
            # Obtaining an instance of the builtin type 'list' (line 323)
            list_54346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 129), 'list')
            # Adding type elements to the builtin type 'list' instance (line 323)
            
            # SSA join for if expression (line 323)
            module_type_store = module_type_store.join_ssa_context()
            if_exp_54347 = union_type.UnionType.add(result_or_keyword_54345, list_54346)
            
            # Getting the type of 'fcompiler' (line 323)
            fcompiler_54348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'fcompiler')
            # Setting the type of the member 'extra_f77_compile_args' of a type (line 323)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 12), fcompiler_54348, 'extra_f77_compile_args', if_exp_54347)
            
            # Assigning a IfExp to a Attribute (line 324):
            
            # Assigning a IfExp to a Attribute (line 324):
            
            
            # Call to hasattr(...): (line 324)
            # Processing the call arguments (line 324)
            # Getting the type of 'ext' (line 324)
            ext_54350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 93), 'ext', False)
            str_54351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 98), 'str', 'extra_f90_compile_args')
            # Processing the call keyword arguments (line 324)
            kwargs_54352 = {}
            # Getting the type of 'hasattr' (line 324)
            hasattr_54349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 85), 'hasattr', False)
            # Calling hasattr(args, kwargs) (line 324)
            hasattr_call_result_54353 = invoke(stypy.reporting.localization.Localization(__file__, 324, 85), hasattr_54349, *[ext_54350, str_54351], **kwargs_54352)
            
            # Testing the type of an if expression (line 324)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 324, 47), hasattr_call_result_54353)
            # SSA begins for if expression (line 324)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
            
            # Evaluating a boolean operation
            # Getting the type of 'ext' (line 324)
            ext_54354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 48), 'ext')
            # Obtaining the member 'extra_f90_compile_args' of a type (line 324)
            extra_f90_compile_args_54355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 48), ext_54354, 'extra_f90_compile_args')
            
            # Obtaining an instance of the builtin type 'list' (line 324)
            list_54356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 78), 'list')
            # Adding type elements to the builtin type 'list' instance (line 324)
            
            # Applying the binary operator 'or' (line 324)
            result_or_keyword_54357 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 48), 'or', extra_f90_compile_args_54355, list_54356)
            
            # SSA branch for the else part of an if expression (line 324)
            module_type_store.open_ssa_branch('if expression else')
            
            # Obtaining an instance of the builtin type 'list' (line 324)
            list_54358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 129), 'list')
            # Adding type elements to the builtin type 'list' instance (line 324)
            
            # SSA join for if expression (line 324)
            module_type_store = module_type_store.join_ssa_context()
            if_exp_54359 = union_type.UnionType.add(result_or_keyword_54357, list_54358)
            
            # Getting the type of 'fcompiler' (line 324)
            fcompiler_54360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'fcompiler')
            # Setting the type of the member 'extra_f90_compile_args' of a type (line 324)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 12), fcompiler_54360, 'extra_f90_compile_args', if_exp_54359)

            if more_types_in_union_54336:
                # SSA join for if statement (line 322)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Name (line 325):
        
        # Assigning a Attribute to a Name (line 325):
        # Getting the type of 'self' (line 325)
        self_54361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 23), 'self')
        # Obtaining the member '_cxx_compiler' of a type (line 325)
        _cxx_compiler_54362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 23), self_54361, '_cxx_compiler')
        # Assigning a type to the variable 'cxx_compiler' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'cxx_compiler', _cxx_compiler_54362)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'cxx_sources' (line 328)
        cxx_sources_54363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 11), 'cxx_sources')
        
        # Getting the type of 'cxx_compiler' (line 328)
        cxx_compiler_54364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 27), 'cxx_compiler')
        # Getting the type of 'None' (line 328)
        None_54365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 43), 'None')
        # Applying the binary operator 'is' (line 328)
        result_is__54366 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 27), 'is', cxx_compiler_54364, None_54365)
        
        # Applying the binary operator 'and' (line 328)
        result_and_keyword_54367 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 11), 'and', cxx_sources_54363, result_is__54366)
        
        # Testing the type of an if condition (line 328)
        if_condition_54368 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 328, 8), result_and_keyword_54367)
        # Assigning a type to the variable 'if_condition_54368' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'if_condition_54368', if_condition_54368)
        # SSA begins for if statement (line 328)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to DistutilsError(...): (line 329)
        # Processing the call arguments (line 329)
        str_54370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 33), 'str', 'extension %r has C++ sourcesbut no C++ compiler found')
        # Getting the type of 'ext' (line 330)
        ext_54371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 49), 'ext', False)
        # Obtaining the member 'name' of a type (line 330)
        name_54372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 49), ext_54371, 'name')
        # Applying the binary operator '%' (line 329)
        result_mod_54373 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 33), '%', str_54370, name_54372)
        
        # Processing the call keyword arguments (line 329)
        kwargs_54374 = {}
        # Getting the type of 'DistutilsError' (line 329)
        DistutilsError_54369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 18), 'DistutilsError', False)
        # Calling DistutilsError(args, kwargs) (line 329)
        DistutilsError_call_result_54375 = invoke(stypy.reporting.localization.Localization(__file__, 329, 18), DistutilsError_54369, *[result_mod_54373], **kwargs_54374)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 329, 12), DistutilsError_call_result_54375, 'raise parameter', BaseException)
        # SSA join for if statement (line 328)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        # Getting the type of 'f_sources' (line 331)
        f_sources_54376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'f_sources')
        # Getting the type of 'fmodule_sources' (line 331)
        fmodule_sources_54377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 25), 'fmodule_sources')
        # Applying the binary operator 'or' (line 331)
        result_or_keyword_54378 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 12), 'or', f_sources_54376, fmodule_sources_54377)
        
        
        # Getting the type of 'fcompiler' (line 331)
        fcompiler_54379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 46), 'fcompiler')
        # Getting the type of 'None' (line 331)
        None_54380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 59), 'None')
        # Applying the binary operator 'is' (line 331)
        result_is__54381 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 46), 'is', fcompiler_54379, None_54380)
        
        # Applying the binary operator 'and' (line 331)
        result_and_keyword_54382 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 11), 'and', result_or_keyword_54378, result_is__54381)
        
        # Testing the type of an if condition (line 331)
        if_condition_54383 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 331, 8), result_and_keyword_54382)
        # Assigning a type to the variable 'if_condition_54383' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'if_condition_54383', if_condition_54383)
        # SSA begins for if statement (line 331)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to DistutilsError(...): (line 332)
        # Processing the call arguments (line 332)
        str_54385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 33), 'str', 'extension %r has Fortran sources but no Fortran compiler found')
        # Getting the type of 'ext' (line 333)
        ext_54386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 53), 'ext', False)
        # Obtaining the member 'name' of a type (line 333)
        name_54387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 53), ext_54386, 'name')
        # Applying the binary operator '%' (line 332)
        result_mod_54388 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 33), '%', str_54385, name_54387)
        
        # Processing the call keyword arguments (line 332)
        kwargs_54389 = {}
        # Getting the type of 'DistutilsError' (line 332)
        DistutilsError_54384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 18), 'DistutilsError', False)
        # Calling DistutilsError(args, kwargs) (line 332)
        DistutilsError_call_result_54390 = invoke(stypy.reporting.localization.Localization(__file__, 332, 18), DistutilsError_54384, *[result_mod_54388], **kwargs_54389)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 332, 12), DistutilsError_call_result_54390, 'raise parameter', BaseException)
        # SSA join for if statement (line 331)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'ext' (line 334)
        ext_54391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 11), 'ext')
        # Obtaining the member 'language' of a type (line 334)
        language_54392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 11), ext_54391, 'language')
        
        # Obtaining an instance of the builtin type 'list' (line 334)
        list_54393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 334)
        # Adding element type (line 334)
        str_54394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 28), 'str', 'f77')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 27), list_54393, str_54394)
        # Adding element type (line 334)
        str_54395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 35), 'str', 'f90')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 27), list_54393, str_54395)
        
        # Applying the binary operator 'in' (line 334)
        result_contains_54396 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 11), 'in', language_54392, list_54393)
        
        
        # Getting the type of 'fcompiler' (line 334)
        fcompiler_54397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 46), 'fcompiler')
        # Getting the type of 'None' (line 334)
        None_54398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 59), 'None')
        # Applying the binary operator 'is' (line 334)
        result_is__54399 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 46), 'is', fcompiler_54397, None_54398)
        
        # Applying the binary operator 'and' (line 334)
        result_and_keyword_54400 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 11), 'and', result_contains_54396, result_is__54399)
        
        # Testing the type of an if condition (line 334)
        if_condition_54401 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 8), result_and_keyword_54400)
        # Assigning a type to the variable 'if_condition_54401' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'if_condition_54401', if_condition_54401)
        # SSA begins for if statement (line 334)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 335)
        # Processing the call arguments (line 335)
        str_54404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 22), 'str', 'extension %r has Fortran libraries but no Fortran linker found, using default linker')
        # Getting the type of 'ext' (line 336)
        ext_54405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 73), 'ext', False)
        # Obtaining the member 'name' of a type (line 336)
        name_54406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 73), ext_54405, 'name')
        # Applying the binary operator '%' (line 335)
        result_mod_54407 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 22), '%', str_54404, name_54406)
        
        # Processing the call keyword arguments (line 335)
        kwargs_54408 = {}
        # Getting the type of 'self' (line 335)
        self_54402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'self', False)
        # Obtaining the member 'warn' of a type (line 335)
        warn_54403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 12), self_54402, 'warn')
        # Calling warn(args, kwargs) (line 335)
        warn_call_result_54409 = invoke(stypy.reporting.localization.Localization(__file__, 335, 12), warn_54403, *[result_mod_54407], **kwargs_54408)
        
        # SSA join for if statement (line 334)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'ext' (line 337)
        ext_54410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 11), 'ext')
        # Obtaining the member 'language' of a type (line 337)
        language_54411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 11), ext_54410, 'language')
        str_54412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 25), 'str', 'c++')
        # Applying the binary operator '==' (line 337)
        result_eq_54413 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 11), '==', language_54411, str_54412)
        
        
        # Getting the type of 'cxx_compiler' (line 337)
        cxx_compiler_54414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 35), 'cxx_compiler')
        # Getting the type of 'None' (line 337)
        None_54415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 51), 'None')
        # Applying the binary operator 'is' (line 337)
        result_is__54416 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 35), 'is', cxx_compiler_54414, None_54415)
        
        # Applying the binary operator 'and' (line 337)
        result_and_keyword_54417 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 11), 'and', result_eq_54413, result_is__54416)
        
        # Testing the type of an if condition (line 337)
        if_condition_54418 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 337, 8), result_and_keyword_54417)
        # Assigning a type to the variable 'if_condition_54418' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'if_condition_54418', if_condition_54418)
        # SSA begins for if statement (line 337)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 338)
        # Processing the call arguments (line 338)
        str_54421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 22), 'str', 'extension %r has C++ libraries but no C++ linker found, using default linker')
        # Getting the type of 'ext' (line 339)
        ext_54422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 69), 'ext', False)
        # Obtaining the member 'name' of a type (line 339)
        name_54423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 69), ext_54422, 'name')
        # Applying the binary operator '%' (line 338)
        result_mod_54424 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 22), '%', str_54421, name_54423)
        
        # Processing the call keyword arguments (line 338)
        kwargs_54425 = {}
        # Getting the type of 'self' (line 338)
        self_54419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'self', False)
        # Obtaining the member 'warn' of a type (line 338)
        warn_54420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 12), self_54419, 'warn')
        # Calling warn(args, kwargs) (line 338)
        warn_call_result_54426 = invoke(stypy.reporting.localization.Localization(__file__, 338, 12), warn_54420, *[result_mod_54424], **kwargs_54425)
        
        # SSA join for if statement (line 337)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Dict to a Name (line 341):
        
        # Assigning a Dict to a Name (line 341):
        
        # Obtaining an instance of the builtin type 'dict' (line 341)
        dict_54427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 14), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 341)
        # Adding element type (key, value) (line 341)
        str_54428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 15), 'str', 'depends')
        # Getting the type of 'ext' (line 341)
        ext_54429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 25), 'ext')
        # Obtaining the member 'depends' of a type (line 341)
        depends_54430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 25), ext_54429, 'depends')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 14), dict_54427, (str_54428, depends_54430))
        
        # Assigning a type to the variable 'kws' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'kws', dict_54427)
        
        # Assigning a Attribute to a Name (line 342):
        
        # Assigning a Attribute to a Name (line 342):
        # Getting the type of 'self' (line 342)
        self_54431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 21), 'self')
        # Obtaining the member 'build_temp' of a type (line 342)
        build_temp_54432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 21), self_54431, 'build_temp')
        # Assigning a type to the variable 'output_dir' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'output_dir', build_temp_54432)
        
        # Assigning a BinOp to a Name (line 344):
        
        # Assigning a BinOp to a Name (line 344):
        # Getting the type of 'ext' (line 344)
        ext_54433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 23), 'ext')
        # Obtaining the member 'include_dirs' of a type (line 344)
        include_dirs_54434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 23), ext_54433, 'include_dirs')
        
        # Call to get_numpy_include_dirs(...): (line 344)
        # Processing the call keyword arguments (line 344)
        kwargs_54436 = {}
        # Getting the type of 'get_numpy_include_dirs' (line 344)
        get_numpy_include_dirs_54435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 42), 'get_numpy_include_dirs', False)
        # Calling get_numpy_include_dirs(args, kwargs) (line 344)
        get_numpy_include_dirs_call_result_54437 = invoke(stypy.reporting.localization.Localization(__file__, 344, 42), get_numpy_include_dirs_54435, *[], **kwargs_54436)
        
        # Applying the binary operator '+' (line 344)
        result_add_54438 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 23), '+', include_dirs_54434, get_numpy_include_dirs_call_result_54437)
        
        # Assigning a type to the variable 'include_dirs' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'include_dirs', result_add_54438)
        
        # Assigning a List to a Name (line 346):
        
        # Assigning a List to a Name (line 346):
        
        # Obtaining an instance of the builtin type 'list' (line 346)
        list_54439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 346)
        
        # Assigning a type to the variable 'c_objects' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'c_objects', list_54439)
        
        # Getting the type of 'c_sources' (line 347)
        c_sources_54440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 11), 'c_sources')
        # Testing the type of an if condition (line 347)
        if_condition_54441 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 347, 8), c_sources_54440)
        # Assigning a type to the variable 'if_condition_54441' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'if_condition_54441', if_condition_54441)
        # SSA begins for if statement (line 347)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 348)
        # Processing the call arguments (line 348)
        str_54444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 21), 'str', 'compiling C sources')
        # Processing the call keyword arguments (line 348)
        kwargs_54445 = {}
        # Getting the type of 'log' (line 348)
        log_54442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 12), 'log', False)
        # Obtaining the member 'info' of a type (line 348)
        info_54443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 12), log_54442, 'info')
        # Calling info(args, kwargs) (line 348)
        info_call_result_54446 = invoke(stypy.reporting.localization.Localization(__file__, 348, 12), info_54443, *[str_54444], **kwargs_54445)
        
        
        # Assigning a Call to a Name (line 349):
        
        # Assigning a Call to a Name (line 349):
        
        # Call to compile(...): (line 349)
        # Processing the call arguments (line 349)
        # Getting the type of 'c_sources' (line 349)
        c_sources_54450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 46), 'c_sources', False)
        # Processing the call keyword arguments (line 349)
        # Getting the type of 'output_dir' (line 350)
        output_dir_54451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 57), 'output_dir', False)
        keyword_54452 = output_dir_54451
        # Getting the type of 'macros' (line 351)
        macros_54453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 53), 'macros', False)
        keyword_54454 = macros_54453
        # Getting the type of 'include_dirs' (line 352)
        include_dirs_54455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 59), 'include_dirs', False)
        keyword_54456 = include_dirs_54455
        # Getting the type of 'self' (line 353)
        self_54457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 52), 'self', False)
        # Obtaining the member 'debug' of a type (line 353)
        debug_54458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 52), self_54457, 'debug')
        keyword_54459 = debug_54458
        # Getting the type of 'extra_args' (line 354)
        extra_args_54460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 61), 'extra_args', False)
        keyword_54461 = extra_args_54460
        # Getting the type of 'kws' (line 355)
        kws_54462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 48), 'kws', False)
        kwargs_54463 = {'kws_54462': kws_54462, 'macros': keyword_54454, 'extra_postargs': keyword_54461, 'output_dir': keyword_54452, 'debug': keyword_54459, 'include_dirs': keyword_54456}
        # Getting the type of 'self' (line 349)
        self_54447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 24), 'self', False)
        # Obtaining the member 'compiler' of a type (line 349)
        compiler_54448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 24), self_54447, 'compiler')
        # Obtaining the member 'compile' of a type (line 349)
        compile_54449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 24), compiler_54448, 'compile')
        # Calling compile(args, kwargs) (line 349)
        compile_call_result_54464 = invoke(stypy.reporting.localization.Localization(__file__, 349, 24), compile_54449, *[c_sources_54450], **kwargs_54463)
        
        # Assigning a type to the variable 'c_objects' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 12), 'c_objects', compile_call_result_54464)
        # SSA join for if statement (line 347)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'cxx_sources' (line 357)
        cxx_sources_54465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 11), 'cxx_sources')
        # Testing the type of an if condition (line 357)
        if_condition_54466 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 357, 8), cxx_sources_54465)
        # Assigning a type to the variable 'if_condition_54466' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'if_condition_54466', if_condition_54466)
        # SSA begins for if statement (line 357)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 358)
        # Processing the call arguments (line 358)
        str_54469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 21), 'str', 'compiling C++ sources')
        # Processing the call keyword arguments (line 358)
        kwargs_54470 = {}
        # Getting the type of 'log' (line 358)
        log_54467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'log', False)
        # Obtaining the member 'info' of a type (line 358)
        info_54468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 12), log_54467, 'info')
        # Calling info(args, kwargs) (line 358)
        info_call_result_54471 = invoke(stypy.reporting.localization.Localization(__file__, 358, 12), info_54468, *[str_54469], **kwargs_54470)
        
        
        # Getting the type of 'c_objects' (line 359)
        c_objects_54472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'c_objects')
        
        # Call to compile(...): (line 359)
        # Processing the call arguments (line 359)
        # Getting the type of 'cxx_sources' (line 359)
        cxx_sources_54475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 46), 'cxx_sources', False)
        # Processing the call keyword arguments (line 359)
        # Getting the type of 'output_dir' (line 360)
        output_dir_54476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 57), 'output_dir', False)
        keyword_54477 = output_dir_54476
        # Getting the type of 'macros' (line 361)
        macros_54478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 53), 'macros', False)
        keyword_54479 = macros_54478
        # Getting the type of 'include_dirs' (line 362)
        include_dirs_54480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 59), 'include_dirs', False)
        keyword_54481 = include_dirs_54480
        # Getting the type of 'self' (line 363)
        self_54482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 52), 'self', False)
        # Obtaining the member 'debug' of a type (line 363)
        debug_54483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 52), self_54482, 'debug')
        keyword_54484 = debug_54483
        # Getting the type of 'extra_args' (line 364)
        extra_args_54485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 61), 'extra_args', False)
        keyword_54486 = extra_args_54485
        # Getting the type of 'kws' (line 365)
        kws_54487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 48), 'kws', False)
        kwargs_54488 = {'kws_54487': kws_54487, 'macros': keyword_54479, 'extra_postargs': keyword_54486, 'output_dir': keyword_54477, 'debug': keyword_54484, 'include_dirs': keyword_54481}
        # Getting the type of 'cxx_compiler' (line 359)
        cxx_compiler_54473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 25), 'cxx_compiler', False)
        # Obtaining the member 'compile' of a type (line 359)
        compile_54474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 25), cxx_compiler_54473, 'compile')
        # Calling compile(args, kwargs) (line 359)
        compile_call_result_54489 = invoke(stypy.reporting.localization.Localization(__file__, 359, 25), compile_54474, *[cxx_sources_54475], **kwargs_54488)
        
        # Applying the binary operator '+=' (line 359)
        result_iadd_54490 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 12), '+=', c_objects_54472, compile_call_result_54489)
        # Assigning a type to the variable 'c_objects' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'c_objects', result_iadd_54490)
        
        # SSA join for if statement (line 357)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 367):
        
        # Assigning a List to a Name (line 367):
        
        # Obtaining an instance of the builtin type 'list' (line 367)
        list_54491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 367)
        
        # Assigning a type to the variable 'extra_postargs' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'extra_postargs', list_54491)
        
        # Assigning a List to a Name (line 368):
        
        # Assigning a List to a Name (line 368):
        
        # Obtaining an instance of the builtin type 'list' (line 368)
        list_54492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 368)
        
        # Assigning a type to the variable 'f_objects' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'f_objects', list_54492)
        
        # Getting the type of 'fmodule_sources' (line 369)
        fmodule_sources_54493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 11), 'fmodule_sources')
        # Testing the type of an if condition (line 369)
        if_condition_54494 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 369, 8), fmodule_sources_54493)
        # Assigning a type to the variable 'if_condition_54494' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'if_condition_54494', if_condition_54494)
        # SSA begins for if statement (line 369)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 370)
        # Processing the call arguments (line 370)
        str_54497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 21), 'str', 'compiling Fortran 90 module sources')
        # Processing the call keyword arguments (line 370)
        kwargs_54498 = {}
        # Getting the type of 'log' (line 370)
        log_54495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'log', False)
        # Obtaining the member 'info' of a type (line 370)
        info_54496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 12), log_54495, 'info')
        # Calling info(args, kwargs) (line 370)
        info_call_result_54499 = invoke(stypy.reporting.localization.Localization(__file__, 370, 12), info_54496, *[str_54497], **kwargs_54498)
        
        
        # Assigning a Subscript to a Name (line 371):
        
        # Assigning a Subscript to a Name (line 371):
        
        # Obtaining the type of the subscript
        slice_54500 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 371, 26), None, None, None)
        # Getting the type of 'ext' (line 371)
        ext_54501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 26), 'ext')
        # Obtaining the member 'module_dirs' of a type (line 371)
        module_dirs_54502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 26), ext_54501, 'module_dirs')
        # Obtaining the member '__getitem__' of a type (line 371)
        getitem___54503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 26), module_dirs_54502, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 371)
        subscript_call_result_54504 = invoke(stypy.reporting.localization.Localization(__file__, 371, 26), getitem___54503, slice_54500)
        
        # Assigning a type to the variable 'module_dirs' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 12), 'module_dirs', subscript_call_result_54504)
        
        # Assigning a Call to a Name (line 372):
        
        # Assigning a Call to a Name (line 372):
        
        # Call to join(...): (line 372)
        # Processing the call arguments (line 372)
        # Getting the type of 'self' (line 373)
        self_54508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 16), 'self', False)
        # Obtaining the member 'build_temp' of a type (line 373)
        build_temp_54509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 16), self_54508, 'build_temp')
        
        # Call to dirname(...): (line 373)
        # Processing the call arguments (line 373)
        
        # Call to get_ext_filename(...): (line 374)
        # Processing the call arguments (line 374)
        # Getting the type of 'fullname' (line 374)
        fullname_54515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 42), 'fullname', False)
        # Processing the call keyword arguments (line 374)
        kwargs_54516 = {}
        # Getting the type of 'self' (line 374)
        self_54513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 20), 'self', False)
        # Obtaining the member 'get_ext_filename' of a type (line 374)
        get_ext_filename_54514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 20), self_54513, 'get_ext_filename')
        # Calling get_ext_filename(args, kwargs) (line 374)
        get_ext_filename_call_result_54517 = invoke(stypy.reporting.localization.Localization(__file__, 374, 20), get_ext_filename_54514, *[fullname_54515], **kwargs_54516)
        
        # Processing the call keyword arguments (line 373)
        kwargs_54518 = {}
        # Getting the type of 'os' (line 373)
        os_54510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 33), 'os', False)
        # Obtaining the member 'path' of a type (line 373)
        path_54511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 33), os_54510, 'path')
        # Obtaining the member 'dirname' of a type (line 373)
        dirname_54512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 33), path_54511, 'dirname')
        # Calling dirname(args, kwargs) (line 373)
        dirname_call_result_54519 = invoke(stypy.reporting.localization.Localization(__file__, 373, 33), dirname_54512, *[get_ext_filename_call_result_54517], **kwargs_54518)
        
        # Processing the call keyword arguments (line 372)
        kwargs_54520 = {}
        # Getting the type of 'os' (line 372)
        os_54505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 31), 'os', False)
        # Obtaining the member 'path' of a type (line 372)
        path_54506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 31), os_54505, 'path')
        # Obtaining the member 'join' of a type (line 372)
        join_54507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 31), path_54506, 'join')
        # Calling join(args, kwargs) (line 372)
        join_call_result_54521 = invoke(stypy.reporting.localization.Localization(__file__, 372, 31), join_54507, *[build_temp_54509, dirname_call_result_54519], **kwargs_54520)
        
        # Assigning a type to the variable 'module_build_dir' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'module_build_dir', join_call_result_54521)
        
        # Call to mkpath(...): (line 376)
        # Processing the call arguments (line 376)
        # Getting the type of 'module_build_dir' (line 376)
        module_build_dir_54524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 24), 'module_build_dir', False)
        # Processing the call keyword arguments (line 376)
        kwargs_54525 = {}
        # Getting the type of 'self' (line 376)
        self_54522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 376)
        mkpath_54523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 12), self_54522, 'mkpath')
        # Calling mkpath(args, kwargs) (line 376)
        mkpath_call_result_54526 = invoke(stypy.reporting.localization.Localization(__file__, 376, 12), mkpath_54523, *[module_build_dir_54524], **kwargs_54525)
        
        
        # Type idiom detected: calculating its left and rigth part (line 377)
        # Getting the type of 'fcompiler' (line 377)
        fcompiler_54527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 15), 'fcompiler')
        # Obtaining the member 'module_dir_switch' of a type (line 377)
        module_dir_switch_54528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 15), fcompiler_54527, 'module_dir_switch')
        # Getting the type of 'None' (line 377)
        None_54529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 46), 'None')
        
        (may_be_54530, more_types_in_union_54531) = may_be_none(module_dir_switch_54528, None_54529)

        if may_be_54530:

            if more_types_in_union_54531:
                # Runtime conditional SSA (line 377)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 378):
            
            # Assigning a Call to a Name (line 378):
            
            # Call to glob(...): (line 378)
            # Processing the call arguments (line 378)
            str_54533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 40), 'str', '*.mod')
            # Processing the call keyword arguments (line 378)
            kwargs_54534 = {}
            # Getting the type of 'glob' (line 378)
            glob_54532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 35), 'glob', False)
            # Calling glob(args, kwargs) (line 378)
            glob_call_result_54535 = invoke(stypy.reporting.localization.Localization(__file__, 378, 35), glob_54532, *[str_54533], **kwargs_54534)
            
            # Assigning a type to the variable 'existing_modules' (line 378)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 16), 'existing_modules', glob_call_result_54535)

            if more_types_in_union_54531:
                # SSA join for if statement (line 377)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'extra_postargs' (line 379)
        extra_postargs_54536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 12), 'extra_postargs')
        
        # Call to module_options(...): (line 379)
        # Processing the call arguments (line 379)
        # Getting the type of 'module_dirs' (line 380)
        module_dirs_54539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 16), 'module_dirs', False)
        # Getting the type of 'module_build_dir' (line 380)
        module_build_dir_54540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 29), 'module_build_dir', False)
        # Processing the call keyword arguments (line 379)
        kwargs_54541 = {}
        # Getting the type of 'fcompiler' (line 379)
        fcompiler_54537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 30), 'fcompiler', False)
        # Obtaining the member 'module_options' of a type (line 379)
        module_options_54538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 30), fcompiler_54537, 'module_options')
        # Calling module_options(args, kwargs) (line 379)
        module_options_call_result_54542 = invoke(stypy.reporting.localization.Localization(__file__, 379, 30), module_options_54538, *[module_dirs_54539, module_build_dir_54540], **kwargs_54541)
        
        # Applying the binary operator '+=' (line 379)
        result_iadd_54543 = python_operator(stypy.reporting.localization.Localization(__file__, 379, 12), '+=', extra_postargs_54536, module_options_call_result_54542)
        # Assigning a type to the variable 'extra_postargs' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 12), 'extra_postargs', result_iadd_54543)
        
        
        # Getting the type of 'f_objects' (line 381)
        f_objects_54544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 12), 'f_objects')
        
        # Call to compile(...): (line 381)
        # Processing the call arguments (line 381)
        # Getting the type of 'fmodule_sources' (line 381)
        fmodule_sources_54547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 43), 'fmodule_sources', False)
        # Processing the call keyword arguments (line 381)
        # Getting the type of 'self' (line 382)
        self_54548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 54), 'self', False)
        # Obtaining the member 'build_temp' of a type (line 382)
        build_temp_54549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 54), self_54548, 'build_temp')
        keyword_54550 = build_temp_54549
        # Getting the type of 'macros' (line 383)
        macros_54551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 50), 'macros', False)
        keyword_54552 = macros_54551
        # Getting the type of 'include_dirs' (line 384)
        include_dirs_54553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 56), 'include_dirs', False)
        keyword_54554 = include_dirs_54553
        # Getting the type of 'self' (line 385)
        self_54555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 49), 'self', False)
        # Obtaining the member 'debug' of a type (line 385)
        debug_54556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 49), self_54555, 'debug')
        keyword_54557 = debug_54556
        # Getting the type of 'extra_postargs' (line 386)
        extra_postargs_54558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 58), 'extra_postargs', False)
        keyword_54559 = extra_postargs_54558
        # Getting the type of 'ext' (line 387)
        ext_54560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 51), 'ext', False)
        # Obtaining the member 'depends' of a type (line 387)
        depends_54561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 51), ext_54560, 'depends')
        keyword_54562 = depends_54561
        kwargs_54563 = {'depends': keyword_54562, 'macros': keyword_54552, 'extra_postargs': keyword_54559, 'output_dir': keyword_54550, 'debug': keyword_54557, 'include_dirs': keyword_54554}
        # Getting the type of 'fcompiler' (line 381)
        fcompiler_54545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 25), 'fcompiler', False)
        # Obtaining the member 'compile' of a type (line 381)
        compile_54546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 25), fcompiler_54545, 'compile')
        # Calling compile(args, kwargs) (line 381)
        compile_call_result_54564 = invoke(stypy.reporting.localization.Localization(__file__, 381, 25), compile_54546, *[fmodule_sources_54547], **kwargs_54563)
        
        # Applying the binary operator '+=' (line 381)
        result_iadd_54565 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 12), '+=', f_objects_54544, compile_call_result_54564)
        # Assigning a type to the variable 'f_objects' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 12), 'f_objects', result_iadd_54565)
        
        
        # Type idiom detected: calculating its left and rigth part (line 389)
        # Getting the type of 'fcompiler' (line 389)
        fcompiler_54566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 15), 'fcompiler')
        # Obtaining the member 'module_dir_switch' of a type (line 389)
        module_dir_switch_54567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 15), fcompiler_54566, 'module_dir_switch')
        # Getting the type of 'None' (line 389)
        None_54568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 46), 'None')
        
        (may_be_54569, more_types_in_union_54570) = may_be_none(module_dir_switch_54567, None_54568)

        if may_be_54569:

            if more_types_in_union_54570:
                # Runtime conditional SSA (line 389)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Call to glob(...): (line 390)
            # Processing the call arguments (line 390)
            str_54572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 30), 'str', '*.mod')
            # Processing the call keyword arguments (line 390)
            kwargs_54573 = {}
            # Getting the type of 'glob' (line 390)
            glob_54571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 25), 'glob', False)
            # Calling glob(args, kwargs) (line 390)
            glob_call_result_54574 = invoke(stypy.reporting.localization.Localization(__file__, 390, 25), glob_54571, *[str_54572], **kwargs_54573)
            
            # Testing the type of a for loop iterable (line 390)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 390, 16), glob_call_result_54574)
            # Getting the type of the for loop variable (line 390)
            for_loop_var_54575 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 390, 16), glob_call_result_54574)
            # Assigning a type to the variable 'f' (line 390)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 16), 'f', for_loop_var_54575)
            # SSA begins for a for statement (line 390)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Getting the type of 'f' (line 391)
            f_54576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 23), 'f')
            # Getting the type of 'existing_modules' (line 391)
            existing_modules_54577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 28), 'existing_modules')
            # Applying the binary operator 'in' (line 391)
            result_contains_54578 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 23), 'in', f_54576, existing_modules_54577)
            
            # Testing the type of an if condition (line 391)
            if_condition_54579 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 391, 20), result_contains_54578)
            # Assigning a type to the variable 'if_condition_54579' (line 391)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 20), 'if_condition_54579', if_condition_54579)
            # SSA begins for if statement (line 391)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 391)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 393):
            
            # Assigning a Call to a Name (line 393):
            
            # Call to join(...): (line 393)
            # Processing the call arguments (line 393)
            # Getting the type of 'module_build_dir' (line 393)
            module_build_dir_54583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 37), 'module_build_dir', False)
            # Getting the type of 'f' (line 393)
            f_54584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 55), 'f', False)
            # Processing the call keyword arguments (line 393)
            kwargs_54585 = {}
            # Getting the type of 'os' (line 393)
            os_54580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 24), 'os', False)
            # Obtaining the member 'path' of a type (line 393)
            path_54581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 24), os_54580, 'path')
            # Obtaining the member 'join' of a type (line 393)
            join_54582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 24), path_54581, 'join')
            # Calling join(args, kwargs) (line 393)
            join_call_result_54586 = invoke(stypy.reporting.localization.Localization(__file__, 393, 24), join_54582, *[module_build_dir_54583, f_54584], **kwargs_54585)
            
            # Assigning a type to the variable 't' (line 393)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 20), 't', join_call_result_54586)
            
            
            
            # Call to abspath(...): (line 394)
            # Processing the call arguments (line 394)
            # Getting the type of 'f' (line 394)
            f_54590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 39), 'f', False)
            # Processing the call keyword arguments (line 394)
            kwargs_54591 = {}
            # Getting the type of 'os' (line 394)
            os_54587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 23), 'os', False)
            # Obtaining the member 'path' of a type (line 394)
            path_54588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 23), os_54587, 'path')
            # Obtaining the member 'abspath' of a type (line 394)
            abspath_54589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 23), path_54588, 'abspath')
            # Calling abspath(args, kwargs) (line 394)
            abspath_call_result_54592 = invoke(stypy.reporting.localization.Localization(__file__, 394, 23), abspath_54589, *[f_54590], **kwargs_54591)
            
            
            # Call to abspath(...): (line 394)
            # Processing the call arguments (line 394)
            # Getting the type of 't' (line 394)
            t_54596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 59), 't', False)
            # Processing the call keyword arguments (line 394)
            kwargs_54597 = {}
            # Getting the type of 'os' (line 394)
            os_54593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 43), 'os', False)
            # Obtaining the member 'path' of a type (line 394)
            path_54594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 43), os_54593, 'path')
            # Obtaining the member 'abspath' of a type (line 394)
            abspath_54595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 43), path_54594, 'abspath')
            # Calling abspath(args, kwargs) (line 394)
            abspath_call_result_54598 = invoke(stypy.reporting.localization.Localization(__file__, 394, 43), abspath_54595, *[t_54596], **kwargs_54597)
            
            # Applying the binary operator '==' (line 394)
            result_eq_54599 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 23), '==', abspath_call_result_54592, abspath_call_result_54598)
            
            # Testing the type of an if condition (line 394)
            if_condition_54600 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 394, 20), result_eq_54599)
            # Assigning a type to the variable 'if_condition_54600' (line 394)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 20), 'if_condition_54600', if_condition_54600)
            # SSA begins for if statement (line 394)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 394)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Call to isfile(...): (line 396)
            # Processing the call arguments (line 396)
            # Getting the type of 't' (line 396)
            t_54604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 38), 't', False)
            # Processing the call keyword arguments (line 396)
            kwargs_54605 = {}
            # Getting the type of 'os' (line 396)
            os_54601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 23), 'os', False)
            # Obtaining the member 'path' of a type (line 396)
            path_54602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 23), os_54601, 'path')
            # Obtaining the member 'isfile' of a type (line 396)
            isfile_54603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 23), path_54602, 'isfile')
            # Calling isfile(args, kwargs) (line 396)
            isfile_call_result_54606 = invoke(stypy.reporting.localization.Localization(__file__, 396, 23), isfile_54603, *[t_54604], **kwargs_54605)
            
            # Testing the type of an if condition (line 396)
            if_condition_54607 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 396, 20), isfile_call_result_54606)
            # Assigning a type to the variable 'if_condition_54607' (line 396)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 20), 'if_condition_54607', if_condition_54607)
            # SSA begins for if statement (line 396)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to remove(...): (line 397)
            # Processing the call arguments (line 397)
            # Getting the type of 't' (line 397)
            t_54610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 34), 't', False)
            # Processing the call keyword arguments (line 397)
            kwargs_54611 = {}
            # Getting the type of 'os' (line 397)
            os_54608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 24), 'os', False)
            # Obtaining the member 'remove' of a type (line 397)
            remove_54609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 24), os_54608, 'remove')
            # Calling remove(args, kwargs) (line 397)
            remove_call_result_54612 = invoke(stypy.reporting.localization.Localization(__file__, 397, 24), remove_54609, *[t_54610], **kwargs_54611)
            
            # SSA join for if statement (line 396)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # SSA begins for try-except statement (line 398)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Call to move_file(...): (line 399)
            # Processing the call arguments (line 399)
            # Getting the type of 'f' (line 399)
            f_54615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 39), 'f', False)
            # Getting the type of 'module_build_dir' (line 399)
            module_build_dir_54616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 42), 'module_build_dir', False)
            # Processing the call keyword arguments (line 399)
            kwargs_54617 = {}
            # Getting the type of 'self' (line 399)
            self_54613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 24), 'self', False)
            # Obtaining the member 'move_file' of a type (line 399)
            move_file_54614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 24), self_54613, 'move_file')
            # Calling move_file(args, kwargs) (line 399)
            move_file_call_result_54618 = invoke(stypy.reporting.localization.Localization(__file__, 399, 24), move_file_54614, *[f_54615, module_build_dir_54616], **kwargs_54617)
            
            # SSA branch for the except part of a try statement (line 398)
            # SSA branch for the except 'DistutilsFileError' branch of a try statement (line 398)
            module_type_store.open_ssa_branch('except')
            
            # Call to warn(...): (line 401)
            # Processing the call arguments (line 401)
            str_54621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 33), 'str', 'failed to move %r to %r')
            
            # Obtaining an instance of the builtin type 'tuple' (line 402)
            tuple_54622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 34), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 402)
            # Adding element type (line 402)
            # Getting the type of 'f' (line 402)
            f_54623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 34), 'f', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 34), tuple_54622, f_54623)
            # Adding element type (line 402)
            # Getting the type of 'module_build_dir' (line 402)
            module_build_dir_54624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 37), 'module_build_dir', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 34), tuple_54622, module_build_dir_54624)
            
            # Applying the binary operator '%' (line 401)
            result_mod_54625 = python_operator(stypy.reporting.localization.Localization(__file__, 401, 33), '%', str_54621, tuple_54622)
            
            # Processing the call keyword arguments (line 401)
            kwargs_54626 = {}
            # Getting the type of 'log' (line 401)
            log_54619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 24), 'log', False)
            # Obtaining the member 'warn' of a type (line 401)
            warn_54620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 24), log_54619, 'warn')
            # Calling warn(args, kwargs) (line 401)
            warn_call_result_54627 = invoke(stypy.reporting.localization.Localization(__file__, 401, 24), warn_54620, *[result_mod_54625], **kwargs_54626)
            
            # SSA join for try-except statement (line 398)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_54570:
                # SSA join for if statement (line 389)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 369)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'f_sources' (line 403)
        f_sources_54628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 11), 'f_sources')
        # Testing the type of an if condition (line 403)
        if_condition_54629 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 403, 8), f_sources_54628)
        # Assigning a type to the variable 'if_condition_54629' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'if_condition_54629', if_condition_54629)
        # SSA begins for if statement (line 403)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 404)
        # Processing the call arguments (line 404)
        str_54632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 21), 'str', 'compiling Fortran sources')
        # Processing the call keyword arguments (line 404)
        kwargs_54633 = {}
        # Getting the type of 'log' (line 404)
        log_54630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 12), 'log', False)
        # Obtaining the member 'info' of a type (line 404)
        info_54631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 12), log_54630, 'info')
        # Calling info(args, kwargs) (line 404)
        info_call_result_54634 = invoke(stypy.reporting.localization.Localization(__file__, 404, 12), info_54631, *[str_54632], **kwargs_54633)
        
        
        # Getting the type of 'f_objects' (line 405)
        f_objects_54635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 12), 'f_objects')
        
        # Call to compile(...): (line 405)
        # Processing the call arguments (line 405)
        # Getting the type of 'f_sources' (line 405)
        f_sources_54638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 43), 'f_sources', False)
        # Processing the call keyword arguments (line 405)
        # Getting the type of 'self' (line 406)
        self_54639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 54), 'self', False)
        # Obtaining the member 'build_temp' of a type (line 406)
        build_temp_54640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 54), self_54639, 'build_temp')
        keyword_54641 = build_temp_54640
        # Getting the type of 'macros' (line 407)
        macros_54642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 50), 'macros', False)
        keyword_54643 = macros_54642
        # Getting the type of 'include_dirs' (line 408)
        include_dirs_54644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 56), 'include_dirs', False)
        keyword_54645 = include_dirs_54644
        # Getting the type of 'self' (line 409)
        self_54646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 49), 'self', False)
        # Obtaining the member 'debug' of a type (line 409)
        debug_54647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 49), self_54646, 'debug')
        keyword_54648 = debug_54647
        # Getting the type of 'extra_postargs' (line 410)
        extra_postargs_54649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 58), 'extra_postargs', False)
        keyword_54650 = extra_postargs_54649
        # Getting the type of 'ext' (line 411)
        ext_54651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 51), 'ext', False)
        # Obtaining the member 'depends' of a type (line 411)
        depends_54652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 51), ext_54651, 'depends')
        keyword_54653 = depends_54652
        kwargs_54654 = {'depends': keyword_54653, 'macros': keyword_54643, 'extra_postargs': keyword_54650, 'output_dir': keyword_54641, 'debug': keyword_54648, 'include_dirs': keyword_54645}
        # Getting the type of 'fcompiler' (line 405)
        fcompiler_54636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 25), 'fcompiler', False)
        # Obtaining the member 'compile' of a type (line 405)
        compile_54637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 25), fcompiler_54636, 'compile')
        # Calling compile(args, kwargs) (line 405)
        compile_call_result_54655 = invoke(stypy.reporting.localization.Localization(__file__, 405, 25), compile_54637, *[f_sources_54638], **kwargs_54654)
        
        # Applying the binary operator '+=' (line 405)
        result_iadd_54656 = python_operator(stypy.reporting.localization.Localization(__file__, 405, 12), '+=', f_objects_54635, compile_call_result_54655)
        # Assigning a type to the variable 'f_objects' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 12), 'f_objects', result_iadd_54656)
        
        # SSA join for if statement (line 403)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 413):
        
        # Assigning a BinOp to a Name (line 413):
        # Getting the type of 'c_objects' (line 413)
        c_objects_54657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 18), 'c_objects')
        # Getting the type of 'f_objects' (line 413)
        f_objects_54658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 30), 'f_objects')
        # Applying the binary operator '+' (line 413)
        result_add_54659 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 18), '+', c_objects_54657, f_objects_54658)
        
        # Assigning a type to the variable 'objects' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'objects', result_add_54659)
        
        # Getting the type of 'ext' (line 415)
        ext_54660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 11), 'ext')
        # Obtaining the member 'extra_objects' of a type (line 415)
        extra_objects_54661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 11), ext_54660, 'extra_objects')
        # Testing the type of an if condition (line 415)
        if_condition_54662 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 415, 8), extra_objects_54661)
        # Assigning a type to the variable 'if_condition_54662' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'if_condition_54662', if_condition_54662)
        # SSA begins for if statement (line 415)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 416)
        # Processing the call arguments (line 416)
        # Getting the type of 'ext' (line 416)
        ext_54665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 27), 'ext', False)
        # Obtaining the member 'extra_objects' of a type (line 416)
        extra_objects_54666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 27), ext_54665, 'extra_objects')
        # Processing the call keyword arguments (line 416)
        kwargs_54667 = {}
        # Getting the type of 'objects' (line 416)
        objects_54663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 12), 'objects', False)
        # Obtaining the member 'extend' of a type (line 416)
        extend_54664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 12), objects_54663, 'extend')
        # Calling extend(args, kwargs) (line 416)
        extend_call_result_54668 = invoke(stypy.reporting.localization.Localization(__file__, 416, 12), extend_54664, *[extra_objects_54666], **kwargs_54667)
        
        # SSA join for if statement (line 415)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BoolOp to a Name (line 417):
        
        # Assigning a BoolOp to a Name (line 417):
        
        # Evaluating a boolean operation
        # Getting the type of 'ext' (line 417)
        ext_54669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 21), 'ext')
        # Obtaining the member 'extra_link_args' of a type (line 417)
        extra_link_args_54670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 21), ext_54669, 'extra_link_args')
        
        # Obtaining an instance of the builtin type 'list' (line 417)
        list_54671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 417)
        
        # Applying the binary operator 'or' (line 417)
        result_or_keyword_54672 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 21), 'or', extra_link_args_54670, list_54671)
        
        # Assigning a type to the variable 'extra_args' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'extra_args', result_or_keyword_54672)
        
        # Assigning a Subscript to a Name (line 418):
        
        # Assigning a Subscript to a Name (line 418):
        
        # Obtaining the type of the subscript
        slice_54673 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 418, 20), None, None, None)
        
        # Call to get_libraries(...): (line 418)
        # Processing the call arguments (line 418)
        # Getting the type of 'ext' (line 418)
        ext_54676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 39), 'ext', False)
        # Processing the call keyword arguments (line 418)
        kwargs_54677 = {}
        # Getting the type of 'self' (line 418)
        self_54674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 20), 'self', False)
        # Obtaining the member 'get_libraries' of a type (line 418)
        get_libraries_54675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 20), self_54674, 'get_libraries')
        # Calling get_libraries(args, kwargs) (line 418)
        get_libraries_call_result_54678 = invoke(stypy.reporting.localization.Localization(__file__, 418, 20), get_libraries_54675, *[ext_54676], **kwargs_54677)
        
        # Obtaining the member '__getitem__' of a type (line 418)
        getitem___54679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 20), get_libraries_call_result_54678, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 418)
        subscript_call_result_54680 = invoke(stypy.reporting.localization.Localization(__file__, 418, 20), getitem___54679, slice_54673)
        
        # Assigning a type to the variable 'libraries' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'libraries', subscript_call_result_54680)
        
        # Assigning a Subscript to a Name (line 419):
        
        # Assigning a Subscript to a Name (line 419):
        
        # Obtaining the type of the subscript
        slice_54681 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 419, 23), None, None, None)
        # Getting the type of 'ext' (line 419)
        ext_54682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 23), 'ext')
        # Obtaining the member 'library_dirs' of a type (line 419)
        library_dirs_54683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 23), ext_54682, 'library_dirs')
        # Obtaining the member '__getitem__' of a type (line 419)
        getitem___54684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 23), library_dirs_54683, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 419)
        subscript_call_result_54685 = invoke(stypy.reporting.localization.Localization(__file__, 419, 23), getitem___54684, slice_54681)
        
        # Assigning a type to the variable 'library_dirs' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'library_dirs', subscript_call_result_54685)
        
        # Assigning a Attribute to a Name (line 421):
        
        # Assigning a Attribute to a Name (line 421):
        # Getting the type of 'self' (line 421)
        self_54686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 17), 'self')
        # Obtaining the member 'compiler' of a type (line 421)
        compiler_54687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 17), self_54686, 'compiler')
        # Obtaining the member 'link_shared_object' of a type (line 421)
        link_shared_object_54688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 17), compiler_54687, 'link_shared_object')
        # Assigning a type to the variable 'linker' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'linker', link_shared_object_54688)
        
        
        # Getting the type of 'self' (line 423)
        self_54689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 11), 'self')
        # Obtaining the member 'compiler' of a type (line 423)
        compiler_54690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 11), self_54689, 'compiler')
        # Obtaining the member 'compiler_type' of a type (line 423)
        compiler_type_54691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 11), compiler_54690, 'compiler_type')
        
        # Obtaining an instance of the builtin type 'tuple' (line 423)
        tuple_54692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 423)
        # Adding element type (line 423)
        str_54693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 43), 'str', 'msvc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 43), tuple_54692, str_54693)
        # Adding element type (line 423)
        str_54694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 51), 'str', 'intelw')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 43), tuple_54692, str_54694)
        # Adding element type (line 423)
        str_54695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 61), 'str', 'intelemw')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 43), tuple_54692, str_54695)
        
        # Applying the binary operator 'in' (line 423)
        result_contains_54696 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 11), 'in', compiler_type_54691, tuple_54692)
        
        # Testing the type of an if condition (line 423)
        if_condition_54697 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 423, 8), result_contains_54696)
        # Assigning a type to the variable 'if_condition_54697' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'if_condition_54697', if_condition_54697)
        # SSA begins for if statement (line 423)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _libs_with_msvc_and_fortran(...): (line 426)
        # Processing the call arguments (line 426)
        # Getting the type of 'fcompiler' (line 426)
        fcompiler_54700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 45), 'fcompiler', False)
        # Getting the type of 'libraries' (line 426)
        libraries_54701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 56), 'libraries', False)
        # Getting the type of 'library_dirs' (line 426)
        library_dirs_54702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 67), 'library_dirs', False)
        # Processing the call keyword arguments (line 426)
        kwargs_54703 = {}
        # Getting the type of 'self' (line 426)
        self_54698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'self', False)
        # Obtaining the member '_libs_with_msvc_and_fortran' of a type (line 426)
        _libs_with_msvc_and_fortran_54699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 12), self_54698, '_libs_with_msvc_and_fortran')
        # Calling _libs_with_msvc_and_fortran(args, kwargs) (line 426)
        _libs_with_msvc_and_fortran_call_result_54704 = invoke(stypy.reporting.localization.Localization(__file__, 426, 12), _libs_with_msvc_and_fortran_54699, *[fcompiler_54700, libraries_54701, library_dirs_54702], **kwargs_54703)
        
        # SSA branch for the else part of an if statement (line 423)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'ext' (line 428)
        ext_54705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 13), 'ext')
        # Obtaining the member 'language' of a type (line 428)
        language_54706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 13), ext_54705, 'language')
        
        # Obtaining an instance of the builtin type 'list' (line 428)
        list_54707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 428)
        # Adding element type (line 428)
        str_54708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 30), 'str', 'f77')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 29), list_54707, str_54708)
        # Adding element type (line 428)
        str_54709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 37), 'str', 'f90')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 29), list_54707, str_54709)
        
        # Applying the binary operator 'in' (line 428)
        result_contains_54710 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 13), 'in', language_54706, list_54707)
        
        
        # Getting the type of 'fcompiler' (line 428)
        fcompiler_54711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 48), 'fcompiler')
        # Getting the type of 'None' (line 428)
        None_54712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 65), 'None')
        # Applying the binary operator 'isnot' (line 428)
        result_is_not_54713 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 48), 'isnot', fcompiler_54711, None_54712)
        
        # Applying the binary operator 'and' (line 428)
        result_and_keyword_54714 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 13), 'and', result_contains_54710, result_is_not_54713)
        
        # Testing the type of an if condition (line 428)
        if_condition_54715 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 428, 13), result_and_keyword_54714)
        # Assigning a type to the variable 'if_condition_54715' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 13), 'if_condition_54715', if_condition_54715)
        # SSA begins for if statement (line 428)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 429):
        
        # Assigning a Attribute to a Name (line 429):
        # Getting the type of 'fcompiler' (line 429)
        fcompiler_54716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 21), 'fcompiler')
        # Obtaining the member 'link_shared_object' of a type (line 429)
        link_shared_object_54717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 21), fcompiler_54716, 'link_shared_object')
        # Assigning a type to the variable 'linker' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'linker', link_shared_object_54717)
        # SSA join for if statement (line 428)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 423)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'ext' (line 430)
        ext_54718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 11), 'ext')
        # Obtaining the member 'language' of a type (line 430)
        language_54719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 11), ext_54718, 'language')
        str_54720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 25), 'str', 'c++')
        # Applying the binary operator '==' (line 430)
        result_eq_54721 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 11), '==', language_54719, str_54720)
        
        
        # Getting the type of 'cxx_compiler' (line 430)
        cxx_compiler_54722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 35), 'cxx_compiler')
        # Getting the type of 'None' (line 430)
        None_54723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 55), 'None')
        # Applying the binary operator 'isnot' (line 430)
        result_is_not_54724 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 35), 'isnot', cxx_compiler_54722, None_54723)
        
        # Applying the binary operator 'and' (line 430)
        result_and_keyword_54725 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 11), 'and', result_eq_54721, result_is_not_54724)
        
        # Testing the type of an if condition (line 430)
        if_condition_54726 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 430, 8), result_and_keyword_54725)
        # Assigning a type to the variable 'if_condition_54726' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'if_condition_54726', if_condition_54726)
        # SSA begins for if statement (line 430)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 431):
        
        # Assigning a Attribute to a Name (line 431):
        # Getting the type of 'cxx_compiler' (line 431)
        cxx_compiler_54727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 21), 'cxx_compiler')
        # Obtaining the member 'link_shared_object' of a type (line 431)
        link_shared_object_54728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 21), cxx_compiler_54727, 'link_shared_object')
        # Assigning a type to the variable 'linker' (line 431)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 12), 'linker', link_shared_object_54728)
        # SSA join for if statement (line 430)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to linker(...): (line 433)
        # Processing the call arguments (line 433)
        # Getting the type of 'objects' (line 433)
        objects_54730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 15), 'objects', False)
        # Getting the type of 'ext_filename' (line 433)
        ext_filename_54731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 24), 'ext_filename', False)
        # Processing the call keyword arguments (line 433)
        # Getting the type of 'libraries' (line 434)
        libraries_54732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 25), 'libraries', False)
        keyword_54733 = libraries_54732
        # Getting the type of 'library_dirs' (line 435)
        library_dirs_54734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 28), 'library_dirs', False)
        keyword_54735 = library_dirs_54734
        # Getting the type of 'ext' (line 436)
        ext_54736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 36), 'ext', False)
        # Obtaining the member 'runtime_library_dirs' of a type (line 436)
        runtime_library_dirs_54737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 36), ext_54736, 'runtime_library_dirs')
        keyword_54738 = runtime_library_dirs_54737
        # Getting the type of 'extra_args' (line 437)
        extra_args_54739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 30), 'extra_args', False)
        keyword_54740 = extra_args_54739
        
        # Call to get_export_symbols(...): (line 438)
        # Processing the call arguments (line 438)
        # Getting the type of 'ext' (line 438)
        ext_54743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 54), 'ext', False)
        # Processing the call keyword arguments (line 438)
        kwargs_54744 = {}
        # Getting the type of 'self' (line 438)
        self_54741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 30), 'self', False)
        # Obtaining the member 'get_export_symbols' of a type (line 438)
        get_export_symbols_54742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 30), self_54741, 'get_export_symbols')
        # Calling get_export_symbols(args, kwargs) (line 438)
        get_export_symbols_call_result_54745 = invoke(stypy.reporting.localization.Localization(__file__, 438, 30), get_export_symbols_54742, *[ext_54743], **kwargs_54744)
        
        keyword_54746 = get_export_symbols_call_result_54745
        # Getting the type of 'self' (line 439)
        self_54747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 21), 'self', False)
        # Obtaining the member 'debug' of a type (line 439)
        debug_54748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 21), self_54747, 'debug')
        keyword_54749 = debug_54748
        # Getting the type of 'self' (line 440)
        self_54750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 26), 'self', False)
        # Obtaining the member 'build_temp' of a type (line 440)
        build_temp_54751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 26), self_54750, 'build_temp')
        keyword_54752 = build_temp_54751
        # Getting the type of 'ext' (line 441)
        ext_54753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 27), 'ext', False)
        # Obtaining the member 'language' of a type (line 441)
        language_54754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 27), ext_54753, 'language')
        keyword_54755 = language_54754
        kwargs_54756 = {'target_lang': keyword_54755, 'export_symbols': keyword_54746, 'runtime_library_dirs': keyword_54738, 'libraries': keyword_54733, 'extra_postargs': keyword_54740, 'debug': keyword_54749, 'build_temp': keyword_54752, 'library_dirs': keyword_54735}
        # Getting the type of 'linker' (line 433)
        linker_54729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'linker', False)
        # Calling linker(args, kwargs) (line 433)
        linker_call_result_54757 = invoke(stypy.reporting.localization.Localization(__file__, 433, 8), linker_54729, *[objects_54730, ext_filename_54731], **kwargs_54756)
        
        
        # ################# End of 'build_extension(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'build_extension' in the type store
        # Getting the type of 'stypy_return_type' (line 264)
        stypy_return_type_54758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_54758)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'build_extension'
        return stypy_return_type_54758


    @norecursion
    def _add_dummy_mingwex_sym(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_add_dummy_mingwex_sym'
        module_type_store = module_type_store.open_function_context('_add_dummy_mingwex_sym', 443, 4, False)
        # Assigning a type to the variable 'self' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext._add_dummy_mingwex_sym.__dict__.__setitem__('stypy_localization', localization)
        build_ext._add_dummy_mingwex_sym.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext._add_dummy_mingwex_sym.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext._add_dummy_mingwex_sym.__dict__.__setitem__('stypy_function_name', 'build_ext._add_dummy_mingwex_sym')
        build_ext._add_dummy_mingwex_sym.__dict__.__setitem__('stypy_param_names_list', ['c_sources'])
        build_ext._add_dummy_mingwex_sym.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext._add_dummy_mingwex_sym.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext._add_dummy_mingwex_sym.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext._add_dummy_mingwex_sym.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext._add_dummy_mingwex_sym.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext._add_dummy_mingwex_sym.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext._add_dummy_mingwex_sym', ['c_sources'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_add_dummy_mingwex_sym', localization, ['c_sources'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_add_dummy_mingwex_sym(...)' code ##################

        
        # Assigning a Attribute to a Name (line 444):
        
        # Assigning a Attribute to a Name (line 444):
        
        # Call to get_finalized_command(...): (line 444)
        # Processing the call arguments (line 444)
        str_54761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 47), 'str', 'build_src')
        # Processing the call keyword arguments (line 444)
        kwargs_54762 = {}
        # Getting the type of 'self' (line 444)
        self_54759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 20), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 444)
        get_finalized_command_54760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 20), self_54759, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 444)
        get_finalized_command_call_result_54763 = invoke(stypy.reporting.localization.Localization(__file__, 444, 20), get_finalized_command_54760, *[str_54761], **kwargs_54762)
        
        # Obtaining the member 'build_src' of a type (line 444)
        build_src_54764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 20), get_finalized_command_call_result_54763, 'build_src')
        # Assigning a type to the variable 'build_src' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'build_src', build_src_54764)
        
        # Assigning a Attribute to a Name (line 445):
        
        # Assigning a Attribute to a Name (line 445):
        
        # Call to get_finalized_command(...): (line 445)
        # Processing the call arguments (line 445)
        str_54767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 48), 'str', 'build_clib')
        # Processing the call keyword arguments (line 445)
        kwargs_54768 = {}
        # Getting the type of 'self' (line 445)
        self_54765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 21), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 445)
        get_finalized_command_54766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 21), self_54765, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 445)
        get_finalized_command_call_result_54769 = invoke(stypy.reporting.localization.Localization(__file__, 445, 21), get_finalized_command_54766, *[str_54767], **kwargs_54768)
        
        # Obtaining the member 'build_clib' of a type (line 445)
        build_clib_54770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 21), get_finalized_command_call_result_54769, 'build_clib')
        # Assigning a type to the variable 'build_clib' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'build_clib', build_clib_54770)
        
        # Assigning a Call to a Name (line 446):
        
        # Assigning a Call to a Name (line 446):
        
        # Call to compile(...): (line 446)
        # Processing the call arguments (line 446)
        
        # Obtaining an instance of the builtin type 'list' (line 446)
        list_54774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 446)
        # Adding element type (line 446)
        
        # Call to join(...): (line 446)
        # Processing the call arguments (line 446)
        # Getting the type of 'build_src' (line 446)
        build_src_54778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 54), 'build_src', False)
        str_54779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 16), 'str', 'gfortran_vs2003_hack.c')
        # Processing the call keyword arguments (line 446)
        kwargs_54780 = {}
        # Getting the type of 'os' (line 446)
        os_54775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 41), 'os', False)
        # Obtaining the member 'path' of a type (line 446)
        path_54776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 41), os_54775, 'path')
        # Obtaining the member 'join' of a type (line 446)
        join_54777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 41), path_54776, 'join')
        # Calling join(args, kwargs) (line 446)
        join_call_result_54781 = invoke(stypy.reporting.localization.Localization(__file__, 446, 41), join_54777, *[build_src_54778, str_54779], **kwargs_54780)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 40), list_54774, join_call_result_54781)
        
        # Processing the call keyword arguments (line 446)
        # Getting the type of 'self' (line 448)
        self_54782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 27), 'self', False)
        # Obtaining the member 'build_temp' of a type (line 448)
        build_temp_54783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 27), self_54782, 'build_temp')
        keyword_54784 = build_temp_54783
        kwargs_54785 = {'output_dir': keyword_54784}
        # Getting the type of 'self' (line 446)
        self_54771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 18), 'self', False)
        # Obtaining the member 'compiler' of a type (line 446)
        compiler_54772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 18), self_54771, 'compiler')
        # Obtaining the member 'compile' of a type (line 446)
        compile_54773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 18), compiler_54772, 'compile')
        # Calling compile(args, kwargs) (line 446)
        compile_call_result_54786 = invoke(stypy.reporting.localization.Localization(__file__, 446, 18), compile_54773, *[list_54774], **kwargs_54785)
        
        # Assigning a type to the variable 'objects' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'objects', compile_call_result_54786)
        
        # Call to create_static_lib(...): (line 449)
        # Processing the call arguments (line 449)
        # Getting the type of 'objects' (line 449)
        objects_54790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 40), 'objects', False)
        str_54791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 49), 'str', '_gfortran_workaround')
        # Processing the call keyword arguments (line 449)
        # Getting the type of 'build_clib' (line 449)
        build_clib_54792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 84), 'build_clib', False)
        keyword_54793 = build_clib_54792
        # Getting the type of 'self' (line 449)
        self_54794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 102), 'self', False)
        # Obtaining the member 'debug' of a type (line 449)
        debug_54795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 102), self_54794, 'debug')
        keyword_54796 = debug_54795
        kwargs_54797 = {'debug': keyword_54796, 'output_dir': keyword_54793}
        # Getting the type of 'self' (line 449)
        self_54787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'self', False)
        # Obtaining the member 'compiler' of a type (line 449)
        compiler_54788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 8), self_54787, 'compiler')
        # Obtaining the member 'create_static_lib' of a type (line 449)
        create_static_lib_54789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 8), compiler_54788, 'create_static_lib')
        # Calling create_static_lib(args, kwargs) (line 449)
        create_static_lib_call_result_54798 = invoke(stypy.reporting.localization.Localization(__file__, 449, 8), create_static_lib_54789, *[objects_54790, str_54791], **kwargs_54797)
        
        
        # ################# End of '_add_dummy_mingwex_sym(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_add_dummy_mingwex_sym' in the type store
        # Getting the type of 'stypy_return_type' (line 443)
        stypy_return_type_54799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_54799)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_add_dummy_mingwex_sym'
        return stypy_return_type_54799


    @norecursion
    def _libs_with_msvc_and_fortran(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_libs_with_msvc_and_fortran'
        module_type_store = module_type_store.open_function_context('_libs_with_msvc_and_fortran', 451, 4, False)
        # Assigning a type to the variable 'self' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext._libs_with_msvc_and_fortran.__dict__.__setitem__('stypy_localization', localization)
        build_ext._libs_with_msvc_and_fortran.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext._libs_with_msvc_and_fortran.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext._libs_with_msvc_and_fortran.__dict__.__setitem__('stypy_function_name', 'build_ext._libs_with_msvc_and_fortran')
        build_ext._libs_with_msvc_and_fortran.__dict__.__setitem__('stypy_param_names_list', ['fcompiler', 'c_libraries', 'c_library_dirs'])
        build_ext._libs_with_msvc_and_fortran.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext._libs_with_msvc_and_fortran.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext._libs_with_msvc_and_fortran.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext._libs_with_msvc_and_fortran.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext._libs_with_msvc_and_fortran.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext._libs_with_msvc_and_fortran.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext._libs_with_msvc_and_fortran', ['fcompiler', 'c_libraries', 'c_library_dirs'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_libs_with_msvc_and_fortran', localization, ['fcompiler', 'c_libraries', 'c_library_dirs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_libs_with_msvc_and_fortran(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 453)
        # Getting the type of 'fcompiler' (line 453)
        fcompiler_54800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 11), 'fcompiler')
        # Getting the type of 'None' (line 453)
        None_54801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 24), 'None')
        
        (may_be_54802, more_types_in_union_54803) = may_be_none(fcompiler_54800, None_54801)

        if may_be_54802:

            if more_types_in_union_54803:
                # Runtime conditional SSA (line 453)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'stypy_return_type' (line 453)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 30), 'stypy_return_type', types.NoneType)

            if more_types_in_union_54803:
                # SSA join for if statement (line 453)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'c_libraries' (line 455)
        c_libraries_54804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 23), 'c_libraries')
        # Testing the type of a for loop iterable (line 455)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 455, 8), c_libraries_54804)
        # Getting the type of the for loop variable (line 455)
        for_loop_var_54805 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 455, 8), c_libraries_54804)
        # Assigning a type to the variable 'libname' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'libname', for_loop_var_54805)
        # SSA begins for a for statement (line 455)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to startswith(...): (line 456)
        # Processing the call arguments (line 456)
        str_54808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 34), 'str', 'msvc')
        # Processing the call keyword arguments (line 456)
        kwargs_54809 = {}
        # Getting the type of 'libname' (line 456)
        libname_54806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 15), 'libname', False)
        # Obtaining the member 'startswith' of a type (line 456)
        startswith_54807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 15), libname_54806, 'startswith')
        # Calling startswith(args, kwargs) (line 456)
        startswith_call_result_54810 = invoke(stypy.reporting.localization.Localization(__file__, 456, 15), startswith_54807, *[str_54808], **kwargs_54809)
        
        # Testing the type of an if condition (line 456)
        if_condition_54811 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 456, 12), startswith_call_result_54810)
        # Assigning a type to the variable 'if_condition_54811' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 12), 'if_condition_54811', if_condition_54811)
        # SSA begins for if statement (line 456)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 456)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 457):
        
        # Assigning a Name to a Name (line 457):
        # Getting the type of 'False' (line 457)
        False_54812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 25), 'False')
        # Assigning a type to the variable 'fileexists' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'fileexists', False_54812)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'c_library_dirs' (line 458)
        c_library_dirs_54813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 26), 'c_library_dirs')
        
        # Obtaining an instance of the builtin type 'list' (line 458)
        list_54814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 458)
        
        # Applying the binary operator 'or' (line 458)
        result_or_keyword_54815 = python_operator(stypy.reporting.localization.Localization(__file__, 458, 26), 'or', c_library_dirs_54813, list_54814)
        
        # Testing the type of a for loop iterable (line 458)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 458, 12), result_or_keyword_54815)
        # Getting the type of the for loop variable (line 458)
        for_loop_var_54816 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 458, 12), result_or_keyword_54815)
        # Assigning a type to the variable 'libdir' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 12), 'libdir', for_loop_var_54816)
        # SSA begins for a for statement (line 458)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 459):
        
        # Assigning a Call to a Name (line 459):
        
        # Call to join(...): (line 459)
        # Processing the call arguments (line 459)
        # Getting the type of 'libdir' (line 459)
        libdir_54820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 39), 'libdir', False)
        str_54821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 47), 'str', '%s.lib')
        # Getting the type of 'libname' (line 459)
        libname_54822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 59), 'libname', False)
        # Applying the binary operator '%' (line 459)
        result_mod_54823 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 47), '%', str_54821, libname_54822)
        
        # Processing the call keyword arguments (line 459)
        kwargs_54824 = {}
        # Getting the type of 'os' (line 459)
        os_54817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 459)
        path_54818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 26), os_54817, 'path')
        # Obtaining the member 'join' of a type (line 459)
        join_54819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 26), path_54818, 'join')
        # Calling join(args, kwargs) (line 459)
        join_call_result_54825 = invoke(stypy.reporting.localization.Localization(__file__, 459, 26), join_54819, *[libdir_54820, result_mod_54823], **kwargs_54824)
        
        # Assigning a type to the variable 'libfile' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 16), 'libfile', join_call_result_54825)
        
        
        # Call to isfile(...): (line 460)
        # Processing the call arguments (line 460)
        # Getting the type of 'libfile' (line 460)
        libfile_54829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 34), 'libfile', False)
        # Processing the call keyword arguments (line 460)
        kwargs_54830 = {}
        # Getting the type of 'os' (line 460)
        os_54826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 460)
        path_54827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 19), os_54826, 'path')
        # Obtaining the member 'isfile' of a type (line 460)
        isfile_54828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 19), path_54827, 'isfile')
        # Calling isfile(args, kwargs) (line 460)
        isfile_call_result_54831 = invoke(stypy.reporting.localization.Localization(__file__, 460, 19), isfile_54828, *[libfile_54829], **kwargs_54830)
        
        # Testing the type of an if condition (line 460)
        if_condition_54832 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 460, 16), isfile_call_result_54831)
        # Assigning a type to the variable 'if_condition_54832' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 16), 'if_condition_54832', if_condition_54832)
        # SSA begins for if statement (line 460)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 461):
        
        # Assigning a Name to a Name (line 461):
        # Getting the type of 'True' (line 461)
        True_54833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 33), 'True')
        # Assigning a type to the variable 'fileexists' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 20), 'fileexists', True_54833)
        # SSA join for if statement (line 460)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'fileexists' (line 463)
        fileexists_54834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 15), 'fileexists')
        # Testing the type of an if condition (line 463)
        if_condition_54835 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 463, 12), fileexists_54834)
        # Assigning a type to the variable 'if_condition_54835' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 12), 'if_condition_54835', if_condition_54835)
        # SSA begins for if statement (line 463)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 463)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 465):
        
        # Assigning a Name to a Name (line 465):
        # Getting the type of 'False' (line 465)
        False_54836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 25), 'False')
        # Assigning a type to the variable 'fileexists' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 12), 'fileexists', False_54836)
        
        # Getting the type of 'c_library_dirs' (line 466)
        c_library_dirs_54837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 26), 'c_library_dirs')
        # Testing the type of a for loop iterable (line 466)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 466, 12), c_library_dirs_54837)
        # Getting the type of the for loop variable (line 466)
        for_loop_var_54838 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 466, 12), c_library_dirs_54837)
        # Assigning a type to the variable 'libdir' (line 466)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'libdir', for_loop_var_54838)
        # SSA begins for a for statement (line 466)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 467):
        
        # Assigning a Call to a Name (line 467):
        
        # Call to join(...): (line 467)
        # Processing the call arguments (line 467)
        # Getting the type of 'libdir' (line 467)
        libdir_54842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 39), 'libdir', False)
        str_54843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 47), 'str', 'lib%s.a')
        # Getting the type of 'libname' (line 467)
        libname_54844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 60), 'libname', False)
        # Applying the binary operator '%' (line 467)
        result_mod_54845 = python_operator(stypy.reporting.localization.Localization(__file__, 467, 47), '%', str_54843, libname_54844)
        
        # Processing the call keyword arguments (line 467)
        kwargs_54846 = {}
        # Getting the type of 'os' (line 467)
        os_54839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 467)
        path_54840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 26), os_54839, 'path')
        # Obtaining the member 'join' of a type (line 467)
        join_54841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 26), path_54840, 'join')
        # Calling join(args, kwargs) (line 467)
        join_call_result_54847 = invoke(stypy.reporting.localization.Localization(__file__, 467, 26), join_54841, *[libdir_54842, result_mod_54845], **kwargs_54846)
        
        # Assigning a type to the variable 'libfile' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 16), 'libfile', join_call_result_54847)
        
        
        # Call to isfile(...): (line 468)
        # Processing the call arguments (line 468)
        # Getting the type of 'libfile' (line 468)
        libfile_54851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 34), 'libfile', False)
        # Processing the call keyword arguments (line 468)
        kwargs_54852 = {}
        # Getting the type of 'os' (line 468)
        os_54848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 468)
        path_54849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 19), os_54848, 'path')
        # Obtaining the member 'isfile' of a type (line 468)
        isfile_54850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 19), path_54849, 'isfile')
        # Calling isfile(args, kwargs) (line 468)
        isfile_call_result_54853 = invoke(stypy.reporting.localization.Localization(__file__, 468, 19), isfile_54850, *[libfile_54851], **kwargs_54852)
        
        # Testing the type of an if condition (line 468)
        if_condition_54854 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 468, 16), isfile_call_result_54853)
        # Assigning a type to the variable 'if_condition_54854' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 16), 'if_condition_54854', if_condition_54854)
        # SSA begins for if statement (line 468)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 471):
        
        # Assigning a Call to a Name (line 471):
        
        # Call to join(...): (line 471)
        # Processing the call arguments (line 471)
        # Getting the type of 'self' (line 471)
        self_54858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 44), 'self', False)
        # Obtaining the member 'build_temp' of a type (line 471)
        build_temp_54859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 44), self_54858, 'build_temp')
        # Getting the type of 'libname' (line 471)
        libname_54860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 61), 'libname', False)
        str_54861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 71), 'str', '.lib')
        # Applying the binary operator '+' (line 471)
        result_add_54862 = python_operator(stypy.reporting.localization.Localization(__file__, 471, 61), '+', libname_54860, str_54861)
        
        # Processing the call keyword arguments (line 471)
        kwargs_54863 = {}
        # Getting the type of 'os' (line 471)
        os_54855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 31), 'os', False)
        # Obtaining the member 'path' of a type (line 471)
        path_54856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 31), os_54855, 'path')
        # Obtaining the member 'join' of a type (line 471)
        join_54857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 31), path_54856, 'join')
        # Calling join(args, kwargs) (line 471)
        join_call_result_54864 = invoke(stypy.reporting.localization.Localization(__file__, 471, 31), join_54857, *[build_temp_54859, result_add_54862], **kwargs_54863)
        
        # Assigning a type to the variable 'libfile2' (line 471)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 20), 'libfile2', join_call_result_54864)
        
        # Call to copy_file(...): (line 472)
        # Processing the call arguments (line 472)
        # Getting the type of 'libfile' (line 472)
        libfile_54866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 30), 'libfile', False)
        # Getting the type of 'libfile2' (line 472)
        libfile2_54867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 39), 'libfile2', False)
        # Processing the call keyword arguments (line 472)
        kwargs_54868 = {}
        # Getting the type of 'copy_file' (line 472)
        copy_file_54865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 20), 'copy_file', False)
        # Calling copy_file(args, kwargs) (line 472)
        copy_file_call_result_54869 = invoke(stypy.reporting.localization.Localization(__file__, 472, 20), copy_file_54865, *[libfile_54866, libfile2_54867], **kwargs_54868)
        
        
        
        # Getting the type of 'self' (line 473)
        self_54870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 23), 'self')
        # Obtaining the member 'build_temp' of a type (line 473)
        build_temp_54871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 23), self_54870, 'build_temp')
        # Getting the type of 'c_library_dirs' (line 473)
        c_library_dirs_54872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 46), 'c_library_dirs')
        # Applying the binary operator 'notin' (line 473)
        result_contains_54873 = python_operator(stypy.reporting.localization.Localization(__file__, 473, 23), 'notin', build_temp_54871, c_library_dirs_54872)
        
        # Testing the type of an if condition (line 473)
        if_condition_54874 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 473, 20), result_contains_54873)
        # Assigning a type to the variable 'if_condition_54874' (line 473)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 20), 'if_condition_54874', if_condition_54874)
        # SSA begins for if statement (line 473)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 474)
        # Processing the call arguments (line 474)
        # Getting the type of 'self' (line 474)
        self_54877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 46), 'self', False)
        # Obtaining the member 'build_temp' of a type (line 474)
        build_temp_54878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 46), self_54877, 'build_temp')
        # Processing the call keyword arguments (line 474)
        kwargs_54879 = {}
        # Getting the type of 'c_library_dirs' (line 474)
        c_library_dirs_54875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 24), 'c_library_dirs', False)
        # Obtaining the member 'append' of a type (line 474)
        append_54876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 24), c_library_dirs_54875, 'append')
        # Calling append(args, kwargs) (line 474)
        append_call_result_54880 = invoke(stypy.reporting.localization.Localization(__file__, 474, 24), append_54876, *[build_temp_54878], **kwargs_54879)
        
        # SSA join for if statement (line 473)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 475):
        
        # Assigning a Name to a Name (line 475):
        # Getting the type of 'True' (line 475)
        True_54881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 33), 'True')
        # Assigning a type to the variable 'fileexists' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 20), 'fileexists', True_54881)
        # SSA join for if statement (line 468)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'fileexists' (line 477)
        fileexists_54882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 15), 'fileexists')
        # Testing the type of an if condition (line 477)
        if_condition_54883 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 477, 12), fileexists_54882)
        # Assigning a type to the variable 'if_condition_54883' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 12), 'if_condition_54883', if_condition_54883)
        # SSA begins for if statement (line 477)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 477)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to warn(...): (line 478)
        # Processing the call arguments (line 478)
        str_54886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 21), 'str', 'could not find library %r in directories %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 479)
        tuple_54887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 479)
        # Adding element type (line 479)
        # Getting the type of 'libname' (line 479)
        libname_54888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 24), 'libname', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 24), tuple_54887, libname_54888)
        # Adding element type (line 479)
        # Getting the type of 'c_library_dirs' (line 479)
        c_library_dirs_54889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 33), 'c_library_dirs', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 24), tuple_54887, c_library_dirs_54889)
        
        # Applying the binary operator '%' (line 478)
        result_mod_54890 = python_operator(stypy.reporting.localization.Localization(__file__, 478, 21), '%', str_54886, tuple_54887)
        
        # Processing the call keyword arguments (line 478)
        kwargs_54891 = {}
        # Getting the type of 'log' (line 478)
        log_54884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 12), 'log', False)
        # Obtaining the member 'warn' of a type (line 478)
        warn_54885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 12), log_54884, 'warn')
        # Calling warn(args, kwargs) (line 478)
        warn_call_result_54892 = invoke(stypy.reporting.localization.Localization(__file__, 478, 12), warn_54885, *[result_mod_54890], **kwargs_54891)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 482):
        
        # Assigning a List to a Name (line 482):
        
        # Obtaining an instance of the builtin type 'list' (line 482)
        list_54893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 482)
        
        # Assigning a type to the variable 'f_lib_dirs' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 8), 'f_lib_dirs', list_54893)
        
        # Getting the type of 'fcompiler' (line 483)
        fcompiler_54894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 19), 'fcompiler')
        # Obtaining the member 'library_dirs' of a type (line 483)
        library_dirs_54895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 19), fcompiler_54894, 'library_dirs')
        # Testing the type of a for loop iterable (line 483)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 483, 8), library_dirs_54895)
        # Getting the type of the for loop variable (line 483)
        for_loop_var_54896 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 483, 8), library_dirs_54895)
        # Assigning a type to the variable 'dir' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'dir', for_loop_var_54896)
        # SSA begins for a for statement (line 483)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to startswith(...): (line 486)
        # Processing the call arguments (line 486)
        str_54899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 30), 'str', '/usr/lib')
        # Processing the call keyword arguments (line 486)
        kwargs_54900 = {}
        # Getting the type of 'dir' (line 486)
        dir_54897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 15), 'dir', False)
        # Obtaining the member 'startswith' of a type (line 486)
        startswith_54898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 15), dir_54897, 'startswith')
        # Calling startswith(args, kwargs) (line 486)
        startswith_call_result_54901 = invoke(stypy.reporting.localization.Localization(__file__, 486, 15), startswith_54898, *[str_54899], **kwargs_54900)
        
        # Testing the type of an if condition (line 486)
        if_condition_54902 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 486, 12), startswith_call_result_54901)
        # Assigning a type to the variable 'if_condition_54902' (line 486)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 12), 'if_condition_54902', if_condition_54902)
        # SSA begins for if statement (line 486)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 487):
        
        # Assigning a Call to a Name:
        
        # Call to exec_command(...): (line 487)
        # Processing the call arguments (line 487)
        
        # Obtaining an instance of the builtin type 'list' (line 487)
        list_54904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 487)
        # Adding element type (line 487)
        str_54905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 37), 'str', 'cygpath')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 36), list_54904, str_54905)
        # Adding element type (line 487)
        str_54906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 48), 'str', '-w')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 36), list_54904, str_54906)
        # Adding element type (line 487)
        # Getting the type of 'dir' (line 487)
        dir_54907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 54), 'dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 36), list_54904, dir_54907)
        
        # Processing the call keyword arguments (line 487)
        # Getting the type of 'False' (line 487)
        False_54908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 68), 'False', False)
        keyword_54909 = False_54908
        kwargs_54910 = {'use_tee': keyword_54909}
        # Getting the type of 'exec_command' (line 487)
        exec_command_54903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 23), 'exec_command', False)
        # Calling exec_command(args, kwargs) (line 487)
        exec_command_call_result_54911 = invoke(stypy.reporting.localization.Localization(__file__, 487, 23), exec_command_54903, *[list_54904], **kwargs_54910)
        
        # Assigning a type to the variable 'call_assignment_53469' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 16), 'call_assignment_53469', exec_command_call_result_54911)
        
        # Assigning a Call to a Name (line 487):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_54914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 16), 'int')
        # Processing the call keyword arguments
        kwargs_54915 = {}
        # Getting the type of 'call_assignment_53469' (line 487)
        call_assignment_53469_54912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 16), 'call_assignment_53469', False)
        # Obtaining the member '__getitem__' of a type (line 487)
        getitem___54913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 16), call_assignment_53469_54912, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_54916 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___54913, *[int_54914], **kwargs_54915)
        
        # Assigning a type to the variable 'call_assignment_53470' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 16), 'call_assignment_53470', getitem___call_result_54916)
        
        # Assigning a Name to a Name (line 487):
        # Getting the type of 'call_assignment_53470' (line 487)
        call_assignment_53470_54917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 16), 'call_assignment_53470')
        # Assigning a type to the variable 's' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 16), 's', call_assignment_53470_54917)
        
        # Assigning a Call to a Name (line 487):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_54920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 16), 'int')
        # Processing the call keyword arguments
        kwargs_54921 = {}
        # Getting the type of 'call_assignment_53469' (line 487)
        call_assignment_53469_54918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 16), 'call_assignment_53469', False)
        # Obtaining the member '__getitem__' of a type (line 487)
        getitem___54919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 16), call_assignment_53469_54918, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_54922 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___54919, *[int_54920], **kwargs_54921)
        
        # Assigning a type to the variable 'call_assignment_53471' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 16), 'call_assignment_53471', getitem___call_result_54922)
        
        # Assigning a Name to a Name (line 487):
        # Getting the type of 'call_assignment_53471' (line 487)
        call_assignment_53471_54923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 16), 'call_assignment_53471')
        # Assigning a type to the variable 'o' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 19), 'o', call_assignment_53471_54923)
        
        
        # Getting the type of 's' (line 488)
        s_54924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 23), 's')
        # Applying the 'not' unary operator (line 488)
        result_not__54925 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 19), 'not', s_54924)
        
        # Testing the type of an if condition (line 488)
        if_condition_54926 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 488, 16), result_not__54925)
        # Assigning a type to the variable 'if_condition_54926' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 16), 'if_condition_54926', if_condition_54926)
        # SSA begins for if statement (line 488)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 489):
        
        # Assigning a Name to a Name (line 489):
        # Getting the type of 'o' (line 489)
        o_54927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 26), 'o')
        # Assigning a type to the variable 'dir' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 20), 'dir', o_54927)
        # SSA join for if statement (line 488)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 486)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 490)
        # Processing the call arguments (line 490)
        # Getting the type of 'dir' (line 490)
        dir_54930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 30), 'dir', False)
        # Processing the call keyword arguments (line 490)
        kwargs_54931 = {}
        # Getting the type of 'f_lib_dirs' (line 490)
        f_lib_dirs_54928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'f_lib_dirs', False)
        # Obtaining the member 'append' of a type (line 490)
        append_54929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 12), f_lib_dirs_54928, 'append')
        # Calling append(args, kwargs) (line 490)
        append_call_result_54932 = invoke(stypy.reporting.localization.Localization(__file__, 490, 12), append_54929, *[dir_54930], **kwargs_54931)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to extend(...): (line 491)
        # Processing the call arguments (line 491)
        # Getting the type of 'f_lib_dirs' (line 491)
        f_lib_dirs_54935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 30), 'f_lib_dirs', False)
        # Processing the call keyword arguments (line 491)
        kwargs_54936 = {}
        # Getting the type of 'c_library_dirs' (line 491)
        c_library_dirs_54933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'c_library_dirs', False)
        # Obtaining the member 'extend' of a type (line 491)
        extend_54934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 8), c_library_dirs_54933, 'extend')
        # Calling extend(args, kwargs) (line 491)
        extend_call_result_54937 = invoke(stypy.reporting.localization.Localization(__file__, 491, 8), extend_54934, *[f_lib_dirs_54935], **kwargs_54936)
        
        
        # Getting the type of 'fcompiler' (line 494)
        fcompiler_54938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 19), 'fcompiler')
        # Obtaining the member 'libraries' of a type (line 494)
        libraries_54939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 19), fcompiler_54938, 'libraries')
        # Testing the type of a for loop iterable (line 494)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 494, 8), libraries_54939)
        # Getting the type of the for loop variable (line 494)
        for_loop_var_54940 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 494, 8), libraries_54939)
        # Assigning a type to the variable 'lib' (line 494)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'lib', for_loop_var_54940)
        # SSA begins for a for statement (line 494)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Call to startswith(...): (line 495)
        # Processing the call arguments (line 495)
        str_54943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 34), 'str', 'msvc')
        # Processing the call keyword arguments (line 495)
        kwargs_54944 = {}
        # Getting the type of 'lib' (line 495)
        lib_54941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 19), 'lib', False)
        # Obtaining the member 'startswith' of a type (line 495)
        startswith_54942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 19), lib_54941, 'startswith')
        # Calling startswith(args, kwargs) (line 495)
        startswith_call_result_54945 = invoke(stypy.reporting.localization.Localization(__file__, 495, 19), startswith_54942, *[str_54943], **kwargs_54944)
        
        # Applying the 'not' unary operator (line 495)
        result_not__54946 = python_operator(stypy.reporting.localization.Localization(__file__, 495, 15), 'not', startswith_call_result_54945)
        
        # Testing the type of an if condition (line 495)
        if_condition_54947 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 495, 12), result_not__54946)
        # Assigning a type to the variable 'if_condition_54947' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 12), 'if_condition_54947', if_condition_54947)
        # SSA begins for if statement (line 495)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 496)
        # Processing the call arguments (line 496)
        # Getting the type of 'lib' (line 496)
        lib_54950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 35), 'lib', False)
        # Processing the call keyword arguments (line 496)
        kwargs_54951 = {}
        # Getting the type of 'c_libraries' (line 496)
        c_libraries_54948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 16), 'c_libraries', False)
        # Obtaining the member 'append' of a type (line 496)
        append_54949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 16), c_libraries_54948, 'append')
        # Calling append(args, kwargs) (line 496)
        append_call_result_54952 = invoke(stypy.reporting.localization.Localization(__file__, 496, 16), append_54949, *[lib_54950], **kwargs_54951)
        
        
        # Assigning a Call to a Name (line 497):
        
        # Assigning a Call to a Name (line 497):
        
        # Call to combine_paths(...): (line 497)
        # Processing the call arguments (line 497)
        # Getting the type of 'f_lib_dirs' (line 497)
        f_lib_dirs_54954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 34), 'f_lib_dirs', False)
        str_54955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 46), 'str', 'lib')
        # Getting the type of 'lib' (line 497)
        lib_54956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 54), 'lib', False)
        # Applying the binary operator '+' (line 497)
        result_add_54957 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 46), '+', str_54955, lib_54956)
        
        str_54958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 60), 'str', '.a')
        # Applying the binary operator '+' (line 497)
        result_add_54959 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 58), '+', result_add_54957, str_54958)
        
        # Processing the call keyword arguments (line 497)
        kwargs_54960 = {}
        # Getting the type of 'combine_paths' (line 497)
        combine_paths_54953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 20), 'combine_paths', False)
        # Calling combine_paths(args, kwargs) (line 497)
        combine_paths_call_result_54961 = invoke(stypy.reporting.localization.Localization(__file__, 497, 20), combine_paths_54953, *[f_lib_dirs_54954, result_add_54959], **kwargs_54960)
        
        # Assigning a type to the variable 'p' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 16), 'p', combine_paths_call_result_54961)
        
        # Getting the type of 'p' (line 498)
        p_54962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 19), 'p')
        # Testing the type of an if condition (line 498)
        if_condition_54963 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 498, 16), p_54962)
        # Assigning a type to the variable 'if_condition_54963' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 16), 'if_condition_54963', if_condition_54963)
        # SSA begins for if statement (line 498)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 499):
        
        # Assigning a Call to a Name (line 499):
        
        # Call to join(...): (line 499)
        # Processing the call arguments (line 499)
        # Getting the type of 'self' (line 499)
        self_54967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 44), 'self', False)
        # Obtaining the member 'build_temp' of a type (line 499)
        build_temp_54968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 44), self_54967, 'build_temp')
        # Getting the type of 'lib' (line 499)
        lib_54969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 61), 'lib', False)
        str_54970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 67), 'str', '.lib')
        # Applying the binary operator '+' (line 499)
        result_add_54971 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 61), '+', lib_54969, str_54970)
        
        # Processing the call keyword arguments (line 499)
        kwargs_54972 = {}
        # Getting the type of 'os' (line 499)
        os_54964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 31), 'os', False)
        # Obtaining the member 'path' of a type (line 499)
        path_54965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 31), os_54964, 'path')
        # Obtaining the member 'join' of a type (line 499)
        join_54966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 31), path_54965, 'join')
        # Calling join(args, kwargs) (line 499)
        join_call_result_54973 = invoke(stypy.reporting.localization.Localization(__file__, 499, 31), join_54966, *[build_temp_54968, result_add_54971], **kwargs_54972)
        
        # Assigning a type to the variable 'dst_name' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 20), 'dst_name', join_call_result_54973)
        
        
        
        # Call to isfile(...): (line 500)
        # Processing the call arguments (line 500)
        # Getting the type of 'dst_name' (line 500)
        dst_name_54977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 42), 'dst_name', False)
        # Processing the call keyword arguments (line 500)
        kwargs_54978 = {}
        # Getting the type of 'os' (line 500)
        os_54974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 500)
        path_54975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 27), os_54974, 'path')
        # Obtaining the member 'isfile' of a type (line 500)
        isfile_54976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 27), path_54975, 'isfile')
        # Calling isfile(args, kwargs) (line 500)
        isfile_call_result_54979 = invoke(stypy.reporting.localization.Localization(__file__, 500, 27), isfile_54976, *[dst_name_54977], **kwargs_54978)
        
        # Applying the 'not' unary operator (line 500)
        result_not__54980 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 23), 'not', isfile_call_result_54979)
        
        # Testing the type of an if condition (line 500)
        if_condition_54981 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 500, 20), result_not__54980)
        # Assigning a type to the variable 'if_condition_54981' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 20), 'if_condition_54981', if_condition_54981)
        # SSA begins for if statement (line 500)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to copy_file(...): (line 501)
        # Processing the call arguments (line 501)
        
        # Obtaining the type of the subscript
        int_54983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 36), 'int')
        # Getting the type of 'p' (line 501)
        p_54984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 34), 'p', False)
        # Obtaining the member '__getitem__' of a type (line 501)
        getitem___54985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 34), p_54984, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 501)
        subscript_call_result_54986 = invoke(stypy.reporting.localization.Localization(__file__, 501, 34), getitem___54985, int_54983)
        
        # Getting the type of 'dst_name' (line 501)
        dst_name_54987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 40), 'dst_name', False)
        # Processing the call keyword arguments (line 501)
        kwargs_54988 = {}
        # Getting the type of 'copy_file' (line 501)
        copy_file_54982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 24), 'copy_file', False)
        # Calling copy_file(args, kwargs) (line 501)
        copy_file_call_result_54989 = invoke(stypy.reporting.localization.Localization(__file__, 501, 24), copy_file_54982, *[subscript_call_result_54986, dst_name_54987], **kwargs_54988)
        
        # SSA join for if statement (line 500)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 502)
        self_54990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 23), 'self')
        # Obtaining the member 'build_temp' of a type (line 502)
        build_temp_54991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 23), self_54990, 'build_temp')
        # Getting the type of 'c_library_dirs' (line 502)
        c_library_dirs_54992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 46), 'c_library_dirs')
        # Applying the binary operator 'notin' (line 502)
        result_contains_54993 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 23), 'notin', build_temp_54991, c_library_dirs_54992)
        
        # Testing the type of an if condition (line 502)
        if_condition_54994 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 502, 20), result_contains_54993)
        # Assigning a type to the variable 'if_condition_54994' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 20), 'if_condition_54994', if_condition_54994)
        # SSA begins for if statement (line 502)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 503)
        # Processing the call arguments (line 503)
        # Getting the type of 'self' (line 503)
        self_54997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 46), 'self', False)
        # Obtaining the member 'build_temp' of a type (line 503)
        build_temp_54998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 46), self_54997, 'build_temp')
        # Processing the call keyword arguments (line 503)
        kwargs_54999 = {}
        # Getting the type of 'c_library_dirs' (line 503)
        c_library_dirs_54995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 24), 'c_library_dirs', False)
        # Obtaining the member 'append' of a type (line 503)
        append_54996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 24), c_library_dirs_54995, 'append')
        # Calling append(args, kwargs) (line 503)
        append_call_result_55000 = invoke(stypy.reporting.localization.Localization(__file__, 503, 24), append_54996, *[build_temp_54998], **kwargs_54999)
        
        # SSA join for if statement (line 502)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 498)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 495)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_libs_with_msvc_and_fortran(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_libs_with_msvc_and_fortran' in the type store
        # Getting the type of 'stypy_return_type' (line 451)
        stypy_return_type_55001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_55001)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_libs_with_msvc_and_fortran'
        return stypy_return_type_55001


    @norecursion
    def get_source_files(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_source_files'
        module_type_store = module_type_store.open_function_context('get_source_files', 505, 4, False)
        # Assigning a type to the variable 'self' (line 506)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext.get_source_files.__dict__.__setitem__('stypy_localization', localization)
        build_ext.get_source_files.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext.get_source_files.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext.get_source_files.__dict__.__setitem__('stypy_function_name', 'build_ext.get_source_files')
        build_ext.get_source_files.__dict__.__setitem__('stypy_param_names_list', [])
        build_ext.get_source_files.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext.get_source_files.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext.get_source_files.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext.get_source_files.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext.get_source_files.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext.get_source_files.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.get_source_files', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_source_files', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_source_files(...)' code ##################

        
        # Call to check_extensions_list(...): (line 506)
        # Processing the call arguments (line 506)
        # Getting the type of 'self' (line 506)
        self_55004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 35), 'self', False)
        # Obtaining the member 'extensions' of a type (line 506)
        extensions_55005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 35), self_55004, 'extensions')
        # Processing the call keyword arguments (line 506)
        kwargs_55006 = {}
        # Getting the type of 'self' (line 506)
        self_55002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'self', False)
        # Obtaining the member 'check_extensions_list' of a type (line 506)
        check_extensions_list_55003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 8), self_55002, 'check_extensions_list')
        # Calling check_extensions_list(args, kwargs) (line 506)
        check_extensions_list_call_result_55007 = invoke(stypy.reporting.localization.Localization(__file__, 506, 8), check_extensions_list_55003, *[extensions_55005], **kwargs_55006)
        
        
        # Assigning a List to a Name (line 507):
        
        # Assigning a List to a Name (line 507):
        
        # Obtaining an instance of the builtin type 'list' (line 507)
        list_55008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 507)
        
        # Assigning a type to the variable 'filenames' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'filenames', list_55008)
        
        # Getting the type of 'self' (line 508)
        self_55009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 19), 'self')
        # Obtaining the member 'extensions' of a type (line 508)
        extensions_55010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 19), self_55009, 'extensions')
        # Testing the type of a for loop iterable (line 508)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 508, 8), extensions_55010)
        # Getting the type of the for loop variable (line 508)
        for_loop_var_55011 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 508, 8), extensions_55010)
        # Assigning a type to the variable 'ext' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'ext', for_loop_var_55011)
        # SSA begins for a for statement (line 508)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to extend(...): (line 509)
        # Processing the call arguments (line 509)
        
        # Call to get_ext_source_files(...): (line 509)
        # Processing the call arguments (line 509)
        # Getting the type of 'ext' (line 509)
        ext_55015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 50), 'ext', False)
        # Processing the call keyword arguments (line 509)
        kwargs_55016 = {}
        # Getting the type of 'get_ext_source_files' (line 509)
        get_ext_source_files_55014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 29), 'get_ext_source_files', False)
        # Calling get_ext_source_files(args, kwargs) (line 509)
        get_ext_source_files_call_result_55017 = invoke(stypy.reporting.localization.Localization(__file__, 509, 29), get_ext_source_files_55014, *[ext_55015], **kwargs_55016)
        
        # Processing the call keyword arguments (line 509)
        kwargs_55018 = {}
        # Getting the type of 'filenames' (line 509)
        filenames_55012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 12), 'filenames', False)
        # Obtaining the member 'extend' of a type (line 509)
        extend_55013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 12), filenames_55012, 'extend')
        # Calling extend(args, kwargs) (line 509)
        extend_call_result_55019 = invoke(stypy.reporting.localization.Localization(__file__, 509, 12), extend_55013, *[get_ext_source_files_call_result_55017], **kwargs_55018)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'filenames' (line 510)
        filenames_55020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 15), 'filenames')
        # Assigning a type to the variable 'stypy_return_type' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'stypy_return_type', filenames_55020)
        
        # ################# End of 'get_source_files(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_source_files' in the type store
        # Getting the type of 'stypy_return_type' (line 505)
        stypy_return_type_55021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_55021)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_source_files'
        return stypy_return_type_55021


    @norecursion
    def get_outputs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_outputs'
        module_type_store = module_type_store.open_function_context('get_outputs', 512, 4, False)
        # Assigning a type to the variable 'self' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext.get_outputs.__dict__.__setitem__('stypy_localization', localization)
        build_ext.get_outputs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext.get_outputs.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext.get_outputs.__dict__.__setitem__('stypy_function_name', 'build_ext.get_outputs')
        build_ext.get_outputs.__dict__.__setitem__('stypy_param_names_list', [])
        build_ext.get_outputs.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext.get_outputs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext.get_outputs.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext.get_outputs.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext.get_outputs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext.get_outputs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.get_outputs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_outputs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_outputs(...)' code ##################

        
        # Call to check_extensions_list(...): (line 513)
        # Processing the call arguments (line 513)
        # Getting the type of 'self' (line 513)
        self_55024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 35), 'self', False)
        # Obtaining the member 'extensions' of a type (line 513)
        extensions_55025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 35), self_55024, 'extensions')
        # Processing the call keyword arguments (line 513)
        kwargs_55026 = {}
        # Getting the type of 'self' (line 513)
        self_55022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'self', False)
        # Obtaining the member 'check_extensions_list' of a type (line 513)
        check_extensions_list_55023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 8), self_55022, 'check_extensions_list')
        # Calling check_extensions_list(args, kwargs) (line 513)
        check_extensions_list_call_result_55027 = invoke(stypy.reporting.localization.Localization(__file__, 513, 8), check_extensions_list_55023, *[extensions_55025], **kwargs_55026)
        
        
        # Assigning a List to a Name (line 515):
        
        # Assigning a List to a Name (line 515):
        
        # Obtaining an instance of the builtin type 'list' (line 515)
        list_55028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 515)
        
        # Assigning a type to the variable 'outputs' (line 515)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 8), 'outputs', list_55028)
        
        # Getting the type of 'self' (line 516)
        self_55029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 19), 'self')
        # Obtaining the member 'extensions' of a type (line 516)
        extensions_55030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 19), self_55029, 'extensions')
        # Testing the type of a for loop iterable (line 516)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 516, 8), extensions_55030)
        # Getting the type of the for loop variable (line 516)
        for_loop_var_55031 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 516, 8), extensions_55030)
        # Assigning a type to the variable 'ext' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'ext', for_loop_var_55031)
        # SSA begins for a for statement (line 516)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'ext' (line 517)
        ext_55032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 19), 'ext')
        # Obtaining the member 'sources' of a type (line 517)
        sources_55033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 19), ext_55032, 'sources')
        # Applying the 'not' unary operator (line 517)
        result_not__55034 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 15), 'not', sources_55033)
        
        # Testing the type of an if condition (line 517)
        if_condition_55035 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 517, 12), result_not__55034)
        # Assigning a type to the variable 'if_condition_55035' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 12), 'if_condition_55035', if_condition_55035)
        # SSA begins for if statement (line 517)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 517)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 519):
        
        # Assigning a Call to a Name (line 519):
        
        # Call to get_ext_fullname(...): (line 519)
        # Processing the call arguments (line 519)
        # Getting the type of 'ext' (line 519)
        ext_55038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 45), 'ext', False)
        # Obtaining the member 'name' of a type (line 519)
        name_55039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 45), ext_55038, 'name')
        # Processing the call keyword arguments (line 519)
        kwargs_55040 = {}
        # Getting the type of 'self' (line 519)
        self_55036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 23), 'self', False)
        # Obtaining the member 'get_ext_fullname' of a type (line 519)
        get_ext_fullname_55037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 23), self_55036, 'get_ext_fullname')
        # Calling get_ext_fullname(args, kwargs) (line 519)
        get_ext_fullname_call_result_55041 = invoke(stypy.reporting.localization.Localization(__file__, 519, 23), get_ext_fullname_55037, *[name_55039], **kwargs_55040)
        
        # Assigning a type to the variable 'fullname' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 12), 'fullname', get_ext_fullname_call_result_55041)
        
        # Call to append(...): (line 520)
        # Processing the call arguments (line 520)
        
        # Call to join(...): (line 520)
        # Processing the call arguments (line 520)
        # Getting the type of 'self' (line 520)
        self_55047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 40), 'self', False)
        # Obtaining the member 'build_lib' of a type (line 520)
        build_lib_55048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 40), self_55047, 'build_lib')
        
        # Call to get_ext_filename(...): (line 521)
        # Processing the call arguments (line 521)
        # Getting the type of 'fullname' (line 521)
        fullname_55051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 62), 'fullname', False)
        # Processing the call keyword arguments (line 521)
        kwargs_55052 = {}
        # Getting the type of 'self' (line 521)
        self_55049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 40), 'self', False)
        # Obtaining the member 'get_ext_filename' of a type (line 521)
        get_ext_filename_55050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 40), self_55049, 'get_ext_filename')
        # Calling get_ext_filename(args, kwargs) (line 521)
        get_ext_filename_call_result_55053 = invoke(stypy.reporting.localization.Localization(__file__, 521, 40), get_ext_filename_55050, *[fullname_55051], **kwargs_55052)
        
        # Processing the call keyword arguments (line 520)
        kwargs_55054 = {}
        # Getting the type of 'os' (line 520)
        os_55044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 520)
        path_55045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 27), os_55044, 'path')
        # Obtaining the member 'join' of a type (line 520)
        join_55046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 27), path_55045, 'join')
        # Calling join(args, kwargs) (line 520)
        join_call_result_55055 = invoke(stypy.reporting.localization.Localization(__file__, 520, 27), join_55046, *[build_lib_55048, get_ext_filename_call_result_55053], **kwargs_55054)
        
        # Processing the call keyword arguments (line 520)
        kwargs_55056 = {}
        # Getting the type of 'outputs' (line 520)
        outputs_55042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'outputs', False)
        # Obtaining the member 'append' of a type (line 520)
        append_55043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 12), outputs_55042, 'append')
        # Calling append(args, kwargs) (line 520)
        append_call_result_55057 = invoke(stypy.reporting.localization.Localization(__file__, 520, 12), append_55043, *[join_call_result_55055], **kwargs_55056)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'outputs' (line 522)
        outputs_55058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 15), 'outputs')
        # Assigning a type to the variable 'stypy_return_type' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'stypy_return_type', outputs_55058)
        
        # ################# End of 'get_outputs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_outputs' in the type store
        # Getting the type of 'stypy_return_type' (line 512)
        stypy_return_type_55059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_55059)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_outputs'
        return stypy_return_type_55059


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 30, 0, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'build_ext' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'build_ext', build_ext)

# Assigning a Str to a Name (line 32):
str_55060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 18), 'str', 'build C/C++/F extensions (compile/link to build directory)')
# Getting the type of 'build_ext'
build_ext_55061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_ext')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_ext_55061, 'description', str_55060)

# Assigning a BinOp to a Name (line 34):
# Getting the type of 'old_build_ext' (line 34)
old_build_ext_55062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'old_build_ext')
# Obtaining the member 'user_options' of a type (line 34)
user_options_55063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 19), old_build_ext_55062, 'user_options')

# Obtaining an instance of the builtin type 'list' (line 34)
list_55064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 48), 'list')
# Adding type elements to the builtin type 'list' instance (line 34)
# Adding element type (line 34)

# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_55065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)
str_55066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 9), 'str', 'fcompiler=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 9), tuple_55065, str_55066)
# Adding element type (line 35)
# Getting the type of 'None' (line 35)
None_55067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 23), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 9), tuple_55065, None_55067)
# Adding element type (line 35)
str_55068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 9), 'str', 'specify the Fortran compiler type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 9), tuple_55065, str_55068)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 48), list_55064, tuple_55065)
# Adding element type (line 34)

# Obtaining an instance of the builtin type 'tuple' (line 37)
tuple_55069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 37)
# Adding element type (line 37)
str_55070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 9), 'str', 'parallel=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 9), tuple_55069, str_55070)
# Adding element type (line 37)
str_55071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 22), 'str', 'j')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 9), tuple_55069, str_55071)
# Adding element type (line 37)
str_55072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 9), 'str', 'number of parallel jobs')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 9), tuple_55069, str_55072)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 48), list_55064, tuple_55069)

# Applying the binary operator '+' (line 34)
result_add_55073 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 19), '+', user_options_55063, list_55064)

# Getting the type of 'build_ext'
build_ext_55074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_ext')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_ext_55074, 'user_options', result_add_55073)

# Assigning a BinOp to a Name (line 41):
# Getting the type of 'old_build_ext' (line 41)
old_build_ext_55075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 19), 'old_build_ext')
# Obtaining the member 'help_options' of a type (line 41)
help_options_55076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 19), old_build_ext_55075, 'help_options')

# Obtaining an instance of the builtin type 'list' (line 41)
list_55077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 48), 'list')
# Adding type elements to the builtin type 'list' instance (line 41)
# Adding element type (line 41)

# Obtaining an instance of the builtin type 'tuple' (line 42)
tuple_55078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 42)
# Adding element type (line 42)
str_55079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 9), 'str', 'help-fcompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 9), tuple_55078, str_55079)
# Adding element type (line 42)
# Getting the type of 'None' (line 42)
None_55080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 27), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 9), tuple_55078, None_55080)
# Adding element type (line 42)
str_55081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 33), 'str', 'list available Fortran compilers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 9), tuple_55078, str_55081)
# Adding element type (line 42)
# Getting the type of 'show_fortran_compilers' (line 43)
show_fortran_compilers_55082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 9), 'show_fortran_compilers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 9), tuple_55078, show_fortran_compilers_55082)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 48), list_55077, tuple_55078)

# Applying the binary operator '+' (line 41)
result_add_55083 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 19), '+', help_options_55076, list_55077)

# Getting the type of 'build_ext'
build_ext_55084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_ext')
# Setting the type of the member 'help_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_ext_55084, 'help_options', result_add_55083)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import os
4: import re
5: import sys
6: import types
7: from copy import copy
8: from distutils import ccompiler
9: from distutils.ccompiler import *
10: from distutils.errors import DistutilsExecError, DistutilsModuleError, \
11:                              DistutilsPlatformError
12: from distutils.sysconfig import customize_compiler
13: from distutils.version import LooseVersion
14: 
15: from numpy.distutils import log
16: from numpy.distutils.compat import get_exception
17: from numpy.distutils.exec_command import exec_command
18: from numpy.distutils.misc_util import cyg2win32, is_sequence, mingw32, \
19:                                       quote_args, get_num_build_jobs
20: 
21: 
22: def replace_method(klass, method_name, func):
23:     if sys.version_info[0] < 3:
24:         m = types.MethodType(func, None, klass)
25:     else:
26:         # Py3k does not have unbound method anymore, MethodType does not work
27:         m = lambda self, *args, **kw: func(self, *args, **kw)
28:     setattr(klass, method_name, m)
29: 
30: # Using customized CCompiler.spawn.
31: def CCompiler_spawn(self, cmd, display=None):
32:     '''
33:     Execute a command in a sub-process.
34: 
35:     Parameters
36:     ----------
37:     cmd : str
38:         The command to execute.
39:     display : str or sequence of str, optional
40:         The text to add to the log file kept by `numpy.distutils`.
41:         If not given, `display` is equal to `cmd`.
42: 
43:     Returns
44:     -------
45:     None
46: 
47:     Raises
48:     ------
49:     DistutilsExecError
50:         If the command failed, i.e. the exit status was not 0.
51: 
52:     '''
53:     if display is None:
54:         display = cmd
55:         if is_sequence(display):
56:             display = ' '.join(list(display))
57:     log.info(display)
58:     s, o = exec_command(cmd)
59:     if s:
60:         if is_sequence(cmd):
61:             cmd = ' '.join(list(cmd))
62:         try:
63:             print(o)
64:         except UnicodeError:
65:             # When installing through pip, `o` can contain non-ascii chars
66:             pass
67:         if re.search('Too many open files', o):
68:             msg = '\nTry rerunning setup command until build succeeds.'
69:         else:
70:             msg = ''
71:         raise DistutilsExecError('Command "%s" failed with exit status %d%s' % (cmd, s, msg))
72: 
73: replace_method(CCompiler, 'spawn', CCompiler_spawn)
74: 
75: def CCompiler_object_filenames(self, source_filenames, strip_dir=0, output_dir=''):
76:     '''
77:     Return the name of the object files for the given source files.
78: 
79:     Parameters
80:     ----------
81:     source_filenames : list of str
82:         The list of paths to source files. Paths can be either relative or
83:         absolute, this is handled transparently.
84:     strip_dir : bool, optional
85:         Whether to strip the directory from the returned paths. If True,
86:         the file name prepended by `output_dir` is returned. Default is False.
87:     output_dir : str, optional
88:         If given, this path is prepended to the returned paths to the
89:         object files.
90: 
91:     Returns
92:     -------
93:     obj_names : list of str
94:         The list of paths to the object files corresponding to the source
95:         files in `source_filenames`.
96: 
97:     '''
98:     if output_dir is None:
99:         output_dir = ''
100:     obj_names = []
101:     for src_name in source_filenames:
102:         base, ext = os.path.splitext(os.path.normpath(src_name))
103:         base = os.path.splitdrive(base)[1] # Chop off the drive
104:         base = base[os.path.isabs(base):]  # If abs, chop off leading /
105:         if base.startswith('..'):
106:             # Resolve starting relative path components, middle ones
107:             # (if any) have been handled by os.path.normpath above.
108:             i = base.rfind('..')+2
109:             d = base[:i]
110:             d = os.path.basename(os.path.abspath(d))
111:             base = d + base[i:]
112:         if ext not in self.src_extensions:
113:             raise UnknownFileError("unknown file type '%s' (from '%s')" % (ext, src_name))
114:         if strip_dir:
115:             base = os.path.basename(base)
116:         obj_name = os.path.join(output_dir, base + self.obj_extension)
117:         obj_names.append(obj_name)
118:     return obj_names
119: 
120: replace_method(CCompiler, 'object_filenames', CCompiler_object_filenames)
121: 
122: def CCompiler_compile(self, sources, output_dir=None, macros=None,
123:                       include_dirs=None, debug=0, extra_preargs=None,
124:                       extra_postargs=None, depends=None):
125:     '''
126:     Compile one or more source files.
127: 
128:     Please refer to the Python distutils API reference for more details.
129: 
130:     Parameters
131:     ----------
132:     sources : list of str
133:         A list of filenames
134:     output_dir : str, optional
135:         Path to the output directory.
136:     macros : list of tuples
137:         A list of macro definitions.
138:     include_dirs : list of str, optional
139:         The directories to add to the default include file search path for
140:         this compilation only.
141:     debug : bool, optional
142:         Whether or not to output debug symbols in or alongside the object
143:         file(s).
144:     extra_preargs, extra_postargs : ?
145:         Extra pre- and post-arguments.
146:     depends : list of str, optional
147:         A list of file names that all targets depend on.
148: 
149:     Returns
150:     -------
151:     objects : list of str
152:         A list of object file names, one per source file `sources`.
153: 
154:     Raises
155:     ------
156:     CompileError
157:         If compilation fails.
158: 
159:     '''
160:     # This method is effective only with Python >=2.3 distutils.
161:     # Any changes here should be applied also to fcompiler.compile
162:     # method to support pre Python 2.3 distutils.
163:     if not sources:
164:         return []
165:     # FIXME:RELATIVE_IMPORT
166:     if sys.version_info[0] < 3:
167:         from .fcompiler import FCompiler, is_f_file, has_f90_header
168:     else:
169:         from numpy.distutils.fcompiler import (FCompiler, is_f_file,
170:                                                has_f90_header)
171:     if isinstance(self, FCompiler):
172:         display = []
173:         for fc in ['f77', 'f90', 'fix']:
174:             fcomp = getattr(self, 'compiler_'+fc)
175:             if fcomp is None:
176:                 continue
177:             display.append("Fortran %s compiler: %s" % (fc, ' '.join(fcomp)))
178:         display = '\n'.join(display)
179:     else:
180:         ccomp = self.compiler_so
181:         display = "C compiler: %s\n" % (' '.join(ccomp),)
182:     log.info(display)
183:     macros, objects, extra_postargs, pp_opts, build = \
184:             self._setup_compile(output_dir, macros, include_dirs, sources,
185:                                 depends, extra_postargs)
186:     cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
187:     display = "compile options: '%s'" % (' '.join(cc_args))
188:     if extra_postargs:
189:         display += "\nextra options: '%s'" % (' '.join(extra_postargs))
190:     log.info(display)
191: 
192:     def single_compile(args):
193:         obj, (src, ext) = args
194:         self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
195: 
196:     if isinstance(self, FCompiler):
197:         objects_to_build = list(build.keys())
198:         f77_objects, other_objects = [], []
199:         for obj in objects:
200:             if obj in objects_to_build:
201:                 src, ext = build[obj]
202:                 if self.compiler_type=='absoft':
203:                     obj = cyg2win32(obj)
204:                     src = cyg2win32(src)
205:                 if is_f_file(src) and not has_f90_header(src):
206:                     f77_objects.append((obj, (src, ext)))
207:                 else:
208:                     other_objects.append((obj, (src, ext)))
209: 
210:         # f77 objects can be built in parallel
211:         build_items = f77_objects
212:         # build f90 modules serial, module files are generated during
213:         # compilation and may be used by files later in the list so the
214:         # ordering is important
215:         for o in other_objects:
216:             single_compile(o)
217:     else:
218:         build_items = build.items()
219: 
220:     jobs = get_num_build_jobs()
221:     if len(build) > 1 and jobs > 1:
222:         # build parallel
223:         import multiprocessing.pool
224:         pool = multiprocessing.pool.ThreadPool(jobs)
225:         pool.map(single_compile, build_items)
226:         pool.close()
227:     else:
228:         # build serial
229:         for o in build_items:
230:             single_compile(o)
231: 
232:     # Return *all* object filenames, not just the ones we just built.
233:     return objects
234: 
235: replace_method(CCompiler, 'compile', CCompiler_compile)
236: 
237: def CCompiler_customize_cmd(self, cmd, ignore=()):
238:     '''
239:     Customize compiler using distutils command.
240: 
241:     Parameters
242:     ----------
243:     cmd : class instance
244:         An instance inheriting from `distutils.cmd.Command`.
245:     ignore : sequence of str, optional
246:         List of `CCompiler` commands (without ``'set_'``) that should not be
247:         altered. Strings that are checked for are:
248:         ``('include_dirs', 'define', 'undef', 'libraries', 'library_dirs',
249:         'rpath', 'link_objects')``.
250: 
251:     Returns
252:     -------
253:     None
254: 
255:     '''
256:     log.info('customize %s using %s' % (self.__class__.__name__,
257:                                         cmd.__class__.__name__))
258:     def allow(attr):
259:         return getattr(cmd, attr, None) is not None and attr not in ignore
260: 
261:     if allow('include_dirs'):
262:         self.set_include_dirs(cmd.include_dirs)
263:     if allow('define'):
264:         for (name, value) in cmd.define:
265:             self.define_macro(name, value)
266:     if allow('undef'):
267:         for macro in cmd.undef:
268:             self.undefine_macro(macro)
269:     if allow('libraries'):
270:         self.set_libraries(self.libraries + cmd.libraries)
271:     if allow('library_dirs'):
272:         self.set_library_dirs(self.library_dirs + cmd.library_dirs)
273:     if allow('rpath'):
274:         self.set_runtime_library_dirs(cmd.rpath)
275:     if allow('link_objects'):
276:         self.set_link_objects(cmd.link_objects)
277: 
278: replace_method(CCompiler, 'customize_cmd', CCompiler_customize_cmd)
279: 
280: def _compiler_to_string(compiler):
281:     props = []
282:     mx = 0
283:     keys = list(compiler.executables.keys())
284:     for key in ['version', 'libraries', 'library_dirs',
285:                 'object_switch', 'compile_switch',
286:                 'include_dirs', 'define', 'undef', 'rpath', 'link_objects']:
287:         if key not in keys:
288:             keys.append(key)
289:     for key in keys:
290:         if hasattr(compiler, key):
291:             v = getattr(compiler, key)
292:             mx = max(mx, len(key))
293:             props.append((key, repr(v)))
294:     lines = []
295:     format = '%-' + repr(mx+1) + 's = %s'
296:     for prop in props:
297:         lines.append(format % prop)
298:     return '\n'.join(lines)
299: 
300: def CCompiler_show_customization(self):
301:     '''
302:     Print the compiler customizations to stdout.
303: 
304:     Parameters
305:     ----------
306:     None
307: 
308:     Returns
309:     -------
310:     None
311: 
312:     Notes
313:     -----
314:     Printing is only done if the distutils log threshold is < 2.
315: 
316:     '''
317:     if 0:
318:         for attrname in ['include_dirs', 'define', 'undef',
319:                          'libraries', 'library_dirs',
320:                          'rpath', 'link_objects']:
321:             attr = getattr(self, attrname, None)
322:             if not attr:
323:                 continue
324:             log.info("compiler '%s' is set to %s" % (attrname, attr))
325:     try:
326:         self.get_version()
327:     except:
328:         pass
329:     if log._global_log.threshold<2:
330:         print('*'*80)
331:         print(self.__class__)
332:         print(_compiler_to_string(self))
333:         print('*'*80)
334: 
335: replace_method(CCompiler, 'show_customization', CCompiler_show_customization)
336: 
337: def CCompiler_customize(self, dist, need_cxx=0):
338:     '''
339:     Do any platform-specific customization of a compiler instance.
340: 
341:     This method calls `distutils.sysconfig.customize_compiler` for
342:     platform-specific customization, as well as optionally remove a flag
343:     to suppress spurious warnings in case C++ code is being compiled.
344: 
345:     Parameters
346:     ----------
347:     dist : object
348:         This parameter is not used for anything.
349:     need_cxx : bool, optional
350:         Whether or not C++ has to be compiled. If so (True), the
351:         ``"-Wstrict-prototypes"`` option is removed to prevent spurious
352:         warnings. Default is False.
353: 
354:     Returns
355:     -------
356:     None
357: 
358:     Notes
359:     -----
360:     All the default options used by distutils can be extracted with::
361: 
362:       from distutils import sysconfig
363:       sysconfig.get_config_vars('CC', 'CXX', 'OPT', 'BASECFLAGS',
364:                                 'CCSHARED', 'LDSHARED', 'SO')
365: 
366:     '''
367:     # See FCompiler.customize for suggested usage.
368:     log.info('customize %s' % (self.__class__.__name__))
369:     customize_compiler(self)
370:     if need_cxx:
371:         # In general, distutils uses -Wstrict-prototypes, but this option is
372:         # not valid for C++ code, only for C.  Remove it if it's there to
373:         # avoid a spurious warning on every compilation.
374:         try:
375:             self.compiler_so.remove('-Wstrict-prototypes')
376:         except (AttributeError, ValueError):
377:             pass
378: 
379:         if hasattr(self, 'compiler') and 'cc' in self.compiler[0]:
380:             if not self.compiler_cxx:
381:                 if self.compiler[0].startswith('gcc'):
382:                     a, b = 'gcc', 'g++'
383:                 else:
384:                     a, b = 'cc', 'c++'
385:                 self.compiler_cxx = [self.compiler[0].replace(a, b)]\
386:                                     + self.compiler[1:]
387:         else:
388:             if hasattr(self, 'compiler'):
389:                 log.warn("#### %s #######" % (self.compiler,))
390:             if not hasattr(self, 'compiler_cxx'):
391:                 log.warn('Missing compiler_cxx fix for ' + self.__class__.__name__)
392:     return
393: 
394: replace_method(CCompiler, 'customize', CCompiler_customize)
395: 
396: def simple_version_match(pat=r'[-.\d]+', ignore='', start=''):
397:     '''
398:     Simple matching of version numbers, for use in CCompiler and FCompiler.
399: 
400:     Parameters
401:     ----------
402:     pat : str, optional
403:         A regular expression matching version numbers.
404:         Default is ``r'[-.\\d]+'``.
405:     ignore : str, optional
406:         A regular expression matching patterns to skip.
407:         Default is ``''``, in which case nothing is skipped.
408:     start : str, optional
409:         A regular expression matching the start of where to start looking
410:         for version numbers.
411:         Default is ``''``, in which case searching is started at the
412:         beginning of the version string given to `matcher`.
413: 
414:     Returns
415:     -------
416:     matcher : callable
417:         A function that is appropriate to use as the ``.version_match``
418:         attribute of a `CCompiler` class. `matcher` takes a single parameter,
419:         a version string.
420: 
421:     '''
422:     def matcher(self, version_string):
423:         # version string may appear in the second line, so getting rid
424:         # of new lines:
425:         version_string = version_string.replace('\n', ' ')
426:         pos = 0
427:         if start:
428:             m = re.match(start, version_string)
429:             if not m:
430:                 return None
431:             pos = m.end()
432:         while True:
433:             m = re.search(pat, version_string[pos:])
434:             if not m:
435:                 return None
436:             if ignore and re.match(ignore, m.group(0)):
437:                 pos = m.end()
438:                 continue
439:             break
440:         return m.group(0)
441:     return matcher
442: 
443: def CCompiler_get_version(self, force=False, ok_status=[0]):
444:     '''
445:     Return compiler version, or None if compiler is not available.
446: 
447:     Parameters
448:     ----------
449:     force : bool, optional
450:         If True, force a new determination of the version, even if the
451:         compiler already has a version attribute. Default is False.
452:     ok_status : list of int, optional
453:         The list of status values returned by the version look-up process
454:         for which a version string is returned. If the status value is not
455:         in `ok_status`, None is returned. Default is ``[0]``.
456: 
457:     Returns
458:     -------
459:     version : str or None
460:         Version string, in the format of `distutils.version.LooseVersion`.
461: 
462:     '''
463:     if not force and hasattr(self, 'version'):
464:         return self.version
465:     self.find_executables()
466:     try:
467:         version_cmd = self.version_cmd
468:     except AttributeError:
469:         return None
470:     if not version_cmd or not version_cmd[0]:
471:         return None
472:     try:
473:         matcher = self.version_match
474:     except AttributeError:
475:         try:
476:             pat = self.version_pattern
477:         except AttributeError:
478:             return None
479:         def matcher(version_string):
480:             m = re.match(pat, version_string)
481:             if not m:
482:                 return None
483:             version = m.group('version')
484:             return version
485: 
486:     status, output = exec_command(version_cmd, use_tee=0)
487: 
488:     version = None
489:     if status in ok_status:
490:         version = matcher(output)
491:         if version:
492:             version = LooseVersion(version)
493:     self.version = version
494:     return version
495: 
496: replace_method(CCompiler, 'get_version', CCompiler_get_version)
497: 
498: def CCompiler_cxx_compiler(self):
499:     '''
500:     Return the C++ compiler.
501: 
502:     Parameters
503:     ----------
504:     None
505: 
506:     Returns
507:     -------
508:     cxx : class instance
509:         The C++ compiler, as a `CCompiler` instance.
510: 
511:     '''
512:     if self.compiler_type in ('msvc', 'intelw', 'intelemw'):
513:         return self
514: 
515:     cxx = copy(self)
516:     cxx.compiler_so = [cxx.compiler_cxx[0]] + cxx.compiler_so[1:]
517:     if sys.platform.startswith('aix') and 'ld_so_aix' in cxx.linker_so[0]:
518:         # AIX needs the ld_so_aix script included with Python
519:         cxx.linker_so = [cxx.linker_so[0], cxx.compiler_cxx[0]] \
520:                         + cxx.linker_so[2:]
521:     else:
522:         cxx.linker_so = [cxx.compiler_cxx[0]] + cxx.linker_so[1:]
523:     return cxx
524: 
525: replace_method(CCompiler, 'cxx_compiler', CCompiler_cxx_compiler)
526: 
527: compiler_class['intel'] = ('intelccompiler', 'IntelCCompiler',
528:                            "Intel C Compiler for 32-bit applications")
529: compiler_class['intele'] = ('intelccompiler', 'IntelItaniumCCompiler',
530:                             "Intel C Itanium Compiler for Itanium-based applications")
531: compiler_class['intelem'] = ('intelccompiler', 'IntelEM64TCCompiler',
532:                              "Intel C Compiler for 64-bit applications")
533: compiler_class['intelw'] = ('intelccompiler', 'IntelCCompilerW',
534:                             "Intel C Compiler for 32-bit applications on Windows")
535: compiler_class['intelemw'] = ('intelccompiler', 'IntelEM64TCCompilerW',
536:                               "Intel C Compiler for 64-bit applications on Windows")
537: compiler_class['pathcc'] = ('pathccompiler', 'PathScaleCCompiler',
538:                             "PathScale Compiler for SiCortex-based applications")
539: ccompiler._default_compilers += (('linux.*', 'intel'),
540:                                  ('linux.*', 'intele'),
541:                                  ('linux.*', 'intelem'),
542:                                  ('linux.*', 'pathcc'),
543:                                  ('nt', 'intelw'),
544:                                  ('nt', 'intelemw'))
545: 
546: if sys.platform == 'win32':
547:     compiler_class['mingw32'] = ('mingw32ccompiler', 'Mingw32CCompiler',
548:                                  "Mingw32 port of GNU C Compiler for Win32"\
549:                                  "(for MSC built Python)")
550:     if mingw32():
551:         # On windows platforms, we want to default to mingw32 (gcc)
552:         # because msvc can't build blitz stuff.
553:         log.info('Setting mingw32 as default compiler for nt.')
554:         ccompiler._default_compilers = (('nt', 'mingw32'),) \
555:                                        + ccompiler._default_compilers
556: 
557: 
558: _distutils_new_compiler = new_compiler
559: def new_compiler (plat=None,
560:                   compiler=None,
561:                   verbose=0,
562:                   dry_run=0,
563:                   force=0):
564:     # Try first C compilers from numpy.distutils.
565:     if plat is None:
566:         plat = os.name
567:     try:
568:         if compiler is None:
569:             compiler = get_default_compiler(plat)
570:         (module_name, class_name, long_description) = compiler_class[compiler]
571:     except KeyError:
572:         msg = "don't know how to compile C/C++ code on platform '%s'" % plat
573:         if compiler is not None:
574:             msg = msg + " with '%s' compiler" % compiler
575:         raise DistutilsPlatformError(msg)
576:     module_name = "numpy.distutils." + module_name
577:     try:
578:         __import__ (module_name)
579:     except ImportError:
580:         msg = str(get_exception())
581:         log.info('%s in numpy.distutils; trying from distutils',
582:                  str(msg))
583:         module_name = module_name[6:]
584:         try:
585:             __import__(module_name)
586:         except ImportError:
587:             msg = str(get_exception())
588:             raise DistutilsModuleError("can't compile C/C++ code: unable to load module '%s'" % \
589:                   module_name)
590:     try:
591:         module = sys.modules[module_name]
592:         klass = vars(module)[class_name]
593:     except KeyError:
594:         raise DistutilsModuleError(("can't compile C/C++ code: unable to find class '%s' " +
595:                "in module '%s'") % (class_name, module_name))
596:     compiler = klass(None, dry_run, force)
597:     log.debug('new_compiler returns %s' % (klass))
598:     return compiler
599: 
600: ccompiler.new_compiler = new_compiler
601: 
602: _distutils_gen_lib_options = gen_lib_options
603: def gen_lib_options(compiler, library_dirs, runtime_library_dirs, libraries):
604:     library_dirs = quote_args(library_dirs)
605:     runtime_library_dirs = quote_args(runtime_library_dirs)
606:     r = _distutils_gen_lib_options(compiler, library_dirs,
607:                                    runtime_library_dirs, libraries)
608:     lib_opts = []
609:     for i in r:
610:         if is_sequence(i):
611:             lib_opts.extend(list(i))
612:         else:
613:             lib_opts.append(i)
614:     return lib_opts
615: ccompiler.gen_lib_options = gen_lib_options
616: 
617: # Also fix up the various compiler modules, which do
618: # from distutils.ccompiler import gen_lib_options
619: # Don't bother with mwerks, as we don't support Classic Mac.
620: for _cc in ['msvc9', 'msvc', 'bcpp', 'cygwinc', 'emxc', 'unixc']:
621:     _m = sys.modules.get('distutils.' + _cc + 'compiler')
622:     if _m is not None:
623:         setattr(_m, 'gen_lib_options', gen_lib_options)
624: 
625: _distutils_gen_preprocess_options = gen_preprocess_options
626: def gen_preprocess_options (macros, include_dirs):
627:     include_dirs = quote_args(include_dirs)
628:     return _distutils_gen_preprocess_options(macros, include_dirs)
629: ccompiler.gen_preprocess_options = gen_preprocess_options
630: 
631: ##Fix distutils.util.split_quoted:
632: # NOTE:  I removed this fix in revision 4481 (see ticket #619), but it appears
633: # that removing this fix causes f2py problems on Windows XP (see ticket #723).
634: # Specifically, on WinXP when gfortran is installed in a directory path, which
635: # contains spaces, then f2py is unable to find it.
636: import string
637: _wordchars_re = re.compile(r'[^\\\'\"%s ]*' % string.whitespace)
638: _squote_re = re.compile(r"'(?:[^'\\]|\\.)*'")
639: _dquote_re = re.compile(r'"(?:[^"\\]|\\.)*"')
640: _has_white_re = re.compile(r'\s')
641: def split_quoted(s):
642:     s = s.strip()
643:     words = []
644:     pos = 0
645: 
646:     while s:
647:         m = _wordchars_re.match(s, pos)
648:         end = m.end()
649:         if end == len(s):
650:             words.append(s[:end])
651:             break
652: 
653:         if s[end] in string.whitespace: # unescaped, unquoted whitespace: now
654:             words.append(s[:end])       # we definitely have a word delimiter
655:             s = s[end:].lstrip()
656:             pos = 0
657: 
658:         elif s[end] == '\\':            # preserve whatever is being escaped;
659:                                         # will become part of the current word
660:             s = s[:end] + s[end+1:]
661:             pos = end+1
662: 
663:         else:
664:             if s[end] == "'":           # slurp singly-quoted string
665:                 m = _squote_re.match(s, end)
666:             elif s[end] == '"':         # slurp doubly-quoted string
667:                 m = _dquote_re.match(s, end)
668:             else:
669:                 raise RuntimeError("this can't happen (bad char '%c')" % s[end])
670: 
671:             if m is None:
672:                 raise ValueError("bad string (mismatched %s quotes?)" % s[end])
673: 
674:             (beg, end) = m.span()
675:             if _has_white_re.search(s[beg+1:end-1]):
676:                 s = s[:beg] + s[beg+1:end-1] + s[end:]
677:                 pos = m.end() - 2
678:             else:
679:                 # Keeping quotes when a quoted word does not contain
680:                 # white-space. XXX: send a patch to distutils
681:                 pos = m.end()
682: 
683:         if pos >= len(s):
684:             words.append(s)
685:             break
686: 
687:     return words
688: ccompiler.split_quoted = split_quoted
689: ##Fix distutils.util.split_quoted:
690: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import re' statement (line 4)
import re

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import sys' statement (line 5)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import types' statement (line 6)
import types

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'types', types, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from copy import copy' statement (line 7)
from copy import copy

import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'copy', None, module_type_store, ['copy'], [copy])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from distutils import ccompiler' statement (line 8)
from distutils import ccompiler

import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils', None, module_type_store, ['ccompiler'], [ccompiler])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from distutils.ccompiler import ' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_26368 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.ccompiler')

if (type(import_26368) is not StypyTypeError):

    if (import_26368 != 'pyd_module'):
        __import__(import_26368)
        sys_modules_26369 = sys.modules[import_26368]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.ccompiler', sys_modules_26369.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_26369, sys_modules_26369.module_type_store, module_type_store)
    else:
        from distutils.ccompiler import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.ccompiler', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'distutils.ccompiler' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.ccompiler', import_26368)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from distutils.errors import DistutilsExecError, DistutilsModuleError, DistutilsPlatformError' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_26370 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors')

if (type(import_26370) is not StypyTypeError):

    if (import_26370 != 'pyd_module'):
        __import__(import_26370)
        sys_modules_26371 = sys.modules[import_26370]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors', sys_modules_26371.module_type_store, module_type_store, ['DistutilsExecError', 'DistutilsModuleError', 'DistutilsPlatformError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_26371, sys_modules_26371.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsExecError, DistutilsModuleError, DistutilsPlatformError

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors', None, module_type_store, ['DistutilsExecError', 'DistutilsModuleError', 'DistutilsPlatformError'], [DistutilsExecError, DistutilsModuleError, DistutilsPlatformError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors', import_26370)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils.sysconfig import customize_compiler' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_26372 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.sysconfig')

if (type(import_26372) is not StypyTypeError):

    if (import_26372 != 'pyd_module'):
        __import__(import_26372)
        sys_modules_26373 = sys.modules[import_26372]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.sysconfig', sys_modules_26373.module_type_store, module_type_store, ['customize_compiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_26373, sys_modules_26373.module_type_store, module_type_store)
    else:
        from distutils.sysconfig import customize_compiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.sysconfig', None, module_type_store, ['customize_compiler'], [customize_compiler])

else:
    # Assigning a type to the variable 'distutils.sysconfig' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.sysconfig', import_26372)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from distutils.version import LooseVersion' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_26374 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.version')

if (type(import_26374) is not StypyTypeError):

    if (import_26374 != 'pyd_module'):
        __import__(import_26374)
        sys_modules_26375 = sys.modules[import_26374]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.version', sys_modules_26375.module_type_store, module_type_store, ['LooseVersion'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_26375, sys_modules_26375.module_type_store, module_type_store)
    else:
        from distutils.version import LooseVersion

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.version', None, module_type_store, ['LooseVersion'], [LooseVersion])

else:
    # Assigning a type to the variable 'distutils.version' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.version', import_26374)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from numpy.distutils import log' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_26376 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.distutils')

if (type(import_26376) is not StypyTypeError):

    if (import_26376 != 'pyd_module'):
        __import__(import_26376)
        sys_modules_26377 = sys.modules[import_26376]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.distutils', sys_modules_26377.module_type_store, module_type_store, ['log'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_26377, sys_modules_26377.module_type_store, module_type_store)
    else:
        from numpy.distutils import log

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.distutils', None, module_type_store, ['log'], [log])

else:
    # Assigning a type to the variable 'numpy.distutils' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.distutils', import_26376)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from numpy.distutils.compat import get_exception' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_26378 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy.distutils.compat')

if (type(import_26378) is not StypyTypeError):

    if (import_26378 != 'pyd_module'):
        __import__(import_26378)
        sys_modules_26379 = sys.modules[import_26378]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy.distutils.compat', sys_modules_26379.module_type_store, module_type_store, ['get_exception'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_26379, sys_modules_26379.module_type_store, module_type_store)
    else:
        from numpy.distutils.compat import get_exception

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy.distutils.compat', None, module_type_store, ['get_exception'], [get_exception])

else:
    # Assigning a type to the variable 'numpy.distutils.compat' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy.distutils.compat', import_26378)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from numpy.distutils.exec_command import exec_command' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_26380 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy.distutils.exec_command')

if (type(import_26380) is not StypyTypeError):

    if (import_26380 != 'pyd_module'):
        __import__(import_26380)
        sys_modules_26381 = sys.modules[import_26380]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy.distutils.exec_command', sys_modules_26381.module_type_store, module_type_store, ['exec_command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_26381, sys_modules_26381.module_type_store, module_type_store)
    else:
        from numpy.distutils.exec_command import exec_command

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy.distutils.exec_command', None, module_type_store, ['exec_command'], [exec_command])

else:
    # Assigning a type to the variable 'numpy.distutils.exec_command' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy.distutils.exec_command', import_26380)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from numpy.distutils.misc_util import cyg2win32, is_sequence, mingw32, quote_args, get_num_build_jobs' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_26382 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.distutils.misc_util')

if (type(import_26382) is not StypyTypeError):

    if (import_26382 != 'pyd_module'):
        __import__(import_26382)
        sys_modules_26383 = sys.modules[import_26382]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.distutils.misc_util', sys_modules_26383.module_type_store, module_type_store, ['cyg2win32', 'is_sequence', 'mingw32', 'quote_args', 'get_num_build_jobs'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_26383, sys_modules_26383.module_type_store, module_type_store)
    else:
        from numpy.distutils.misc_util import cyg2win32, is_sequence, mingw32, quote_args, get_num_build_jobs

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.distutils.misc_util', None, module_type_store, ['cyg2win32', 'is_sequence', 'mingw32', 'quote_args', 'get_num_build_jobs'], [cyg2win32, is_sequence, mingw32, quote_args, get_num_build_jobs])

else:
    # Assigning a type to the variable 'numpy.distutils.misc_util' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.distutils.misc_util', import_26382)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')


@norecursion
def replace_method(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'replace_method'
    module_type_store = module_type_store.open_function_context('replace_method', 22, 0, False)
    
    # Passed parameters checking function
    replace_method.stypy_localization = localization
    replace_method.stypy_type_of_self = None
    replace_method.stypy_type_store = module_type_store
    replace_method.stypy_function_name = 'replace_method'
    replace_method.stypy_param_names_list = ['klass', 'method_name', 'func']
    replace_method.stypy_varargs_param_name = None
    replace_method.stypy_kwargs_param_name = None
    replace_method.stypy_call_defaults = defaults
    replace_method.stypy_call_varargs = varargs
    replace_method.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'replace_method', ['klass', 'method_name', 'func'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'replace_method', localization, ['klass', 'method_name', 'func'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'replace_method(...)' code ##################

    
    
    
    # Obtaining the type of the subscript
    int_26384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 24), 'int')
    # Getting the type of 'sys' (line 23)
    sys_26385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 7), 'sys')
    # Obtaining the member 'version_info' of a type (line 23)
    version_info_26386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 7), sys_26385, 'version_info')
    # Obtaining the member '__getitem__' of a type (line 23)
    getitem___26387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 7), version_info_26386, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 23)
    subscript_call_result_26388 = invoke(stypy.reporting.localization.Localization(__file__, 23, 7), getitem___26387, int_26384)
    
    int_26389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 29), 'int')
    # Applying the binary operator '<' (line 23)
    result_lt_26390 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 7), '<', subscript_call_result_26388, int_26389)
    
    # Testing the type of an if condition (line 23)
    if_condition_26391 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 23, 4), result_lt_26390)
    # Assigning a type to the variable 'if_condition_26391' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'if_condition_26391', if_condition_26391)
    # SSA begins for if statement (line 23)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 24):
    
    # Assigning a Call to a Name (line 24):
    
    # Assigning a Call to a Name (line 24):
    
    # Call to MethodType(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 'func' (line 24)
    func_26394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 29), 'func', False)
    # Getting the type of 'None' (line 24)
    None_26395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 35), 'None', False)
    # Getting the type of 'klass' (line 24)
    klass_26396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 41), 'klass', False)
    # Processing the call keyword arguments (line 24)
    kwargs_26397 = {}
    # Getting the type of 'types' (line 24)
    types_26392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'types', False)
    # Obtaining the member 'MethodType' of a type (line 24)
    MethodType_26393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 12), types_26392, 'MethodType')
    # Calling MethodType(args, kwargs) (line 24)
    MethodType_call_result_26398 = invoke(stypy.reporting.localization.Localization(__file__, 24, 12), MethodType_26393, *[func_26394, None_26395, klass_26396], **kwargs_26397)
    
    # Assigning a type to the variable 'm' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'm', MethodType_call_result_26398)
    # SSA branch for the else part of an if statement (line 23)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Lambda to a Name (line 27):
    
    # Assigning a Lambda to a Name (line 27):
    
    # Assigning a Lambda to a Name (line 27):

    @norecursion
    def _stypy_temp_lambda_14(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_14'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_14', 27, 12, True)
        # Passed parameters checking function
        _stypy_temp_lambda_14.stypy_localization = localization
        _stypy_temp_lambda_14.stypy_type_of_self = None
        _stypy_temp_lambda_14.stypy_type_store = module_type_store
        _stypy_temp_lambda_14.stypy_function_name = '_stypy_temp_lambda_14'
        _stypy_temp_lambda_14.stypy_param_names_list = ['self']
        _stypy_temp_lambda_14.stypy_varargs_param_name = 'args'
        _stypy_temp_lambda_14.stypy_kwargs_param_name = 'kw'
        _stypy_temp_lambda_14.stypy_call_defaults = defaults
        _stypy_temp_lambda_14.stypy_call_varargs = varargs
        _stypy_temp_lambda_14.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_14', ['self'], 'args', 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_14', ['self'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to func(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'self' (line 27)
        self_26400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 43), 'self', False)
        # Getting the type of 'args' (line 27)
        args_26401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 50), 'args', False)
        # Processing the call keyword arguments (line 27)
        # Getting the type of 'kw' (line 27)
        kw_26402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 58), 'kw', False)
        kwargs_26403 = {'kw_26402': kw_26402}
        # Getting the type of 'func' (line 27)
        func_26399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 38), 'func', False)
        # Calling func(args, kwargs) (line 27)
        func_call_result_26404 = invoke(stypy.reporting.localization.Localization(__file__, 27, 38), func_26399, *[self_26400, args_26401], **kwargs_26403)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'stypy_return_type', func_call_result_26404)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_14' in the type store
        # Getting the type of 'stypy_return_type' (line 27)
        stypy_return_type_26405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26405)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_14'
        return stypy_return_type_26405

    # Assigning a type to the variable '_stypy_temp_lambda_14' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), '_stypy_temp_lambda_14', _stypy_temp_lambda_14)
    # Getting the type of '_stypy_temp_lambda_14' (line 27)
    _stypy_temp_lambda_14_26406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), '_stypy_temp_lambda_14')
    # Assigning a type to the variable 'm' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'm', _stypy_temp_lambda_14_26406)
    # SSA join for if statement (line 23)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to setattr(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'klass' (line 28)
    klass_26408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'klass', False)
    # Getting the type of 'method_name' (line 28)
    method_name_26409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 19), 'method_name', False)
    # Getting the type of 'm' (line 28)
    m_26410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 32), 'm', False)
    # Processing the call keyword arguments (line 28)
    kwargs_26411 = {}
    # Getting the type of 'setattr' (line 28)
    setattr_26407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'setattr', False)
    # Calling setattr(args, kwargs) (line 28)
    setattr_call_result_26412 = invoke(stypy.reporting.localization.Localization(__file__, 28, 4), setattr_26407, *[klass_26408, method_name_26409, m_26410], **kwargs_26411)
    
    
    # ################# End of 'replace_method(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'replace_method' in the type store
    # Getting the type of 'stypy_return_type' (line 22)
    stypy_return_type_26413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_26413)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'replace_method'
    return stypy_return_type_26413

# Assigning a type to the variable 'replace_method' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'replace_method', replace_method)

@norecursion
def CCompiler_spawn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 31)
    None_26414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 39), 'None')
    defaults = [None_26414]
    # Create a new context for function 'CCompiler_spawn'
    module_type_store = module_type_store.open_function_context('CCompiler_spawn', 31, 0, False)
    
    # Passed parameters checking function
    CCompiler_spawn.stypy_localization = localization
    CCompiler_spawn.stypy_type_of_self = None
    CCompiler_spawn.stypy_type_store = module_type_store
    CCompiler_spawn.stypy_function_name = 'CCompiler_spawn'
    CCompiler_spawn.stypy_param_names_list = ['self', 'cmd', 'display']
    CCompiler_spawn.stypy_varargs_param_name = None
    CCompiler_spawn.stypy_kwargs_param_name = None
    CCompiler_spawn.stypy_call_defaults = defaults
    CCompiler_spawn.stypy_call_varargs = varargs
    CCompiler_spawn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'CCompiler_spawn', ['self', 'cmd', 'display'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'CCompiler_spawn', localization, ['self', 'cmd', 'display'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'CCompiler_spawn(...)' code ##################

    str_26415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, (-1)), 'str', '\n    Execute a command in a sub-process.\n\n    Parameters\n    ----------\n    cmd : str\n        The command to execute.\n    display : str or sequence of str, optional\n        The text to add to the log file kept by `numpy.distutils`.\n        If not given, `display` is equal to `cmd`.\n\n    Returns\n    -------\n    None\n\n    Raises\n    ------\n    DistutilsExecError\n        If the command failed, i.e. the exit status was not 0.\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 53)
    # Getting the type of 'display' (line 53)
    display_26416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 7), 'display')
    # Getting the type of 'None' (line 53)
    None_26417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 18), 'None')
    
    (may_be_26418, more_types_in_union_26419) = may_be_none(display_26416, None_26417)

    if may_be_26418:

        if more_types_in_union_26419:
            # Runtime conditional SSA (line 53)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 54):
        
        # Assigning a Name to a Name (line 54):
        
        # Assigning a Name to a Name (line 54):
        # Getting the type of 'cmd' (line 54)
        cmd_26420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 18), 'cmd')
        # Assigning a type to the variable 'display' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'display', cmd_26420)
        
        
        # Call to is_sequence(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'display' (line 55)
        display_26422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 23), 'display', False)
        # Processing the call keyword arguments (line 55)
        kwargs_26423 = {}
        # Getting the type of 'is_sequence' (line 55)
        is_sequence_26421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 11), 'is_sequence', False)
        # Calling is_sequence(args, kwargs) (line 55)
        is_sequence_call_result_26424 = invoke(stypy.reporting.localization.Localization(__file__, 55, 11), is_sequence_26421, *[display_26422], **kwargs_26423)
        
        # Testing the type of an if condition (line 55)
        if_condition_26425 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 55, 8), is_sequence_call_result_26424)
        # Assigning a type to the variable 'if_condition_26425' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'if_condition_26425', if_condition_26425)
        # SSA begins for if statement (line 55)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 56):
        
        # Assigning a Call to a Name (line 56):
        
        # Assigning a Call to a Name (line 56):
        
        # Call to join(...): (line 56)
        # Processing the call arguments (line 56)
        
        # Call to list(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'display' (line 56)
        display_26429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 36), 'display', False)
        # Processing the call keyword arguments (line 56)
        kwargs_26430 = {}
        # Getting the type of 'list' (line 56)
        list_26428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 31), 'list', False)
        # Calling list(args, kwargs) (line 56)
        list_call_result_26431 = invoke(stypy.reporting.localization.Localization(__file__, 56, 31), list_26428, *[display_26429], **kwargs_26430)
        
        # Processing the call keyword arguments (line 56)
        kwargs_26432 = {}
        str_26426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 22), 'str', ' ')
        # Obtaining the member 'join' of a type (line 56)
        join_26427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 22), str_26426, 'join')
        # Calling join(args, kwargs) (line 56)
        join_call_result_26433 = invoke(stypy.reporting.localization.Localization(__file__, 56, 22), join_26427, *[list_call_result_26431], **kwargs_26432)
        
        # Assigning a type to the variable 'display' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'display', join_call_result_26433)
        # SSA join for if statement (line 55)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_26419:
            # SSA join for if statement (line 53)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to info(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'display' (line 57)
    display_26436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 13), 'display', False)
    # Processing the call keyword arguments (line 57)
    kwargs_26437 = {}
    # Getting the type of 'log' (line 57)
    log_26434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'log', False)
    # Obtaining the member 'info' of a type (line 57)
    info_26435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 4), log_26434, 'info')
    # Calling info(args, kwargs) (line 57)
    info_call_result_26438 = invoke(stypy.reporting.localization.Localization(__file__, 57, 4), info_26435, *[display_26436], **kwargs_26437)
    
    
    # Assigning a Call to a Tuple (line 58):
    
    # Assigning a Call to a Name:
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'cmd' (line 58)
    cmd_26440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 24), 'cmd', False)
    # Processing the call keyword arguments (line 58)
    kwargs_26441 = {}
    # Getting the type of 'exec_command' (line 58)
    exec_command_26439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 11), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 58)
    exec_command_call_result_26442 = invoke(stypy.reporting.localization.Localization(__file__, 58, 11), exec_command_26439, *[cmd_26440], **kwargs_26441)
    
    # Assigning a type to the variable 'call_assignment_26335' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'call_assignment_26335', exec_command_call_result_26442)
    
    # Assigning a Call to a Name (line 58):
    
    # Assigning a Call to a Name (line 58):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_26445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 4), 'int')
    # Processing the call keyword arguments
    kwargs_26446 = {}
    # Getting the type of 'call_assignment_26335' (line 58)
    call_assignment_26335_26443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'call_assignment_26335', False)
    # Obtaining the member '__getitem__' of a type (line 58)
    getitem___26444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 4), call_assignment_26335_26443, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_26447 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___26444, *[int_26445], **kwargs_26446)
    
    # Assigning a type to the variable 'call_assignment_26336' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'call_assignment_26336', getitem___call_result_26447)
    
    # Assigning a Name to a Name (line 58):
    
    # Assigning a Name to a Name (line 58):
    # Getting the type of 'call_assignment_26336' (line 58)
    call_assignment_26336_26448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'call_assignment_26336')
    # Assigning a type to the variable 's' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 's', call_assignment_26336_26448)
    
    # Assigning a Call to a Name (line 58):
    
    # Assigning a Call to a Name (line 58):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_26451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 4), 'int')
    # Processing the call keyword arguments
    kwargs_26452 = {}
    # Getting the type of 'call_assignment_26335' (line 58)
    call_assignment_26335_26449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'call_assignment_26335', False)
    # Obtaining the member '__getitem__' of a type (line 58)
    getitem___26450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 4), call_assignment_26335_26449, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_26453 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___26450, *[int_26451], **kwargs_26452)
    
    # Assigning a type to the variable 'call_assignment_26337' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'call_assignment_26337', getitem___call_result_26453)
    
    # Assigning a Name to a Name (line 58):
    
    # Assigning a Name to a Name (line 58):
    # Getting the type of 'call_assignment_26337' (line 58)
    call_assignment_26337_26454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'call_assignment_26337')
    # Assigning a type to the variable 'o' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 7), 'o', call_assignment_26337_26454)
    
    # Getting the type of 's' (line 59)
    s_26455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 7), 's')
    # Testing the type of an if condition (line 59)
    if_condition_26456 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 4), s_26455)
    # Assigning a type to the variable 'if_condition_26456' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'if_condition_26456', if_condition_26456)
    # SSA begins for if statement (line 59)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to is_sequence(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'cmd' (line 60)
    cmd_26458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'cmd', False)
    # Processing the call keyword arguments (line 60)
    kwargs_26459 = {}
    # Getting the type of 'is_sequence' (line 60)
    is_sequence_26457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 11), 'is_sequence', False)
    # Calling is_sequence(args, kwargs) (line 60)
    is_sequence_call_result_26460 = invoke(stypy.reporting.localization.Localization(__file__, 60, 11), is_sequence_26457, *[cmd_26458], **kwargs_26459)
    
    # Testing the type of an if condition (line 60)
    if_condition_26461 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 8), is_sequence_call_result_26460)
    # Assigning a type to the variable 'if_condition_26461' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'if_condition_26461', if_condition_26461)
    # SSA begins for if statement (line 60)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 61):
    
    # Assigning a Call to a Name (line 61):
    
    # Assigning a Call to a Name (line 61):
    
    # Call to join(...): (line 61)
    # Processing the call arguments (line 61)
    
    # Call to list(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'cmd' (line 61)
    cmd_26465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 32), 'cmd', False)
    # Processing the call keyword arguments (line 61)
    kwargs_26466 = {}
    # Getting the type of 'list' (line 61)
    list_26464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 27), 'list', False)
    # Calling list(args, kwargs) (line 61)
    list_call_result_26467 = invoke(stypy.reporting.localization.Localization(__file__, 61, 27), list_26464, *[cmd_26465], **kwargs_26466)
    
    # Processing the call keyword arguments (line 61)
    kwargs_26468 = {}
    str_26462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 18), 'str', ' ')
    # Obtaining the member 'join' of a type (line 61)
    join_26463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 18), str_26462, 'join')
    # Calling join(args, kwargs) (line 61)
    join_call_result_26469 = invoke(stypy.reporting.localization.Localization(__file__, 61, 18), join_26463, *[list_call_result_26467], **kwargs_26468)
    
    # Assigning a type to the variable 'cmd' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'cmd', join_call_result_26469)
    # SSA join for if statement (line 60)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 62)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to print(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'o' (line 63)
    o_26471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 18), 'o', False)
    # Processing the call keyword arguments (line 63)
    kwargs_26472 = {}
    # Getting the type of 'print' (line 63)
    print_26470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'print', False)
    # Calling print(args, kwargs) (line 63)
    print_call_result_26473 = invoke(stypy.reporting.localization.Localization(__file__, 63, 12), print_26470, *[o_26471], **kwargs_26472)
    
    # SSA branch for the except part of a try statement (line 62)
    # SSA branch for the except 'UnicodeError' branch of a try statement (line 62)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 62)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to search(...): (line 67)
    # Processing the call arguments (line 67)
    str_26476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 21), 'str', 'Too many open files')
    # Getting the type of 'o' (line 67)
    o_26477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 44), 'o', False)
    # Processing the call keyword arguments (line 67)
    kwargs_26478 = {}
    # Getting the type of 're' (line 67)
    re_26474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 're', False)
    # Obtaining the member 'search' of a type (line 67)
    search_26475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 11), re_26474, 'search')
    # Calling search(args, kwargs) (line 67)
    search_call_result_26479 = invoke(stypy.reporting.localization.Localization(__file__, 67, 11), search_26475, *[str_26476, o_26477], **kwargs_26478)
    
    # Testing the type of an if condition (line 67)
    if_condition_26480 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 8), search_call_result_26479)
    # Assigning a type to the variable 'if_condition_26480' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'if_condition_26480', if_condition_26480)
    # SSA begins for if statement (line 67)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 68):
    
    # Assigning a Str to a Name (line 68):
    
    # Assigning a Str to a Name (line 68):
    str_26481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 18), 'str', '\nTry rerunning setup command until build succeeds.')
    # Assigning a type to the variable 'msg' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'msg', str_26481)
    # SSA branch for the else part of an if statement (line 67)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 70):
    
    # Assigning a Str to a Name (line 70):
    
    # Assigning a Str to a Name (line 70):
    str_26482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 18), 'str', '')
    # Assigning a type to the variable 'msg' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'msg', str_26482)
    # SSA join for if statement (line 67)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to DistutilsExecError(...): (line 71)
    # Processing the call arguments (line 71)
    str_26484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 33), 'str', 'Command "%s" failed with exit status %d%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 71)
    tuple_26485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 80), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 71)
    # Adding element type (line 71)
    # Getting the type of 'cmd' (line 71)
    cmd_26486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 80), 'cmd', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 80), tuple_26485, cmd_26486)
    # Adding element type (line 71)
    # Getting the type of 's' (line 71)
    s_26487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 85), 's', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 80), tuple_26485, s_26487)
    # Adding element type (line 71)
    # Getting the type of 'msg' (line 71)
    msg_26488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 88), 'msg', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 80), tuple_26485, msg_26488)
    
    # Applying the binary operator '%' (line 71)
    result_mod_26489 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 33), '%', str_26484, tuple_26485)
    
    # Processing the call keyword arguments (line 71)
    kwargs_26490 = {}
    # Getting the type of 'DistutilsExecError' (line 71)
    DistutilsExecError_26483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 14), 'DistutilsExecError', False)
    # Calling DistutilsExecError(args, kwargs) (line 71)
    DistutilsExecError_call_result_26491 = invoke(stypy.reporting.localization.Localization(__file__, 71, 14), DistutilsExecError_26483, *[result_mod_26489], **kwargs_26490)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 71, 8), DistutilsExecError_call_result_26491, 'raise parameter', BaseException)
    # SSA join for if statement (line 59)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'CCompiler_spawn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'CCompiler_spawn' in the type store
    # Getting the type of 'stypy_return_type' (line 31)
    stypy_return_type_26492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_26492)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'CCompiler_spawn'
    return stypy_return_type_26492

# Assigning a type to the variable 'CCompiler_spawn' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'CCompiler_spawn', CCompiler_spawn)

# Call to replace_method(...): (line 73)
# Processing the call arguments (line 73)
# Getting the type of 'CCompiler' (line 73)
CCompiler_26494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 15), 'CCompiler', False)
str_26495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 26), 'str', 'spawn')
# Getting the type of 'CCompiler_spawn' (line 73)
CCompiler_spawn_26496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 35), 'CCompiler_spawn', False)
# Processing the call keyword arguments (line 73)
kwargs_26497 = {}
# Getting the type of 'replace_method' (line 73)
replace_method_26493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'replace_method', False)
# Calling replace_method(args, kwargs) (line 73)
replace_method_call_result_26498 = invoke(stypy.reporting.localization.Localization(__file__, 73, 0), replace_method_26493, *[CCompiler_26494, str_26495, CCompiler_spawn_26496], **kwargs_26497)


@norecursion
def CCompiler_object_filenames(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_26499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 65), 'int')
    str_26500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 79), 'str', '')
    defaults = [int_26499, str_26500]
    # Create a new context for function 'CCompiler_object_filenames'
    module_type_store = module_type_store.open_function_context('CCompiler_object_filenames', 75, 0, False)
    
    # Passed parameters checking function
    CCompiler_object_filenames.stypy_localization = localization
    CCompiler_object_filenames.stypy_type_of_self = None
    CCompiler_object_filenames.stypy_type_store = module_type_store
    CCompiler_object_filenames.stypy_function_name = 'CCompiler_object_filenames'
    CCompiler_object_filenames.stypy_param_names_list = ['self', 'source_filenames', 'strip_dir', 'output_dir']
    CCompiler_object_filenames.stypy_varargs_param_name = None
    CCompiler_object_filenames.stypy_kwargs_param_name = None
    CCompiler_object_filenames.stypy_call_defaults = defaults
    CCompiler_object_filenames.stypy_call_varargs = varargs
    CCompiler_object_filenames.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'CCompiler_object_filenames', ['self', 'source_filenames', 'strip_dir', 'output_dir'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'CCompiler_object_filenames', localization, ['self', 'source_filenames', 'strip_dir', 'output_dir'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'CCompiler_object_filenames(...)' code ##################

    str_26501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, (-1)), 'str', '\n    Return the name of the object files for the given source files.\n\n    Parameters\n    ----------\n    source_filenames : list of str\n        The list of paths to source files. Paths can be either relative or\n        absolute, this is handled transparently.\n    strip_dir : bool, optional\n        Whether to strip the directory from the returned paths. If True,\n        the file name prepended by `output_dir` is returned. Default is False.\n    output_dir : str, optional\n        If given, this path is prepended to the returned paths to the\n        object files.\n\n    Returns\n    -------\n    obj_names : list of str\n        The list of paths to the object files corresponding to the source\n        files in `source_filenames`.\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 98)
    # Getting the type of 'output_dir' (line 98)
    output_dir_26502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 7), 'output_dir')
    # Getting the type of 'None' (line 98)
    None_26503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 21), 'None')
    
    (may_be_26504, more_types_in_union_26505) = may_be_none(output_dir_26502, None_26503)

    if may_be_26504:

        if more_types_in_union_26505:
            # Runtime conditional SSA (line 98)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Str to a Name (line 99):
        
        # Assigning a Str to a Name (line 99):
        
        # Assigning a Str to a Name (line 99):
        str_26506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 21), 'str', '')
        # Assigning a type to the variable 'output_dir' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'output_dir', str_26506)

        if more_types_in_union_26505:
            # SSA join for if statement (line 98)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a List to a Name (line 100):
    
    # Assigning a List to a Name (line 100):
    
    # Assigning a List to a Name (line 100):
    
    # Obtaining an instance of the builtin type 'list' (line 100)
    list_26507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 100)
    
    # Assigning a type to the variable 'obj_names' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'obj_names', list_26507)
    
    # Getting the type of 'source_filenames' (line 101)
    source_filenames_26508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 20), 'source_filenames')
    # Testing the type of a for loop iterable (line 101)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 101, 4), source_filenames_26508)
    # Getting the type of the for loop variable (line 101)
    for_loop_var_26509 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 101, 4), source_filenames_26508)
    # Assigning a type to the variable 'src_name' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'src_name', for_loop_var_26509)
    # SSA begins for a for statement (line 101)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Tuple (line 102):
    
    # Assigning a Call to a Name:
    
    # Assigning a Call to a Name:
    
    # Call to splitext(...): (line 102)
    # Processing the call arguments (line 102)
    
    # Call to normpath(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'src_name' (line 102)
    src_name_26516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 54), 'src_name', False)
    # Processing the call keyword arguments (line 102)
    kwargs_26517 = {}
    # Getting the type of 'os' (line 102)
    os_26513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 37), 'os', False)
    # Obtaining the member 'path' of a type (line 102)
    path_26514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 37), os_26513, 'path')
    # Obtaining the member 'normpath' of a type (line 102)
    normpath_26515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 37), path_26514, 'normpath')
    # Calling normpath(args, kwargs) (line 102)
    normpath_call_result_26518 = invoke(stypy.reporting.localization.Localization(__file__, 102, 37), normpath_26515, *[src_name_26516], **kwargs_26517)
    
    # Processing the call keyword arguments (line 102)
    kwargs_26519 = {}
    # Getting the type of 'os' (line 102)
    os_26510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 20), 'os', False)
    # Obtaining the member 'path' of a type (line 102)
    path_26511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 20), os_26510, 'path')
    # Obtaining the member 'splitext' of a type (line 102)
    splitext_26512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 20), path_26511, 'splitext')
    # Calling splitext(args, kwargs) (line 102)
    splitext_call_result_26520 = invoke(stypy.reporting.localization.Localization(__file__, 102, 20), splitext_26512, *[normpath_call_result_26518], **kwargs_26519)
    
    # Assigning a type to the variable 'call_assignment_26338' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'call_assignment_26338', splitext_call_result_26520)
    
    # Assigning a Call to a Name (line 102):
    
    # Assigning a Call to a Name (line 102):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_26523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 8), 'int')
    # Processing the call keyword arguments
    kwargs_26524 = {}
    # Getting the type of 'call_assignment_26338' (line 102)
    call_assignment_26338_26521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'call_assignment_26338', False)
    # Obtaining the member '__getitem__' of a type (line 102)
    getitem___26522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), call_assignment_26338_26521, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_26525 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___26522, *[int_26523], **kwargs_26524)
    
    # Assigning a type to the variable 'call_assignment_26339' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'call_assignment_26339', getitem___call_result_26525)
    
    # Assigning a Name to a Name (line 102):
    
    # Assigning a Name to a Name (line 102):
    # Getting the type of 'call_assignment_26339' (line 102)
    call_assignment_26339_26526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'call_assignment_26339')
    # Assigning a type to the variable 'base' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'base', call_assignment_26339_26526)
    
    # Assigning a Call to a Name (line 102):
    
    # Assigning a Call to a Name (line 102):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_26529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 8), 'int')
    # Processing the call keyword arguments
    kwargs_26530 = {}
    # Getting the type of 'call_assignment_26338' (line 102)
    call_assignment_26338_26527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'call_assignment_26338', False)
    # Obtaining the member '__getitem__' of a type (line 102)
    getitem___26528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), call_assignment_26338_26527, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_26531 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___26528, *[int_26529], **kwargs_26530)
    
    # Assigning a type to the variable 'call_assignment_26340' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'call_assignment_26340', getitem___call_result_26531)
    
    # Assigning a Name to a Name (line 102):
    
    # Assigning a Name to a Name (line 102):
    # Getting the type of 'call_assignment_26340' (line 102)
    call_assignment_26340_26532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'call_assignment_26340')
    # Assigning a type to the variable 'ext' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 14), 'ext', call_assignment_26340_26532)
    
    # Assigning a Subscript to a Name (line 103):
    
    # Assigning a Subscript to a Name (line 103):
    
    # Assigning a Subscript to a Name (line 103):
    
    # Obtaining the type of the subscript
    int_26533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 40), 'int')
    
    # Call to splitdrive(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'base' (line 103)
    base_26537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 34), 'base', False)
    # Processing the call keyword arguments (line 103)
    kwargs_26538 = {}
    # Getting the type of 'os' (line 103)
    os_26534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 103)
    path_26535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 15), os_26534, 'path')
    # Obtaining the member 'splitdrive' of a type (line 103)
    splitdrive_26536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 15), path_26535, 'splitdrive')
    # Calling splitdrive(args, kwargs) (line 103)
    splitdrive_call_result_26539 = invoke(stypy.reporting.localization.Localization(__file__, 103, 15), splitdrive_26536, *[base_26537], **kwargs_26538)
    
    # Obtaining the member '__getitem__' of a type (line 103)
    getitem___26540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 15), splitdrive_call_result_26539, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 103)
    subscript_call_result_26541 = invoke(stypy.reporting.localization.Localization(__file__, 103, 15), getitem___26540, int_26533)
    
    # Assigning a type to the variable 'base' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'base', subscript_call_result_26541)
    
    # Assigning a Subscript to a Name (line 104):
    
    # Assigning a Subscript to a Name (line 104):
    
    # Assigning a Subscript to a Name (line 104):
    
    # Obtaining the type of the subscript
    
    # Call to isabs(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'base' (line 104)
    base_26545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 34), 'base', False)
    # Processing the call keyword arguments (line 104)
    kwargs_26546 = {}
    # Getting the type of 'os' (line 104)
    os_26542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 20), 'os', False)
    # Obtaining the member 'path' of a type (line 104)
    path_26543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 20), os_26542, 'path')
    # Obtaining the member 'isabs' of a type (line 104)
    isabs_26544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 20), path_26543, 'isabs')
    # Calling isabs(args, kwargs) (line 104)
    isabs_call_result_26547 = invoke(stypy.reporting.localization.Localization(__file__, 104, 20), isabs_26544, *[base_26545], **kwargs_26546)
    
    slice_26548 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 104, 15), isabs_call_result_26547, None, None)
    # Getting the type of 'base' (line 104)
    base_26549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 'base')
    # Obtaining the member '__getitem__' of a type (line 104)
    getitem___26550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 15), base_26549, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 104)
    subscript_call_result_26551 = invoke(stypy.reporting.localization.Localization(__file__, 104, 15), getitem___26550, slice_26548)
    
    # Assigning a type to the variable 'base' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'base', subscript_call_result_26551)
    
    
    # Call to startswith(...): (line 105)
    # Processing the call arguments (line 105)
    str_26554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 27), 'str', '..')
    # Processing the call keyword arguments (line 105)
    kwargs_26555 = {}
    # Getting the type of 'base' (line 105)
    base_26552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 11), 'base', False)
    # Obtaining the member 'startswith' of a type (line 105)
    startswith_26553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 11), base_26552, 'startswith')
    # Calling startswith(args, kwargs) (line 105)
    startswith_call_result_26556 = invoke(stypy.reporting.localization.Localization(__file__, 105, 11), startswith_26553, *[str_26554], **kwargs_26555)
    
    # Testing the type of an if condition (line 105)
    if_condition_26557 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 8), startswith_call_result_26556)
    # Assigning a type to the variable 'if_condition_26557' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'if_condition_26557', if_condition_26557)
    # SSA begins for if statement (line 105)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 108):
    
    # Assigning a BinOp to a Name (line 108):
    
    # Assigning a BinOp to a Name (line 108):
    
    # Call to rfind(...): (line 108)
    # Processing the call arguments (line 108)
    str_26560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 27), 'str', '..')
    # Processing the call keyword arguments (line 108)
    kwargs_26561 = {}
    # Getting the type of 'base' (line 108)
    base_26558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'base', False)
    # Obtaining the member 'rfind' of a type (line 108)
    rfind_26559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 16), base_26558, 'rfind')
    # Calling rfind(args, kwargs) (line 108)
    rfind_call_result_26562 = invoke(stypy.reporting.localization.Localization(__file__, 108, 16), rfind_26559, *[str_26560], **kwargs_26561)
    
    int_26563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 33), 'int')
    # Applying the binary operator '+' (line 108)
    result_add_26564 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 16), '+', rfind_call_result_26562, int_26563)
    
    # Assigning a type to the variable 'i' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'i', result_add_26564)
    
    # Assigning a Subscript to a Name (line 109):
    
    # Assigning a Subscript to a Name (line 109):
    
    # Assigning a Subscript to a Name (line 109):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 109)
    i_26565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 22), 'i')
    slice_26566 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 109, 16), None, i_26565, None)
    # Getting the type of 'base' (line 109)
    base_26567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'base')
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___26568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 16), base_26567, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_26569 = invoke(stypy.reporting.localization.Localization(__file__, 109, 16), getitem___26568, slice_26566)
    
    # Assigning a type to the variable 'd' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'd', subscript_call_result_26569)
    
    # Assigning a Call to a Name (line 110):
    
    # Assigning a Call to a Name (line 110):
    
    # Assigning a Call to a Name (line 110):
    
    # Call to basename(...): (line 110)
    # Processing the call arguments (line 110)
    
    # Call to abspath(...): (line 110)
    # Processing the call arguments (line 110)
    # Getting the type of 'd' (line 110)
    d_26576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 49), 'd', False)
    # Processing the call keyword arguments (line 110)
    kwargs_26577 = {}
    # Getting the type of 'os' (line 110)
    os_26573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 33), 'os', False)
    # Obtaining the member 'path' of a type (line 110)
    path_26574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 33), os_26573, 'path')
    # Obtaining the member 'abspath' of a type (line 110)
    abspath_26575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 33), path_26574, 'abspath')
    # Calling abspath(args, kwargs) (line 110)
    abspath_call_result_26578 = invoke(stypy.reporting.localization.Localization(__file__, 110, 33), abspath_26575, *[d_26576], **kwargs_26577)
    
    # Processing the call keyword arguments (line 110)
    kwargs_26579 = {}
    # Getting the type of 'os' (line 110)
    os_26570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'os', False)
    # Obtaining the member 'path' of a type (line 110)
    path_26571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 16), os_26570, 'path')
    # Obtaining the member 'basename' of a type (line 110)
    basename_26572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 16), path_26571, 'basename')
    # Calling basename(args, kwargs) (line 110)
    basename_call_result_26580 = invoke(stypy.reporting.localization.Localization(__file__, 110, 16), basename_26572, *[abspath_call_result_26578], **kwargs_26579)
    
    # Assigning a type to the variable 'd' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'd', basename_call_result_26580)
    
    # Assigning a BinOp to a Name (line 111):
    
    # Assigning a BinOp to a Name (line 111):
    
    # Assigning a BinOp to a Name (line 111):
    # Getting the type of 'd' (line 111)
    d_26581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'd')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 111)
    i_26582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 28), 'i')
    slice_26583 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 111, 23), i_26582, None, None)
    # Getting the type of 'base' (line 111)
    base_26584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 23), 'base')
    # Obtaining the member '__getitem__' of a type (line 111)
    getitem___26585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 23), base_26584, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 111)
    subscript_call_result_26586 = invoke(stypy.reporting.localization.Localization(__file__, 111, 23), getitem___26585, slice_26583)
    
    # Applying the binary operator '+' (line 111)
    result_add_26587 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 19), '+', d_26581, subscript_call_result_26586)
    
    # Assigning a type to the variable 'base' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'base', result_add_26587)
    # SSA join for if statement (line 105)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'ext' (line 112)
    ext_26588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'ext')
    # Getting the type of 'self' (line 112)
    self_26589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 22), 'self')
    # Obtaining the member 'src_extensions' of a type (line 112)
    src_extensions_26590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 22), self_26589, 'src_extensions')
    # Applying the binary operator 'notin' (line 112)
    result_contains_26591 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 11), 'notin', ext_26588, src_extensions_26590)
    
    # Testing the type of an if condition (line 112)
    if_condition_26592 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 8), result_contains_26591)
    # Assigning a type to the variable 'if_condition_26592' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'if_condition_26592', if_condition_26592)
    # SSA begins for if statement (line 112)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to UnknownFileError(...): (line 113)
    # Processing the call arguments (line 113)
    str_26594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 35), 'str', "unknown file type '%s' (from '%s')")
    
    # Obtaining an instance of the builtin type 'tuple' (line 113)
    tuple_26595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 75), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 113)
    # Adding element type (line 113)
    # Getting the type of 'ext' (line 113)
    ext_26596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 75), 'ext', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 75), tuple_26595, ext_26596)
    # Adding element type (line 113)
    # Getting the type of 'src_name' (line 113)
    src_name_26597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 80), 'src_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 75), tuple_26595, src_name_26597)
    
    # Applying the binary operator '%' (line 113)
    result_mod_26598 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 35), '%', str_26594, tuple_26595)
    
    # Processing the call keyword arguments (line 113)
    kwargs_26599 = {}
    # Getting the type of 'UnknownFileError' (line 113)
    UnknownFileError_26593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 18), 'UnknownFileError', False)
    # Calling UnknownFileError(args, kwargs) (line 113)
    UnknownFileError_call_result_26600 = invoke(stypy.reporting.localization.Localization(__file__, 113, 18), UnknownFileError_26593, *[result_mod_26598], **kwargs_26599)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 113, 12), UnknownFileError_call_result_26600, 'raise parameter', BaseException)
    # SSA join for if statement (line 112)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'strip_dir' (line 114)
    strip_dir_26601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 11), 'strip_dir')
    # Testing the type of an if condition (line 114)
    if_condition_26602 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 114, 8), strip_dir_26601)
    # Assigning a type to the variable 'if_condition_26602' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'if_condition_26602', if_condition_26602)
    # SSA begins for if statement (line 114)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 115):
    
    # Assigning a Call to a Name (line 115):
    
    # Assigning a Call to a Name (line 115):
    
    # Call to basename(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'base' (line 115)
    base_26606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 36), 'base', False)
    # Processing the call keyword arguments (line 115)
    kwargs_26607 = {}
    # Getting the type of 'os' (line 115)
    os_26603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), 'os', False)
    # Obtaining the member 'path' of a type (line 115)
    path_26604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 19), os_26603, 'path')
    # Obtaining the member 'basename' of a type (line 115)
    basename_26605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 19), path_26604, 'basename')
    # Calling basename(args, kwargs) (line 115)
    basename_call_result_26608 = invoke(stypy.reporting.localization.Localization(__file__, 115, 19), basename_26605, *[base_26606], **kwargs_26607)
    
    # Assigning a type to the variable 'base' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'base', basename_call_result_26608)
    # SSA join for if statement (line 114)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 116):
    
    # Assigning a Call to a Name (line 116):
    
    # Assigning a Call to a Name (line 116):
    
    # Call to join(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'output_dir' (line 116)
    output_dir_26612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 32), 'output_dir', False)
    # Getting the type of 'base' (line 116)
    base_26613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 44), 'base', False)
    # Getting the type of 'self' (line 116)
    self_26614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 51), 'self', False)
    # Obtaining the member 'obj_extension' of a type (line 116)
    obj_extension_26615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 51), self_26614, 'obj_extension')
    # Applying the binary operator '+' (line 116)
    result_add_26616 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 44), '+', base_26613, obj_extension_26615)
    
    # Processing the call keyword arguments (line 116)
    kwargs_26617 = {}
    # Getting the type of 'os' (line 116)
    os_26609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 19), 'os', False)
    # Obtaining the member 'path' of a type (line 116)
    path_26610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 19), os_26609, 'path')
    # Obtaining the member 'join' of a type (line 116)
    join_26611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 19), path_26610, 'join')
    # Calling join(args, kwargs) (line 116)
    join_call_result_26618 = invoke(stypy.reporting.localization.Localization(__file__, 116, 19), join_26611, *[output_dir_26612, result_add_26616], **kwargs_26617)
    
    # Assigning a type to the variable 'obj_name' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'obj_name', join_call_result_26618)
    
    # Call to append(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'obj_name' (line 117)
    obj_name_26621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 25), 'obj_name', False)
    # Processing the call keyword arguments (line 117)
    kwargs_26622 = {}
    # Getting the type of 'obj_names' (line 117)
    obj_names_26619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'obj_names', False)
    # Obtaining the member 'append' of a type (line 117)
    append_26620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), obj_names_26619, 'append')
    # Calling append(args, kwargs) (line 117)
    append_call_result_26623 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), append_26620, *[obj_name_26621], **kwargs_26622)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'obj_names' (line 118)
    obj_names_26624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 11), 'obj_names')
    # Assigning a type to the variable 'stypy_return_type' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'stypy_return_type', obj_names_26624)
    
    # ################# End of 'CCompiler_object_filenames(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'CCompiler_object_filenames' in the type store
    # Getting the type of 'stypy_return_type' (line 75)
    stypy_return_type_26625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_26625)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'CCompiler_object_filenames'
    return stypy_return_type_26625

# Assigning a type to the variable 'CCompiler_object_filenames' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'CCompiler_object_filenames', CCompiler_object_filenames)

# Call to replace_method(...): (line 120)
# Processing the call arguments (line 120)
# Getting the type of 'CCompiler' (line 120)
CCompiler_26627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'CCompiler', False)
str_26628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 26), 'str', 'object_filenames')
# Getting the type of 'CCompiler_object_filenames' (line 120)
CCompiler_object_filenames_26629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 46), 'CCompiler_object_filenames', False)
# Processing the call keyword arguments (line 120)
kwargs_26630 = {}
# Getting the type of 'replace_method' (line 120)
replace_method_26626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 0), 'replace_method', False)
# Calling replace_method(args, kwargs) (line 120)
replace_method_call_result_26631 = invoke(stypy.reporting.localization.Localization(__file__, 120, 0), replace_method_26626, *[CCompiler_26627, str_26628, CCompiler_object_filenames_26629], **kwargs_26630)


@norecursion
def CCompiler_compile(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 122)
    None_26632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 48), 'None')
    # Getting the type of 'None' (line 122)
    None_26633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 61), 'None')
    # Getting the type of 'None' (line 123)
    None_26634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 35), 'None')
    int_26635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 47), 'int')
    # Getting the type of 'None' (line 123)
    None_26636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 64), 'None')
    # Getting the type of 'None' (line 124)
    None_26637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 37), 'None')
    # Getting the type of 'None' (line 124)
    None_26638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 51), 'None')
    defaults = [None_26632, None_26633, None_26634, int_26635, None_26636, None_26637, None_26638]
    # Create a new context for function 'CCompiler_compile'
    module_type_store = module_type_store.open_function_context('CCompiler_compile', 122, 0, False)
    
    # Passed parameters checking function
    CCompiler_compile.stypy_localization = localization
    CCompiler_compile.stypy_type_of_self = None
    CCompiler_compile.stypy_type_store = module_type_store
    CCompiler_compile.stypy_function_name = 'CCompiler_compile'
    CCompiler_compile.stypy_param_names_list = ['self', 'sources', 'output_dir', 'macros', 'include_dirs', 'debug', 'extra_preargs', 'extra_postargs', 'depends']
    CCompiler_compile.stypy_varargs_param_name = None
    CCompiler_compile.stypy_kwargs_param_name = None
    CCompiler_compile.stypy_call_defaults = defaults
    CCompiler_compile.stypy_call_varargs = varargs
    CCompiler_compile.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'CCompiler_compile', ['self', 'sources', 'output_dir', 'macros', 'include_dirs', 'debug', 'extra_preargs', 'extra_postargs', 'depends'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'CCompiler_compile', localization, ['self', 'sources', 'output_dir', 'macros', 'include_dirs', 'debug', 'extra_preargs', 'extra_postargs', 'depends'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'CCompiler_compile(...)' code ##################

    str_26639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, (-1)), 'str', '\n    Compile one or more source files.\n\n    Please refer to the Python distutils API reference for more details.\n\n    Parameters\n    ----------\n    sources : list of str\n        A list of filenames\n    output_dir : str, optional\n        Path to the output directory.\n    macros : list of tuples\n        A list of macro definitions.\n    include_dirs : list of str, optional\n        The directories to add to the default include file search path for\n        this compilation only.\n    debug : bool, optional\n        Whether or not to output debug symbols in or alongside the object\n        file(s).\n    extra_preargs, extra_postargs : ?\n        Extra pre- and post-arguments.\n    depends : list of str, optional\n        A list of file names that all targets depend on.\n\n    Returns\n    -------\n    objects : list of str\n        A list of object file names, one per source file `sources`.\n\n    Raises\n    ------\n    CompileError\n        If compilation fails.\n\n    ')
    
    
    # Getting the type of 'sources' (line 163)
    sources_26640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 11), 'sources')
    # Applying the 'not' unary operator (line 163)
    result_not__26641 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 7), 'not', sources_26640)
    
    # Testing the type of an if condition (line 163)
    if_condition_26642 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 4), result_not__26641)
    # Assigning a type to the variable 'if_condition_26642' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'if_condition_26642', if_condition_26642)
    # SSA begins for if statement (line 163)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'list' (line 164)
    list_26643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 164)
    
    # Assigning a type to the variable 'stypy_return_type' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'stypy_return_type', list_26643)
    # SSA join for if statement (line 163)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_26644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 24), 'int')
    # Getting the type of 'sys' (line 166)
    sys_26645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 7), 'sys')
    # Obtaining the member 'version_info' of a type (line 166)
    version_info_26646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 7), sys_26645, 'version_info')
    # Obtaining the member '__getitem__' of a type (line 166)
    getitem___26647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 7), version_info_26646, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 166)
    subscript_call_result_26648 = invoke(stypy.reporting.localization.Localization(__file__, 166, 7), getitem___26647, int_26644)
    
    int_26649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 29), 'int')
    # Applying the binary operator '<' (line 166)
    result_lt_26650 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 7), '<', subscript_call_result_26648, int_26649)
    
    # Testing the type of an if condition (line 166)
    if_condition_26651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 166, 4), result_lt_26650)
    # Assigning a type to the variable 'if_condition_26651' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'if_condition_26651', if_condition_26651)
    # SSA begins for if statement (line 166)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 167, 8))
    
    # 'from numpy.distutils.fcompiler import FCompiler, is_f_file, has_f90_header' statement (line 167)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
    import_26652 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 167, 8), 'numpy.distutils.fcompiler')

    if (type(import_26652) is not StypyTypeError):

        if (import_26652 != 'pyd_module'):
            __import__(import_26652)
            sys_modules_26653 = sys.modules[import_26652]
            import_from_module(stypy.reporting.localization.Localization(__file__, 167, 8), 'numpy.distutils.fcompiler', sys_modules_26653.module_type_store, module_type_store, ['FCompiler', 'is_f_file', 'has_f90_header'])
            nest_module(stypy.reporting.localization.Localization(__file__, 167, 8), __file__, sys_modules_26653, sys_modules_26653.module_type_store, module_type_store)
        else:
            from numpy.distutils.fcompiler import FCompiler, is_f_file, has_f90_header

            import_from_module(stypy.reporting.localization.Localization(__file__, 167, 8), 'numpy.distutils.fcompiler', None, module_type_store, ['FCompiler', 'is_f_file', 'has_f90_header'], [FCompiler, is_f_file, has_f90_header])

    else:
        # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'numpy.distutils.fcompiler', import_26652)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')
    
    # SSA branch for the else part of an if statement (line 166)
    module_type_store.open_ssa_branch('else')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 169, 8))
    
    # 'from numpy.distutils.fcompiler import FCompiler, is_f_file, has_f90_header' statement (line 169)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
    import_26654 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 169, 8), 'numpy.distutils.fcompiler')

    if (type(import_26654) is not StypyTypeError):

        if (import_26654 != 'pyd_module'):
            __import__(import_26654)
            sys_modules_26655 = sys.modules[import_26654]
            import_from_module(stypy.reporting.localization.Localization(__file__, 169, 8), 'numpy.distutils.fcompiler', sys_modules_26655.module_type_store, module_type_store, ['FCompiler', 'is_f_file', 'has_f90_header'])
            nest_module(stypy.reporting.localization.Localization(__file__, 169, 8), __file__, sys_modules_26655, sys_modules_26655.module_type_store, module_type_store)
        else:
            from numpy.distutils.fcompiler import FCompiler, is_f_file, has_f90_header

            import_from_module(stypy.reporting.localization.Localization(__file__, 169, 8), 'numpy.distutils.fcompiler', None, module_type_store, ['FCompiler', 'is_f_file', 'has_f90_header'], [FCompiler, is_f_file, has_f90_header])

    else:
        # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'numpy.distutils.fcompiler', import_26654)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')
    
    # SSA join for if statement (line 166)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isinstance(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'self' (line 171)
    self_26657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 18), 'self', False)
    # Getting the type of 'FCompiler' (line 171)
    FCompiler_26658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 24), 'FCompiler', False)
    # Processing the call keyword arguments (line 171)
    kwargs_26659 = {}
    # Getting the type of 'isinstance' (line 171)
    isinstance_26656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 171)
    isinstance_call_result_26660 = invoke(stypy.reporting.localization.Localization(__file__, 171, 7), isinstance_26656, *[self_26657, FCompiler_26658], **kwargs_26659)
    
    # Testing the type of an if condition (line 171)
    if_condition_26661 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 4), isinstance_call_result_26660)
    # Assigning a type to the variable 'if_condition_26661' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'if_condition_26661', if_condition_26661)
    # SSA begins for if statement (line 171)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 172):
    
    # Assigning a List to a Name (line 172):
    
    # Assigning a List to a Name (line 172):
    
    # Obtaining an instance of the builtin type 'list' (line 172)
    list_26662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 172)
    
    # Assigning a type to the variable 'display' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'display', list_26662)
    
    
    # Obtaining an instance of the builtin type 'list' (line 173)
    list_26663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 173)
    # Adding element type (line 173)
    str_26664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 19), 'str', 'f77')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 18), list_26663, str_26664)
    # Adding element type (line 173)
    str_26665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 26), 'str', 'f90')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 18), list_26663, str_26665)
    # Adding element type (line 173)
    str_26666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 33), 'str', 'fix')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 18), list_26663, str_26666)
    
    # Testing the type of a for loop iterable (line 173)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 173, 8), list_26663)
    # Getting the type of the for loop variable (line 173)
    for_loop_var_26667 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 173, 8), list_26663)
    # Assigning a type to the variable 'fc' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'fc', for_loop_var_26667)
    # SSA begins for a for statement (line 173)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 174):
    
    # Assigning a Call to a Name (line 174):
    
    # Assigning a Call to a Name (line 174):
    
    # Call to getattr(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'self' (line 174)
    self_26669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 28), 'self', False)
    str_26670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 34), 'str', 'compiler_')
    # Getting the type of 'fc' (line 174)
    fc_26671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 46), 'fc', False)
    # Applying the binary operator '+' (line 174)
    result_add_26672 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 34), '+', str_26670, fc_26671)
    
    # Processing the call keyword arguments (line 174)
    kwargs_26673 = {}
    # Getting the type of 'getattr' (line 174)
    getattr_26668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 20), 'getattr', False)
    # Calling getattr(args, kwargs) (line 174)
    getattr_call_result_26674 = invoke(stypy.reporting.localization.Localization(__file__, 174, 20), getattr_26668, *[self_26669, result_add_26672], **kwargs_26673)
    
    # Assigning a type to the variable 'fcomp' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'fcomp', getattr_call_result_26674)
    
    # Type idiom detected: calculating its left and rigth part (line 175)
    # Getting the type of 'fcomp' (line 175)
    fcomp_26675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 15), 'fcomp')
    # Getting the type of 'None' (line 175)
    None_26676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 24), 'None')
    
    (may_be_26677, more_types_in_union_26678) = may_be_none(fcomp_26675, None_26676)

    if may_be_26677:

        if more_types_in_union_26678:
            # Runtime conditional SSA (line 175)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        if more_types_in_union_26678:
            # SSA join for if statement (line 175)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to append(...): (line 177)
    # Processing the call arguments (line 177)
    str_26681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 27), 'str', 'Fortran %s compiler: %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 177)
    tuple_26682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 56), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 177)
    # Adding element type (line 177)
    # Getting the type of 'fc' (line 177)
    fc_26683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 56), 'fc', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 56), tuple_26682, fc_26683)
    # Adding element type (line 177)
    
    # Call to join(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'fcomp' (line 177)
    fcomp_26686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 69), 'fcomp', False)
    # Processing the call keyword arguments (line 177)
    kwargs_26687 = {}
    str_26684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 60), 'str', ' ')
    # Obtaining the member 'join' of a type (line 177)
    join_26685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 60), str_26684, 'join')
    # Calling join(args, kwargs) (line 177)
    join_call_result_26688 = invoke(stypy.reporting.localization.Localization(__file__, 177, 60), join_26685, *[fcomp_26686], **kwargs_26687)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 56), tuple_26682, join_call_result_26688)
    
    # Applying the binary operator '%' (line 177)
    result_mod_26689 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 27), '%', str_26681, tuple_26682)
    
    # Processing the call keyword arguments (line 177)
    kwargs_26690 = {}
    # Getting the type of 'display' (line 177)
    display_26679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'display', False)
    # Obtaining the member 'append' of a type (line 177)
    append_26680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 12), display_26679, 'append')
    # Calling append(args, kwargs) (line 177)
    append_call_result_26691 = invoke(stypy.reporting.localization.Localization(__file__, 177, 12), append_26680, *[result_mod_26689], **kwargs_26690)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 178):
    
    # Assigning a Call to a Name (line 178):
    
    # Assigning a Call to a Name (line 178):
    
    # Call to join(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'display' (line 178)
    display_26694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 28), 'display', False)
    # Processing the call keyword arguments (line 178)
    kwargs_26695 = {}
    str_26692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 18), 'str', '\n')
    # Obtaining the member 'join' of a type (line 178)
    join_26693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 18), str_26692, 'join')
    # Calling join(args, kwargs) (line 178)
    join_call_result_26696 = invoke(stypy.reporting.localization.Localization(__file__, 178, 18), join_26693, *[display_26694], **kwargs_26695)
    
    # Assigning a type to the variable 'display' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'display', join_call_result_26696)
    # SSA branch for the else part of an if statement (line 171)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Attribute to a Name (line 180):
    
    # Assigning a Attribute to a Name (line 180):
    
    # Assigning a Attribute to a Name (line 180):
    # Getting the type of 'self' (line 180)
    self_26697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 16), 'self')
    # Obtaining the member 'compiler_so' of a type (line 180)
    compiler_so_26698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 16), self_26697, 'compiler_so')
    # Assigning a type to the variable 'ccomp' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'ccomp', compiler_so_26698)
    
    # Assigning a BinOp to a Name (line 181):
    
    # Assigning a BinOp to a Name (line 181):
    
    # Assigning a BinOp to a Name (line 181):
    str_26699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 18), 'str', 'C compiler: %s\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 181)
    tuple_26700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 181)
    # Adding element type (line 181)
    
    # Call to join(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'ccomp' (line 181)
    ccomp_26703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 49), 'ccomp', False)
    # Processing the call keyword arguments (line 181)
    kwargs_26704 = {}
    str_26701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 40), 'str', ' ')
    # Obtaining the member 'join' of a type (line 181)
    join_26702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 40), str_26701, 'join')
    # Calling join(args, kwargs) (line 181)
    join_call_result_26705 = invoke(stypy.reporting.localization.Localization(__file__, 181, 40), join_26702, *[ccomp_26703], **kwargs_26704)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 40), tuple_26700, join_call_result_26705)
    
    # Applying the binary operator '%' (line 181)
    result_mod_26706 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 18), '%', str_26699, tuple_26700)
    
    # Assigning a type to the variable 'display' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'display', result_mod_26706)
    # SSA join for if statement (line 171)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to info(...): (line 182)
    # Processing the call arguments (line 182)
    # Getting the type of 'display' (line 182)
    display_26709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 13), 'display', False)
    # Processing the call keyword arguments (line 182)
    kwargs_26710 = {}
    # Getting the type of 'log' (line 182)
    log_26707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'log', False)
    # Obtaining the member 'info' of a type (line 182)
    info_26708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 4), log_26707, 'info')
    # Calling info(args, kwargs) (line 182)
    info_call_result_26711 = invoke(stypy.reporting.localization.Localization(__file__, 182, 4), info_26708, *[display_26709], **kwargs_26710)
    
    
    # Assigning a Call to a Tuple (line 183):
    
    # Assigning a Call to a Name:
    
    # Assigning a Call to a Name:
    
    # Call to _setup_compile(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'output_dir' (line 184)
    output_dir_26714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 32), 'output_dir', False)
    # Getting the type of 'macros' (line 184)
    macros_26715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 44), 'macros', False)
    # Getting the type of 'include_dirs' (line 184)
    include_dirs_26716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 52), 'include_dirs', False)
    # Getting the type of 'sources' (line 184)
    sources_26717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 66), 'sources', False)
    # Getting the type of 'depends' (line 185)
    depends_26718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 32), 'depends', False)
    # Getting the type of 'extra_postargs' (line 185)
    extra_postargs_26719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 41), 'extra_postargs', False)
    # Processing the call keyword arguments (line 184)
    kwargs_26720 = {}
    # Getting the type of 'self' (line 184)
    self_26712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'self', False)
    # Obtaining the member '_setup_compile' of a type (line 184)
    _setup_compile_26713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 12), self_26712, '_setup_compile')
    # Calling _setup_compile(args, kwargs) (line 184)
    _setup_compile_call_result_26721 = invoke(stypy.reporting.localization.Localization(__file__, 184, 12), _setup_compile_26713, *[output_dir_26714, macros_26715, include_dirs_26716, sources_26717, depends_26718, extra_postargs_26719], **kwargs_26720)
    
    # Assigning a type to the variable 'call_assignment_26341' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'call_assignment_26341', _setup_compile_call_result_26721)
    
    # Assigning a Call to a Name (line 183):
    
    # Assigning a Call to a Name (line 183):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_26724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 4), 'int')
    # Processing the call keyword arguments
    kwargs_26725 = {}
    # Getting the type of 'call_assignment_26341' (line 183)
    call_assignment_26341_26722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'call_assignment_26341', False)
    # Obtaining the member '__getitem__' of a type (line 183)
    getitem___26723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 4), call_assignment_26341_26722, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_26726 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___26723, *[int_26724], **kwargs_26725)
    
    # Assigning a type to the variable 'call_assignment_26342' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'call_assignment_26342', getitem___call_result_26726)
    
    # Assigning a Name to a Name (line 183):
    
    # Assigning a Name to a Name (line 183):
    # Getting the type of 'call_assignment_26342' (line 183)
    call_assignment_26342_26727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'call_assignment_26342')
    # Assigning a type to the variable 'macros' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'macros', call_assignment_26342_26727)
    
    # Assigning a Call to a Name (line 183):
    
    # Assigning a Call to a Name (line 183):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_26730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 4), 'int')
    # Processing the call keyword arguments
    kwargs_26731 = {}
    # Getting the type of 'call_assignment_26341' (line 183)
    call_assignment_26341_26728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'call_assignment_26341', False)
    # Obtaining the member '__getitem__' of a type (line 183)
    getitem___26729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 4), call_assignment_26341_26728, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_26732 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___26729, *[int_26730], **kwargs_26731)
    
    # Assigning a type to the variable 'call_assignment_26343' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'call_assignment_26343', getitem___call_result_26732)
    
    # Assigning a Name to a Name (line 183):
    
    # Assigning a Name to a Name (line 183):
    # Getting the type of 'call_assignment_26343' (line 183)
    call_assignment_26343_26733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'call_assignment_26343')
    # Assigning a type to the variable 'objects' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'objects', call_assignment_26343_26733)
    
    # Assigning a Call to a Name (line 183):
    
    # Assigning a Call to a Name (line 183):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_26736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 4), 'int')
    # Processing the call keyword arguments
    kwargs_26737 = {}
    # Getting the type of 'call_assignment_26341' (line 183)
    call_assignment_26341_26734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'call_assignment_26341', False)
    # Obtaining the member '__getitem__' of a type (line 183)
    getitem___26735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 4), call_assignment_26341_26734, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_26738 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___26735, *[int_26736], **kwargs_26737)
    
    # Assigning a type to the variable 'call_assignment_26344' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'call_assignment_26344', getitem___call_result_26738)
    
    # Assigning a Name to a Name (line 183):
    
    # Assigning a Name to a Name (line 183):
    # Getting the type of 'call_assignment_26344' (line 183)
    call_assignment_26344_26739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'call_assignment_26344')
    # Assigning a type to the variable 'extra_postargs' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 21), 'extra_postargs', call_assignment_26344_26739)
    
    # Assigning a Call to a Name (line 183):
    
    # Assigning a Call to a Name (line 183):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_26742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 4), 'int')
    # Processing the call keyword arguments
    kwargs_26743 = {}
    # Getting the type of 'call_assignment_26341' (line 183)
    call_assignment_26341_26740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'call_assignment_26341', False)
    # Obtaining the member '__getitem__' of a type (line 183)
    getitem___26741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 4), call_assignment_26341_26740, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_26744 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___26741, *[int_26742], **kwargs_26743)
    
    # Assigning a type to the variable 'call_assignment_26345' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'call_assignment_26345', getitem___call_result_26744)
    
    # Assigning a Name to a Name (line 183):
    
    # Assigning a Name to a Name (line 183):
    # Getting the type of 'call_assignment_26345' (line 183)
    call_assignment_26345_26745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'call_assignment_26345')
    # Assigning a type to the variable 'pp_opts' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 37), 'pp_opts', call_assignment_26345_26745)
    
    # Assigning a Call to a Name (line 183):
    
    # Assigning a Call to a Name (line 183):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_26748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 4), 'int')
    # Processing the call keyword arguments
    kwargs_26749 = {}
    # Getting the type of 'call_assignment_26341' (line 183)
    call_assignment_26341_26746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'call_assignment_26341', False)
    # Obtaining the member '__getitem__' of a type (line 183)
    getitem___26747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 4), call_assignment_26341_26746, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_26750 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___26747, *[int_26748], **kwargs_26749)
    
    # Assigning a type to the variable 'call_assignment_26346' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'call_assignment_26346', getitem___call_result_26750)
    
    # Assigning a Name to a Name (line 183):
    
    # Assigning a Name to a Name (line 183):
    # Getting the type of 'call_assignment_26346' (line 183)
    call_assignment_26346_26751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'call_assignment_26346')
    # Assigning a type to the variable 'build' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 46), 'build', call_assignment_26346_26751)
    
    # Assigning a Call to a Name (line 186):
    
    # Assigning a Call to a Name (line 186):
    
    # Assigning a Call to a Name (line 186):
    
    # Call to _get_cc_args(...): (line 186)
    # Processing the call arguments (line 186)
    # Getting the type of 'pp_opts' (line 186)
    pp_opts_26754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 32), 'pp_opts', False)
    # Getting the type of 'debug' (line 186)
    debug_26755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 41), 'debug', False)
    # Getting the type of 'extra_preargs' (line 186)
    extra_preargs_26756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 48), 'extra_preargs', False)
    # Processing the call keyword arguments (line 186)
    kwargs_26757 = {}
    # Getting the type of 'self' (line 186)
    self_26752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 14), 'self', False)
    # Obtaining the member '_get_cc_args' of a type (line 186)
    _get_cc_args_26753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 14), self_26752, '_get_cc_args')
    # Calling _get_cc_args(args, kwargs) (line 186)
    _get_cc_args_call_result_26758 = invoke(stypy.reporting.localization.Localization(__file__, 186, 14), _get_cc_args_26753, *[pp_opts_26754, debug_26755, extra_preargs_26756], **kwargs_26757)
    
    # Assigning a type to the variable 'cc_args' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'cc_args', _get_cc_args_call_result_26758)
    
    # Assigning a BinOp to a Name (line 187):
    
    # Assigning a BinOp to a Name (line 187):
    
    # Assigning a BinOp to a Name (line 187):
    str_26759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 14), 'str', "compile options: '%s'")
    
    # Call to join(...): (line 187)
    # Processing the call arguments (line 187)
    # Getting the type of 'cc_args' (line 187)
    cc_args_26762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 50), 'cc_args', False)
    # Processing the call keyword arguments (line 187)
    kwargs_26763 = {}
    str_26760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 41), 'str', ' ')
    # Obtaining the member 'join' of a type (line 187)
    join_26761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 41), str_26760, 'join')
    # Calling join(args, kwargs) (line 187)
    join_call_result_26764 = invoke(stypy.reporting.localization.Localization(__file__, 187, 41), join_26761, *[cc_args_26762], **kwargs_26763)
    
    # Applying the binary operator '%' (line 187)
    result_mod_26765 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 14), '%', str_26759, join_call_result_26764)
    
    # Assigning a type to the variable 'display' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'display', result_mod_26765)
    
    # Getting the type of 'extra_postargs' (line 188)
    extra_postargs_26766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 7), 'extra_postargs')
    # Testing the type of an if condition (line 188)
    if_condition_26767 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 188, 4), extra_postargs_26766)
    # Assigning a type to the variable 'if_condition_26767' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'if_condition_26767', if_condition_26767)
    # SSA begins for if statement (line 188)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'display' (line 189)
    display_26768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'display')
    str_26769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 19), 'str', "\nextra options: '%s'")
    
    # Call to join(...): (line 189)
    # Processing the call arguments (line 189)
    # Getting the type of 'extra_postargs' (line 189)
    extra_postargs_26772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 55), 'extra_postargs', False)
    # Processing the call keyword arguments (line 189)
    kwargs_26773 = {}
    str_26770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 46), 'str', ' ')
    # Obtaining the member 'join' of a type (line 189)
    join_26771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 46), str_26770, 'join')
    # Calling join(args, kwargs) (line 189)
    join_call_result_26774 = invoke(stypy.reporting.localization.Localization(__file__, 189, 46), join_26771, *[extra_postargs_26772], **kwargs_26773)
    
    # Applying the binary operator '%' (line 189)
    result_mod_26775 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 19), '%', str_26769, join_call_result_26774)
    
    # Applying the binary operator '+=' (line 189)
    result_iadd_26776 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 8), '+=', display_26768, result_mod_26775)
    # Assigning a type to the variable 'display' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'display', result_iadd_26776)
    
    # SSA join for if statement (line 188)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to info(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'display' (line 190)
    display_26779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 13), 'display', False)
    # Processing the call keyword arguments (line 190)
    kwargs_26780 = {}
    # Getting the type of 'log' (line 190)
    log_26777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'log', False)
    # Obtaining the member 'info' of a type (line 190)
    info_26778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 4), log_26777, 'info')
    # Calling info(args, kwargs) (line 190)
    info_call_result_26781 = invoke(stypy.reporting.localization.Localization(__file__, 190, 4), info_26778, *[display_26779], **kwargs_26780)
    

    @norecursion
    def single_compile(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'single_compile'
        module_type_store = module_type_store.open_function_context('single_compile', 192, 4, False)
        
        # Passed parameters checking function
        single_compile.stypy_localization = localization
        single_compile.stypy_type_of_self = None
        single_compile.stypy_type_store = module_type_store
        single_compile.stypy_function_name = 'single_compile'
        single_compile.stypy_param_names_list = ['args']
        single_compile.stypy_varargs_param_name = None
        single_compile.stypy_kwargs_param_name = None
        single_compile.stypy_call_defaults = defaults
        single_compile.stypy_call_varargs = varargs
        single_compile.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'single_compile', ['args'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'single_compile', localization, ['args'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'single_compile(...)' code ##################

        
        # Assigning a Name to a Tuple (line 193):
        
        # Assigning a Subscript to a Name (line 193):
        
        # Assigning a Subscript to a Name (line 193):
        
        # Obtaining the type of the subscript
        int_26782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 8), 'int')
        # Getting the type of 'args' (line 193)
        args_26783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 26), 'args')
        # Obtaining the member '__getitem__' of a type (line 193)
        getitem___26784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 8), args_26783, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 193)
        subscript_call_result_26785 = invoke(stypy.reporting.localization.Localization(__file__, 193, 8), getitem___26784, int_26782)
        
        # Assigning a type to the variable 'tuple_var_assignment_26347' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'tuple_var_assignment_26347', subscript_call_result_26785)
        
        # Assigning a Subscript to a Name (line 193):
        
        # Assigning a Subscript to a Name (line 193):
        
        # Obtaining the type of the subscript
        int_26786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 8), 'int')
        # Getting the type of 'args' (line 193)
        args_26787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 26), 'args')
        # Obtaining the member '__getitem__' of a type (line 193)
        getitem___26788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 8), args_26787, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 193)
        subscript_call_result_26789 = invoke(stypy.reporting.localization.Localization(__file__, 193, 8), getitem___26788, int_26786)
        
        # Assigning a type to the variable 'tuple_var_assignment_26348' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'tuple_var_assignment_26348', subscript_call_result_26789)
        
        # Assigning a Name to a Name (line 193):
        
        # Assigning a Name to a Name (line 193):
        # Getting the type of 'tuple_var_assignment_26347' (line 193)
        tuple_var_assignment_26347_26790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'tuple_var_assignment_26347')
        # Assigning a type to the variable 'obj' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'obj', tuple_var_assignment_26347_26790)
        
        # Assigning a Name to a Tuple (line 193):
        
        # Assigning a Subscript to a Name (line 193):
        
        # Obtaining the type of the subscript
        int_26791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 8), 'int')
        # Getting the type of 'tuple_var_assignment_26348' (line 193)
        tuple_var_assignment_26348_26792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'tuple_var_assignment_26348')
        # Obtaining the member '__getitem__' of a type (line 193)
        getitem___26793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 8), tuple_var_assignment_26348_26792, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 193)
        subscript_call_result_26794 = invoke(stypy.reporting.localization.Localization(__file__, 193, 8), getitem___26793, int_26791)
        
        # Assigning a type to the variable 'tuple_var_assignment_26366' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'tuple_var_assignment_26366', subscript_call_result_26794)
        
        # Assigning a Subscript to a Name (line 193):
        
        # Obtaining the type of the subscript
        int_26795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 8), 'int')
        # Getting the type of 'tuple_var_assignment_26348' (line 193)
        tuple_var_assignment_26348_26796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'tuple_var_assignment_26348')
        # Obtaining the member '__getitem__' of a type (line 193)
        getitem___26797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 8), tuple_var_assignment_26348_26796, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 193)
        subscript_call_result_26798 = invoke(stypy.reporting.localization.Localization(__file__, 193, 8), getitem___26797, int_26795)
        
        # Assigning a type to the variable 'tuple_var_assignment_26367' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'tuple_var_assignment_26367', subscript_call_result_26798)
        
        # Assigning a Name to a Name (line 193):
        # Getting the type of 'tuple_var_assignment_26366' (line 193)
        tuple_var_assignment_26366_26799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'tuple_var_assignment_26366')
        # Assigning a type to the variable 'src' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 14), 'src', tuple_var_assignment_26366_26799)
        
        # Assigning a Name to a Name (line 193):
        # Getting the type of 'tuple_var_assignment_26367' (line 193)
        tuple_var_assignment_26367_26800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'tuple_var_assignment_26367')
        # Assigning a type to the variable 'ext' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 19), 'ext', tuple_var_assignment_26367_26800)
        
        # Call to _compile(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'obj' (line 194)
        obj_26803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 22), 'obj', False)
        # Getting the type of 'src' (line 194)
        src_26804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 27), 'src', False)
        # Getting the type of 'ext' (line 194)
        ext_26805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 32), 'ext', False)
        # Getting the type of 'cc_args' (line 194)
        cc_args_26806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 37), 'cc_args', False)
        # Getting the type of 'extra_postargs' (line 194)
        extra_postargs_26807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 46), 'extra_postargs', False)
        # Getting the type of 'pp_opts' (line 194)
        pp_opts_26808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 62), 'pp_opts', False)
        # Processing the call keyword arguments (line 194)
        kwargs_26809 = {}
        # Getting the type of 'self' (line 194)
        self_26801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'self', False)
        # Obtaining the member '_compile' of a type (line 194)
        _compile_26802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 8), self_26801, '_compile')
        # Calling _compile(args, kwargs) (line 194)
        _compile_call_result_26810 = invoke(stypy.reporting.localization.Localization(__file__, 194, 8), _compile_26802, *[obj_26803, src_26804, ext_26805, cc_args_26806, extra_postargs_26807, pp_opts_26808], **kwargs_26809)
        
        
        # ################# End of 'single_compile(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'single_compile' in the type store
        # Getting the type of 'stypy_return_type' (line 192)
        stypy_return_type_26811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26811)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'single_compile'
        return stypy_return_type_26811

    # Assigning a type to the variable 'single_compile' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'single_compile', single_compile)
    
    
    # Call to isinstance(...): (line 196)
    # Processing the call arguments (line 196)
    # Getting the type of 'self' (line 196)
    self_26813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 18), 'self', False)
    # Getting the type of 'FCompiler' (line 196)
    FCompiler_26814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 24), 'FCompiler', False)
    # Processing the call keyword arguments (line 196)
    kwargs_26815 = {}
    # Getting the type of 'isinstance' (line 196)
    isinstance_26812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 196)
    isinstance_call_result_26816 = invoke(stypy.reporting.localization.Localization(__file__, 196, 7), isinstance_26812, *[self_26813, FCompiler_26814], **kwargs_26815)
    
    # Testing the type of an if condition (line 196)
    if_condition_26817 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 196, 4), isinstance_call_result_26816)
    # Assigning a type to the variable 'if_condition_26817' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'if_condition_26817', if_condition_26817)
    # SSA begins for if statement (line 196)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 197):
    
    # Assigning a Call to a Name (line 197):
    
    # Assigning a Call to a Name (line 197):
    
    # Call to list(...): (line 197)
    # Processing the call arguments (line 197)
    
    # Call to keys(...): (line 197)
    # Processing the call keyword arguments (line 197)
    kwargs_26821 = {}
    # Getting the type of 'build' (line 197)
    build_26819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 32), 'build', False)
    # Obtaining the member 'keys' of a type (line 197)
    keys_26820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 32), build_26819, 'keys')
    # Calling keys(args, kwargs) (line 197)
    keys_call_result_26822 = invoke(stypy.reporting.localization.Localization(__file__, 197, 32), keys_26820, *[], **kwargs_26821)
    
    # Processing the call keyword arguments (line 197)
    kwargs_26823 = {}
    # Getting the type of 'list' (line 197)
    list_26818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 27), 'list', False)
    # Calling list(args, kwargs) (line 197)
    list_call_result_26824 = invoke(stypy.reporting.localization.Localization(__file__, 197, 27), list_26818, *[keys_call_result_26822], **kwargs_26823)
    
    # Assigning a type to the variable 'objects_to_build' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'objects_to_build', list_call_result_26824)
    
    # Assigning a Tuple to a Tuple (line 198):
    
    # Assigning a List to a Name (line 198):
    
    # Assigning a List to a Name (line 198):
    
    # Obtaining an instance of the builtin type 'list' (line 198)
    list_26825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 37), 'list')
    # Adding type elements to the builtin type 'list' instance (line 198)
    
    # Assigning a type to the variable 'tuple_assignment_26349' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'tuple_assignment_26349', list_26825)
    
    # Assigning a List to a Name (line 198):
    
    # Assigning a List to a Name (line 198):
    
    # Obtaining an instance of the builtin type 'list' (line 198)
    list_26826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 41), 'list')
    # Adding type elements to the builtin type 'list' instance (line 198)
    
    # Assigning a type to the variable 'tuple_assignment_26350' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'tuple_assignment_26350', list_26826)
    
    # Assigning a Name to a Name (line 198):
    
    # Assigning a Name to a Name (line 198):
    # Getting the type of 'tuple_assignment_26349' (line 198)
    tuple_assignment_26349_26827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'tuple_assignment_26349')
    # Assigning a type to the variable 'f77_objects' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'f77_objects', tuple_assignment_26349_26827)
    
    # Assigning a Name to a Name (line 198):
    
    # Assigning a Name to a Name (line 198):
    # Getting the type of 'tuple_assignment_26350' (line 198)
    tuple_assignment_26350_26828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'tuple_assignment_26350')
    # Assigning a type to the variable 'other_objects' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 21), 'other_objects', tuple_assignment_26350_26828)
    
    # Getting the type of 'objects' (line 199)
    objects_26829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 19), 'objects')
    # Testing the type of a for loop iterable (line 199)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 199, 8), objects_26829)
    # Getting the type of the for loop variable (line 199)
    for_loop_var_26830 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 199, 8), objects_26829)
    # Assigning a type to the variable 'obj' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'obj', for_loop_var_26830)
    # SSA begins for a for statement (line 199)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'obj' (line 200)
    obj_26831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 15), 'obj')
    # Getting the type of 'objects_to_build' (line 200)
    objects_to_build_26832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 22), 'objects_to_build')
    # Applying the binary operator 'in' (line 200)
    result_contains_26833 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 15), 'in', obj_26831, objects_to_build_26832)
    
    # Testing the type of an if condition (line 200)
    if_condition_26834 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 12), result_contains_26833)
    # Assigning a type to the variable 'if_condition_26834' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'if_condition_26834', if_condition_26834)
    # SSA begins for if statement (line 200)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Tuple (line 201):
    
    # Assigning a Subscript to a Name (line 201):
    
    # Assigning a Subscript to a Name (line 201):
    
    # Obtaining the type of the subscript
    int_26835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 16), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'obj' (line 201)
    obj_26836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 33), 'obj')
    # Getting the type of 'build' (line 201)
    build_26837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 27), 'build')
    # Obtaining the member '__getitem__' of a type (line 201)
    getitem___26838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 27), build_26837, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 201)
    subscript_call_result_26839 = invoke(stypy.reporting.localization.Localization(__file__, 201, 27), getitem___26838, obj_26836)
    
    # Obtaining the member '__getitem__' of a type (line 201)
    getitem___26840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 16), subscript_call_result_26839, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 201)
    subscript_call_result_26841 = invoke(stypy.reporting.localization.Localization(__file__, 201, 16), getitem___26840, int_26835)
    
    # Assigning a type to the variable 'tuple_var_assignment_26351' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'tuple_var_assignment_26351', subscript_call_result_26841)
    
    # Assigning a Subscript to a Name (line 201):
    
    # Assigning a Subscript to a Name (line 201):
    
    # Obtaining the type of the subscript
    int_26842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 16), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'obj' (line 201)
    obj_26843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 33), 'obj')
    # Getting the type of 'build' (line 201)
    build_26844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 27), 'build')
    # Obtaining the member '__getitem__' of a type (line 201)
    getitem___26845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 27), build_26844, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 201)
    subscript_call_result_26846 = invoke(stypy.reporting.localization.Localization(__file__, 201, 27), getitem___26845, obj_26843)
    
    # Obtaining the member '__getitem__' of a type (line 201)
    getitem___26847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 16), subscript_call_result_26846, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 201)
    subscript_call_result_26848 = invoke(stypy.reporting.localization.Localization(__file__, 201, 16), getitem___26847, int_26842)
    
    # Assigning a type to the variable 'tuple_var_assignment_26352' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'tuple_var_assignment_26352', subscript_call_result_26848)
    
    # Assigning a Name to a Name (line 201):
    
    # Assigning a Name to a Name (line 201):
    # Getting the type of 'tuple_var_assignment_26351' (line 201)
    tuple_var_assignment_26351_26849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'tuple_var_assignment_26351')
    # Assigning a type to the variable 'src' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'src', tuple_var_assignment_26351_26849)
    
    # Assigning a Name to a Name (line 201):
    
    # Assigning a Name to a Name (line 201):
    # Getting the type of 'tuple_var_assignment_26352' (line 201)
    tuple_var_assignment_26352_26850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'tuple_var_assignment_26352')
    # Assigning a type to the variable 'ext' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 21), 'ext', tuple_var_assignment_26352_26850)
    
    
    # Getting the type of 'self' (line 202)
    self_26851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 19), 'self')
    # Obtaining the member 'compiler_type' of a type (line 202)
    compiler_type_26852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 19), self_26851, 'compiler_type')
    str_26853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 39), 'str', 'absoft')
    # Applying the binary operator '==' (line 202)
    result_eq_26854 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 19), '==', compiler_type_26852, str_26853)
    
    # Testing the type of an if condition (line 202)
    if_condition_26855 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 202, 16), result_eq_26854)
    # Assigning a type to the variable 'if_condition_26855' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'if_condition_26855', if_condition_26855)
    # SSA begins for if statement (line 202)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 203):
    
    # Assigning a Call to a Name (line 203):
    
    # Assigning a Call to a Name (line 203):
    
    # Call to cyg2win32(...): (line 203)
    # Processing the call arguments (line 203)
    # Getting the type of 'obj' (line 203)
    obj_26857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 36), 'obj', False)
    # Processing the call keyword arguments (line 203)
    kwargs_26858 = {}
    # Getting the type of 'cyg2win32' (line 203)
    cyg2win32_26856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 26), 'cyg2win32', False)
    # Calling cyg2win32(args, kwargs) (line 203)
    cyg2win32_call_result_26859 = invoke(stypy.reporting.localization.Localization(__file__, 203, 26), cyg2win32_26856, *[obj_26857], **kwargs_26858)
    
    # Assigning a type to the variable 'obj' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 20), 'obj', cyg2win32_call_result_26859)
    
    # Assigning a Call to a Name (line 204):
    
    # Assigning a Call to a Name (line 204):
    
    # Assigning a Call to a Name (line 204):
    
    # Call to cyg2win32(...): (line 204)
    # Processing the call arguments (line 204)
    # Getting the type of 'src' (line 204)
    src_26861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 36), 'src', False)
    # Processing the call keyword arguments (line 204)
    kwargs_26862 = {}
    # Getting the type of 'cyg2win32' (line 204)
    cyg2win32_26860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 26), 'cyg2win32', False)
    # Calling cyg2win32(args, kwargs) (line 204)
    cyg2win32_call_result_26863 = invoke(stypy.reporting.localization.Localization(__file__, 204, 26), cyg2win32_26860, *[src_26861], **kwargs_26862)
    
    # Assigning a type to the variable 'src' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 20), 'src', cyg2win32_call_result_26863)
    # SSA join for if statement (line 202)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Call to is_f_file(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'src' (line 205)
    src_26865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 29), 'src', False)
    # Processing the call keyword arguments (line 205)
    kwargs_26866 = {}
    # Getting the type of 'is_f_file' (line 205)
    is_f_file_26864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 19), 'is_f_file', False)
    # Calling is_f_file(args, kwargs) (line 205)
    is_f_file_call_result_26867 = invoke(stypy.reporting.localization.Localization(__file__, 205, 19), is_f_file_26864, *[src_26865], **kwargs_26866)
    
    
    
    # Call to has_f90_header(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'src' (line 205)
    src_26869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 57), 'src', False)
    # Processing the call keyword arguments (line 205)
    kwargs_26870 = {}
    # Getting the type of 'has_f90_header' (line 205)
    has_f90_header_26868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 42), 'has_f90_header', False)
    # Calling has_f90_header(args, kwargs) (line 205)
    has_f90_header_call_result_26871 = invoke(stypy.reporting.localization.Localization(__file__, 205, 42), has_f90_header_26868, *[src_26869], **kwargs_26870)
    
    # Applying the 'not' unary operator (line 205)
    result_not__26872 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 38), 'not', has_f90_header_call_result_26871)
    
    # Applying the binary operator 'and' (line 205)
    result_and_keyword_26873 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 19), 'and', is_f_file_call_result_26867, result_not__26872)
    
    # Testing the type of an if condition (line 205)
    if_condition_26874 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 16), result_and_keyword_26873)
    # Assigning a type to the variable 'if_condition_26874' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 16), 'if_condition_26874', if_condition_26874)
    # SSA begins for if statement (line 205)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 206)
    # Processing the call arguments (line 206)
    
    # Obtaining an instance of the builtin type 'tuple' (line 206)
    tuple_26877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 206)
    # Adding element type (line 206)
    # Getting the type of 'obj' (line 206)
    obj_26878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 40), 'obj', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 40), tuple_26877, obj_26878)
    # Adding element type (line 206)
    
    # Obtaining an instance of the builtin type 'tuple' (line 206)
    tuple_26879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 206)
    # Adding element type (line 206)
    # Getting the type of 'src' (line 206)
    src_26880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 46), 'src', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 46), tuple_26879, src_26880)
    # Adding element type (line 206)
    # Getting the type of 'ext' (line 206)
    ext_26881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 51), 'ext', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 46), tuple_26879, ext_26881)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 40), tuple_26877, tuple_26879)
    
    # Processing the call keyword arguments (line 206)
    kwargs_26882 = {}
    # Getting the type of 'f77_objects' (line 206)
    f77_objects_26875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 20), 'f77_objects', False)
    # Obtaining the member 'append' of a type (line 206)
    append_26876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 20), f77_objects_26875, 'append')
    # Calling append(args, kwargs) (line 206)
    append_call_result_26883 = invoke(stypy.reporting.localization.Localization(__file__, 206, 20), append_26876, *[tuple_26877], **kwargs_26882)
    
    # SSA branch for the else part of an if statement (line 205)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 208)
    # Processing the call arguments (line 208)
    
    # Obtaining an instance of the builtin type 'tuple' (line 208)
    tuple_26886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 208)
    # Adding element type (line 208)
    # Getting the type of 'obj' (line 208)
    obj_26887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 42), 'obj', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 42), tuple_26886, obj_26887)
    # Adding element type (line 208)
    
    # Obtaining an instance of the builtin type 'tuple' (line 208)
    tuple_26888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 208)
    # Adding element type (line 208)
    # Getting the type of 'src' (line 208)
    src_26889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 48), 'src', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 48), tuple_26888, src_26889)
    # Adding element type (line 208)
    # Getting the type of 'ext' (line 208)
    ext_26890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 53), 'ext', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 48), tuple_26888, ext_26890)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 42), tuple_26886, tuple_26888)
    
    # Processing the call keyword arguments (line 208)
    kwargs_26891 = {}
    # Getting the type of 'other_objects' (line 208)
    other_objects_26884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 20), 'other_objects', False)
    # Obtaining the member 'append' of a type (line 208)
    append_26885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 20), other_objects_26884, 'append')
    # Calling append(args, kwargs) (line 208)
    append_call_result_26892 = invoke(stypy.reporting.localization.Localization(__file__, 208, 20), append_26885, *[tuple_26886], **kwargs_26891)
    
    # SSA join for if statement (line 205)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 200)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 211):
    
    # Assigning a Name to a Name (line 211):
    
    # Assigning a Name to a Name (line 211):
    # Getting the type of 'f77_objects' (line 211)
    f77_objects_26893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 22), 'f77_objects')
    # Assigning a type to the variable 'build_items' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'build_items', f77_objects_26893)
    
    # Getting the type of 'other_objects' (line 215)
    other_objects_26894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 17), 'other_objects')
    # Testing the type of a for loop iterable (line 215)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 215, 8), other_objects_26894)
    # Getting the type of the for loop variable (line 215)
    for_loop_var_26895 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 215, 8), other_objects_26894)
    # Assigning a type to the variable 'o' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'o', for_loop_var_26895)
    # SSA begins for a for statement (line 215)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to single_compile(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'o' (line 216)
    o_26897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 27), 'o', False)
    # Processing the call keyword arguments (line 216)
    kwargs_26898 = {}
    # Getting the type of 'single_compile' (line 216)
    single_compile_26896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'single_compile', False)
    # Calling single_compile(args, kwargs) (line 216)
    single_compile_call_result_26899 = invoke(stypy.reporting.localization.Localization(__file__, 216, 12), single_compile_26896, *[o_26897], **kwargs_26898)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 196)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 218):
    
    # Assigning a Call to a Name (line 218):
    
    # Assigning a Call to a Name (line 218):
    
    # Call to items(...): (line 218)
    # Processing the call keyword arguments (line 218)
    kwargs_26902 = {}
    # Getting the type of 'build' (line 218)
    build_26900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 22), 'build', False)
    # Obtaining the member 'items' of a type (line 218)
    items_26901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 22), build_26900, 'items')
    # Calling items(args, kwargs) (line 218)
    items_call_result_26903 = invoke(stypy.reporting.localization.Localization(__file__, 218, 22), items_26901, *[], **kwargs_26902)
    
    # Assigning a type to the variable 'build_items' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'build_items', items_call_result_26903)
    # SSA join for if statement (line 196)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 220):
    
    # Assigning a Call to a Name (line 220):
    
    # Assigning a Call to a Name (line 220):
    
    # Call to get_num_build_jobs(...): (line 220)
    # Processing the call keyword arguments (line 220)
    kwargs_26905 = {}
    # Getting the type of 'get_num_build_jobs' (line 220)
    get_num_build_jobs_26904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 11), 'get_num_build_jobs', False)
    # Calling get_num_build_jobs(args, kwargs) (line 220)
    get_num_build_jobs_call_result_26906 = invoke(stypy.reporting.localization.Localization(__file__, 220, 11), get_num_build_jobs_26904, *[], **kwargs_26905)
    
    # Assigning a type to the variable 'jobs' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'jobs', get_num_build_jobs_call_result_26906)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 221)
    # Processing the call arguments (line 221)
    # Getting the type of 'build' (line 221)
    build_26908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 11), 'build', False)
    # Processing the call keyword arguments (line 221)
    kwargs_26909 = {}
    # Getting the type of 'len' (line 221)
    len_26907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 7), 'len', False)
    # Calling len(args, kwargs) (line 221)
    len_call_result_26910 = invoke(stypy.reporting.localization.Localization(__file__, 221, 7), len_26907, *[build_26908], **kwargs_26909)
    
    int_26911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 20), 'int')
    # Applying the binary operator '>' (line 221)
    result_gt_26912 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 7), '>', len_call_result_26910, int_26911)
    
    
    # Getting the type of 'jobs' (line 221)
    jobs_26913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 26), 'jobs')
    int_26914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 33), 'int')
    # Applying the binary operator '>' (line 221)
    result_gt_26915 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 26), '>', jobs_26913, int_26914)
    
    # Applying the binary operator 'and' (line 221)
    result_and_keyword_26916 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 7), 'and', result_gt_26912, result_gt_26915)
    
    # Testing the type of an if condition (line 221)
    if_condition_26917 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 4), result_and_keyword_26916)
    # Assigning a type to the variable 'if_condition_26917' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'if_condition_26917', if_condition_26917)
    # SSA begins for if statement (line 221)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 223, 8))
    
    # 'import multiprocessing.pool' statement (line 223)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
    import_26918 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 223, 8), 'multiprocessing.pool')

    if (type(import_26918) is not StypyTypeError):

        if (import_26918 != 'pyd_module'):
            __import__(import_26918)
            sys_modules_26919 = sys.modules[import_26918]
            import_module(stypy.reporting.localization.Localization(__file__, 223, 8), 'multiprocessing.pool', sys_modules_26919.module_type_store, module_type_store)
        else:
            import multiprocessing.pool

            import_module(stypy.reporting.localization.Localization(__file__, 223, 8), 'multiprocessing.pool', multiprocessing.pool, module_type_store)

    else:
        # Assigning a type to the variable 'multiprocessing.pool' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'multiprocessing.pool', import_26918)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')
    
    
    # Assigning a Call to a Name (line 224):
    
    # Assigning a Call to a Name (line 224):
    
    # Assigning a Call to a Name (line 224):
    
    # Call to ThreadPool(...): (line 224)
    # Processing the call arguments (line 224)
    # Getting the type of 'jobs' (line 224)
    jobs_26923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 47), 'jobs', False)
    # Processing the call keyword arguments (line 224)
    kwargs_26924 = {}
    # Getting the type of 'multiprocessing' (line 224)
    multiprocessing_26920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 15), 'multiprocessing', False)
    # Obtaining the member 'pool' of a type (line 224)
    pool_26921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 15), multiprocessing_26920, 'pool')
    # Obtaining the member 'ThreadPool' of a type (line 224)
    ThreadPool_26922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 15), pool_26921, 'ThreadPool')
    # Calling ThreadPool(args, kwargs) (line 224)
    ThreadPool_call_result_26925 = invoke(stypy.reporting.localization.Localization(__file__, 224, 15), ThreadPool_26922, *[jobs_26923], **kwargs_26924)
    
    # Assigning a type to the variable 'pool' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'pool', ThreadPool_call_result_26925)
    
    # Call to map(...): (line 225)
    # Processing the call arguments (line 225)
    # Getting the type of 'single_compile' (line 225)
    single_compile_26928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 17), 'single_compile', False)
    # Getting the type of 'build_items' (line 225)
    build_items_26929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 33), 'build_items', False)
    # Processing the call keyword arguments (line 225)
    kwargs_26930 = {}
    # Getting the type of 'pool' (line 225)
    pool_26926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'pool', False)
    # Obtaining the member 'map' of a type (line 225)
    map_26927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 8), pool_26926, 'map')
    # Calling map(args, kwargs) (line 225)
    map_call_result_26931 = invoke(stypy.reporting.localization.Localization(__file__, 225, 8), map_26927, *[single_compile_26928, build_items_26929], **kwargs_26930)
    
    
    # Call to close(...): (line 226)
    # Processing the call keyword arguments (line 226)
    kwargs_26934 = {}
    # Getting the type of 'pool' (line 226)
    pool_26932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'pool', False)
    # Obtaining the member 'close' of a type (line 226)
    close_26933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), pool_26932, 'close')
    # Calling close(args, kwargs) (line 226)
    close_call_result_26935 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), close_26933, *[], **kwargs_26934)
    
    # SSA branch for the else part of an if statement (line 221)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'build_items' (line 229)
    build_items_26936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 17), 'build_items')
    # Testing the type of a for loop iterable (line 229)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 229, 8), build_items_26936)
    # Getting the type of the for loop variable (line 229)
    for_loop_var_26937 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 229, 8), build_items_26936)
    # Assigning a type to the variable 'o' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'o', for_loop_var_26937)
    # SSA begins for a for statement (line 229)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to single_compile(...): (line 230)
    # Processing the call arguments (line 230)
    # Getting the type of 'o' (line 230)
    o_26939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 27), 'o', False)
    # Processing the call keyword arguments (line 230)
    kwargs_26940 = {}
    # Getting the type of 'single_compile' (line 230)
    single_compile_26938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'single_compile', False)
    # Calling single_compile(args, kwargs) (line 230)
    single_compile_call_result_26941 = invoke(stypy.reporting.localization.Localization(__file__, 230, 12), single_compile_26938, *[o_26939], **kwargs_26940)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 221)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'objects' (line 233)
    objects_26942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 11), 'objects')
    # Assigning a type to the variable 'stypy_return_type' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'stypy_return_type', objects_26942)
    
    # ################# End of 'CCompiler_compile(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'CCompiler_compile' in the type store
    # Getting the type of 'stypy_return_type' (line 122)
    stypy_return_type_26943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_26943)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'CCompiler_compile'
    return stypy_return_type_26943

# Assigning a type to the variable 'CCompiler_compile' (line 122)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), 'CCompiler_compile', CCompiler_compile)

# Call to replace_method(...): (line 235)
# Processing the call arguments (line 235)
# Getting the type of 'CCompiler' (line 235)
CCompiler_26945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 15), 'CCompiler', False)
str_26946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 26), 'str', 'compile')
# Getting the type of 'CCompiler_compile' (line 235)
CCompiler_compile_26947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 37), 'CCompiler_compile', False)
# Processing the call keyword arguments (line 235)
kwargs_26948 = {}
# Getting the type of 'replace_method' (line 235)
replace_method_26944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 0), 'replace_method', False)
# Calling replace_method(args, kwargs) (line 235)
replace_method_call_result_26949 = invoke(stypy.reporting.localization.Localization(__file__, 235, 0), replace_method_26944, *[CCompiler_26945, str_26946, CCompiler_compile_26947], **kwargs_26948)


@norecursion
def CCompiler_customize_cmd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 237)
    tuple_26950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 237)
    
    defaults = [tuple_26950]
    # Create a new context for function 'CCompiler_customize_cmd'
    module_type_store = module_type_store.open_function_context('CCompiler_customize_cmd', 237, 0, False)
    
    # Passed parameters checking function
    CCompiler_customize_cmd.stypy_localization = localization
    CCompiler_customize_cmd.stypy_type_of_self = None
    CCompiler_customize_cmd.stypy_type_store = module_type_store
    CCompiler_customize_cmd.stypy_function_name = 'CCompiler_customize_cmd'
    CCompiler_customize_cmd.stypy_param_names_list = ['self', 'cmd', 'ignore']
    CCompiler_customize_cmd.stypy_varargs_param_name = None
    CCompiler_customize_cmd.stypy_kwargs_param_name = None
    CCompiler_customize_cmd.stypy_call_defaults = defaults
    CCompiler_customize_cmd.stypy_call_varargs = varargs
    CCompiler_customize_cmd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'CCompiler_customize_cmd', ['self', 'cmd', 'ignore'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'CCompiler_customize_cmd', localization, ['self', 'cmd', 'ignore'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'CCompiler_customize_cmd(...)' code ##################

    str_26951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, (-1)), 'str', "\n    Customize compiler using distutils command.\n\n    Parameters\n    ----------\n    cmd : class instance\n        An instance inheriting from `distutils.cmd.Command`.\n    ignore : sequence of str, optional\n        List of `CCompiler` commands (without ``'set_'``) that should not be\n        altered. Strings that are checked for are:\n        ``('include_dirs', 'define', 'undef', 'libraries', 'library_dirs',\n        'rpath', 'link_objects')``.\n\n    Returns\n    -------\n    None\n\n    ")
    
    # Call to info(...): (line 256)
    # Processing the call arguments (line 256)
    str_26954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 13), 'str', 'customize %s using %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 256)
    tuple_26955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 256)
    # Adding element type (line 256)
    # Getting the type of 'self' (line 256)
    self_26956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 40), 'self', False)
    # Obtaining the member '__class__' of a type (line 256)
    class___26957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 40), self_26956, '__class__')
    # Obtaining the member '__name__' of a type (line 256)
    name___26958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 40), class___26957, '__name__')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 40), tuple_26955, name___26958)
    # Adding element type (line 256)
    # Getting the type of 'cmd' (line 257)
    cmd_26959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 40), 'cmd', False)
    # Obtaining the member '__class__' of a type (line 257)
    class___26960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 40), cmd_26959, '__class__')
    # Obtaining the member '__name__' of a type (line 257)
    name___26961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 40), class___26960, '__name__')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 40), tuple_26955, name___26961)
    
    # Applying the binary operator '%' (line 256)
    result_mod_26962 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 13), '%', str_26954, tuple_26955)
    
    # Processing the call keyword arguments (line 256)
    kwargs_26963 = {}
    # Getting the type of 'log' (line 256)
    log_26952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'log', False)
    # Obtaining the member 'info' of a type (line 256)
    info_26953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 4), log_26952, 'info')
    # Calling info(args, kwargs) (line 256)
    info_call_result_26964 = invoke(stypy.reporting.localization.Localization(__file__, 256, 4), info_26953, *[result_mod_26962], **kwargs_26963)
    

    @norecursion
    def allow(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'allow'
        module_type_store = module_type_store.open_function_context('allow', 258, 4, False)
        
        # Passed parameters checking function
        allow.stypy_localization = localization
        allow.stypy_type_of_self = None
        allow.stypy_type_store = module_type_store
        allow.stypy_function_name = 'allow'
        allow.stypy_param_names_list = ['attr']
        allow.stypy_varargs_param_name = None
        allow.stypy_kwargs_param_name = None
        allow.stypy_call_defaults = defaults
        allow.stypy_call_varargs = varargs
        allow.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'allow', ['attr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'allow', localization, ['attr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'allow(...)' code ##################

        
        # Evaluating a boolean operation
        
        
        # Call to getattr(...): (line 259)
        # Processing the call arguments (line 259)
        # Getting the type of 'cmd' (line 259)
        cmd_26966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 23), 'cmd', False)
        # Getting the type of 'attr' (line 259)
        attr_26967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'attr', False)
        # Getting the type of 'None' (line 259)
        None_26968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 34), 'None', False)
        # Processing the call keyword arguments (line 259)
        kwargs_26969 = {}
        # Getting the type of 'getattr' (line 259)
        getattr_26965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 15), 'getattr', False)
        # Calling getattr(args, kwargs) (line 259)
        getattr_call_result_26970 = invoke(stypy.reporting.localization.Localization(__file__, 259, 15), getattr_26965, *[cmd_26966, attr_26967, None_26968], **kwargs_26969)
        
        # Getting the type of 'None' (line 259)
        None_26971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 47), 'None')
        # Applying the binary operator 'isnot' (line 259)
        result_is_not_26972 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 15), 'isnot', getattr_call_result_26970, None_26971)
        
        
        # Getting the type of 'attr' (line 259)
        attr_26973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 56), 'attr')
        # Getting the type of 'ignore' (line 259)
        ignore_26974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 68), 'ignore')
        # Applying the binary operator 'notin' (line 259)
        result_contains_26975 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 56), 'notin', attr_26973, ignore_26974)
        
        # Applying the binary operator 'and' (line 259)
        result_and_keyword_26976 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 15), 'and', result_is_not_26972, result_contains_26975)
        
        # Assigning a type to the variable 'stypy_return_type' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'stypy_return_type', result_and_keyword_26976)
        
        # ################# End of 'allow(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'allow' in the type store
        # Getting the type of 'stypy_return_type' (line 258)
        stypy_return_type_26977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26977)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'allow'
        return stypy_return_type_26977

    # Assigning a type to the variable 'allow' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'allow', allow)
    
    
    # Call to allow(...): (line 261)
    # Processing the call arguments (line 261)
    str_26979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 13), 'str', 'include_dirs')
    # Processing the call keyword arguments (line 261)
    kwargs_26980 = {}
    # Getting the type of 'allow' (line 261)
    allow_26978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 7), 'allow', False)
    # Calling allow(args, kwargs) (line 261)
    allow_call_result_26981 = invoke(stypy.reporting.localization.Localization(__file__, 261, 7), allow_26978, *[str_26979], **kwargs_26980)
    
    # Testing the type of an if condition (line 261)
    if_condition_26982 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 261, 4), allow_call_result_26981)
    # Assigning a type to the variable 'if_condition_26982' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'if_condition_26982', if_condition_26982)
    # SSA begins for if statement (line 261)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to set_include_dirs(...): (line 262)
    # Processing the call arguments (line 262)
    # Getting the type of 'cmd' (line 262)
    cmd_26985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 30), 'cmd', False)
    # Obtaining the member 'include_dirs' of a type (line 262)
    include_dirs_26986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 30), cmd_26985, 'include_dirs')
    # Processing the call keyword arguments (line 262)
    kwargs_26987 = {}
    # Getting the type of 'self' (line 262)
    self_26983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'self', False)
    # Obtaining the member 'set_include_dirs' of a type (line 262)
    set_include_dirs_26984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 8), self_26983, 'set_include_dirs')
    # Calling set_include_dirs(args, kwargs) (line 262)
    set_include_dirs_call_result_26988 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), set_include_dirs_26984, *[include_dirs_26986], **kwargs_26987)
    
    # SSA join for if statement (line 261)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to allow(...): (line 263)
    # Processing the call arguments (line 263)
    str_26990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 13), 'str', 'define')
    # Processing the call keyword arguments (line 263)
    kwargs_26991 = {}
    # Getting the type of 'allow' (line 263)
    allow_26989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 7), 'allow', False)
    # Calling allow(args, kwargs) (line 263)
    allow_call_result_26992 = invoke(stypy.reporting.localization.Localization(__file__, 263, 7), allow_26989, *[str_26990], **kwargs_26991)
    
    # Testing the type of an if condition (line 263)
    if_condition_26993 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 263, 4), allow_call_result_26992)
    # Assigning a type to the variable 'if_condition_26993' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'if_condition_26993', if_condition_26993)
    # SSA begins for if statement (line 263)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'cmd' (line 264)
    cmd_26994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 29), 'cmd')
    # Obtaining the member 'define' of a type (line 264)
    define_26995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 29), cmd_26994, 'define')
    # Testing the type of a for loop iterable (line 264)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 264, 8), define_26995)
    # Getting the type of the for loop variable (line 264)
    for_loop_var_26996 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 264, 8), define_26995)
    # Assigning a type to the variable 'name' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 8), for_loop_var_26996))
    # Assigning a type to the variable 'value' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 8), for_loop_var_26996))
    # SSA begins for a for statement (line 264)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to define_macro(...): (line 265)
    # Processing the call arguments (line 265)
    # Getting the type of 'name' (line 265)
    name_26999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 30), 'name', False)
    # Getting the type of 'value' (line 265)
    value_27000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 36), 'value', False)
    # Processing the call keyword arguments (line 265)
    kwargs_27001 = {}
    # Getting the type of 'self' (line 265)
    self_26997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'self', False)
    # Obtaining the member 'define_macro' of a type (line 265)
    define_macro_26998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 12), self_26997, 'define_macro')
    # Calling define_macro(args, kwargs) (line 265)
    define_macro_call_result_27002 = invoke(stypy.reporting.localization.Localization(__file__, 265, 12), define_macro_26998, *[name_26999, value_27000], **kwargs_27001)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 263)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to allow(...): (line 266)
    # Processing the call arguments (line 266)
    str_27004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 13), 'str', 'undef')
    # Processing the call keyword arguments (line 266)
    kwargs_27005 = {}
    # Getting the type of 'allow' (line 266)
    allow_27003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 7), 'allow', False)
    # Calling allow(args, kwargs) (line 266)
    allow_call_result_27006 = invoke(stypy.reporting.localization.Localization(__file__, 266, 7), allow_27003, *[str_27004], **kwargs_27005)
    
    # Testing the type of an if condition (line 266)
    if_condition_27007 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 266, 4), allow_call_result_27006)
    # Assigning a type to the variable 'if_condition_27007' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'if_condition_27007', if_condition_27007)
    # SSA begins for if statement (line 266)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'cmd' (line 267)
    cmd_27008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 21), 'cmd')
    # Obtaining the member 'undef' of a type (line 267)
    undef_27009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 21), cmd_27008, 'undef')
    # Testing the type of a for loop iterable (line 267)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 267, 8), undef_27009)
    # Getting the type of the for loop variable (line 267)
    for_loop_var_27010 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 267, 8), undef_27009)
    # Assigning a type to the variable 'macro' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'macro', for_loop_var_27010)
    # SSA begins for a for statement (line 267)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to undefine_macro(...): (line 268)
    # Processing the call arguments (line 268)
    # Getting the type of 'macro' (line 268)
    macro_27013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 32), 'macro', False)
    # Processing the call keyword arguments (line 268)
    kwargs_27014 = {}
    # Getting the type of 'self' (line 268)
    self_27011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'self', False)
    # Obtaining the member 'undefine_macro' of a type (line 268)
    undefine_macro_27012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 12), self_27011, 'undefine_macro')
    # Calling undefine_macro(args, kwargs) (line 268)
    undefine_macro_call_result_27015 = invoke(stypy.reporting.localization.Localization(__file__, 268, 12), undefine_macro_27012, *[macro_27013], **kwargs_27014)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 266)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to allow(...): (line 269)
    # Processing the call arguments (line 269)
    str_27017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 13), 'str', 'libraries')
    # Processing the call keyword arguments (line 269)
    kwargs_27018 = {}
    # Getting the type of 'allow' (line 269)
    allow_27016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 7), 'allow', False)
    # Calling allow(args, kwargs) (line 269)
    allow_call_result_27019 = invoke(stypy.reporting.localization.Localization(__file__, 269, 7), allow_27016, *[str_27017], **kwargs_27018)
    
    # Testing the type of an if condition (line 269)
    if_condition_27020 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 269, 4), allow_call_result_27019)
    # Assigning a type to the variable 'if_condition_27020' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'if_condition_27020', if_condition_27020)
    # SSA begins for if statement (line 269)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to set_libraries(...): (line 270)
    # Processing the call arguments (line 270)
    # Getting the type of 'self' (line 270)
    self_27023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 27), 'self', False)
    # Obtaining the member 'libraries' of a type (line 270)
    libraries_27024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 27), self_27023, 'libraries')
    # Getting the type of 'cmd' (line 270)
    cmd_27025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 44), 'cmd', False)
    # Obtaining the member 'libraries' of a type (line 270)
    libraries_27026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 44), cmd_27025, 'libraries')
    # Applying the binary operator '+' (line 270)
    result_add_27027 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 27), '+', libraries_27024, libraries_27026)
    
    # Processing the call keyword arguments (line 270)
    kwargs_27028 = {}
    # Getting the type of 'self' (line 270)
    self_27021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'self', False)
    # Obtaining the member 'set_libraries' of a type (line 270)
    set_libraries_27022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 8), self_27021, 'set_libraries')
    # Calling set_libraries(args, kwargs) (line 270)
    set_libraries_call_result_27029 = invoke(stypy.reporting.localization.Localization(__file__, 270, 8), set_libraries_27022, *[result_add_27027], **kwargs_27028)
    
    # SSA join for if statement (line 269)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to allow(...): (line 271)
    # Processing the call arguments (line 271)
    str_27031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 13), 'str', 'library_dirs')
    # Processing the call keyword arguments (line 271)
    kwargs_27032 = {}
    # Getting the type of 'allow' (line 271)
    allow_27030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 7), 'allow', False)
    # Calling allow(args, kwargs) (line 271)
    allow_call_result_27033 = invoke(stypy.reporting.localization.Localization(__file__, 271, 7), allow_27030, *[str_27031], **kwargs_27032)
    
    # Testing the type of an if condition (line 271)
    if_condition_27034 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 271, 4), allow_call_result_27033)
    # Assigning a type to the variable 'if_condition_27034' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'if_condition_27034', if_condition_27034)
    # SSA begins for if statement (line 271)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to set_library_dirs(...): (line 272)
    # Processing the call arguments (line 272)
    # Getting the type of 'self' (line 272)
    self_27037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 30), 'self', False)
    # Obtaining the member 'library_dirs' of a type (line 272)
    library_dirs_27038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 30), self_27037, 'library_dirs')
    # Getting the type of 'cmd' (line 272)
    cmd_27039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 50), 'cmd', False)
    # Obtaining the member 'library_dirs' of a type (line 272)
    library_dirs_27040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 50), cmd_27039, 'library_dirs')
    # Applying the binary operator '+' (line 272)
    result_add_27041 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 30), '+', library_dirs_27038, library_dirs_27040)
    
    # Processing the call keyword arguments (line 272)
    kwargs_27042 = {}
    # Getting the type of 'self' (line 272)
    self_27035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'self', False)
    # Obtaining the member 'set_library_dirs' of a type (line 272)
    set_library_dirs_27036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), self_27035, 'set_library_dirs')
    # Calling set_library_dirs(args, kwargs) (line 272)
    set_library_dirs_call_result_27043 = invoke(stypy.reporting.localization.Localization(__file__, 272, 8), set_library_dirs_27036, *[result_add_27041], **kwargs_27042)
    
    # SSA join for if statement (line 271)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to allow(...): (line 273)
    # Processing the call arguments (line 273)
    str_27045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 13), 'str', 'rpath')
    # Processing the call keyword arguments (line 273)
    kwargs_27046 = {}
    # Getting the type of 'allow' (line 273)
    allow_27044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 7), 'allow', False)
    # Calling allow(args, kwargs) (line 273)
    allow_call_result_27047 = invoke(stypy.reporting.localization.Localization(__file__, 273, 7), allow_27044, *[str_27045], **kwargs_27046)
    
    # Testing the type of an if condition (line 273)
    if_condition_27048 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 4), allow_call_result_27047)
    # Assigning a type to the variable 'if_condition_27048' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'if_condition_27048', if_condition_27048)
    # SSA begins for if statement (line 273)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to set_runtime_library_dirs(...): (line 274)
    # Processing the call arguments (line 274)
    # Getting the type of 'cmd' (line 274)
    cmd_27051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 38), 'cmd', False)
    # Obtaining the member 'rpath' of a type (line 274)
    rpath_27052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 38), cmd_27051, 'rpath')
    # Processing the call keyword arguments (line 274)
    kwargs_27053 = {}
    # Getting the type of 'self' (line 274)
    self_27049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'self', False)
    # Obtaining the member 'set_runtime_library_dirs' of a type (line 274)
    set_runtime_library_dirs_27050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), self_27049, 'set_runtime_library_dirs')
    # Calling set_runtime_library_dirs(args, kwargs) (line 274)
    set_runtime_library_dirs_call_result_27054 = invoke(stypy.reporting.localization.Localization(__file__, 274, 8), set_runtime_library_dirs_27050, *[rpath_27052], **kwargs_27053)
    
    # SSA join for if statement (line 273)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to allow(...): (line 275)
    # Processing the call arguments (line 275)
    str_27056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 13), 'str', 'link_objects')
    # Processing the call keyword arguments (line 275)
    kwargs_27057 = {}
    # Getting the type of 'allow' (line 275)
    allow_27055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 7), 'allow', False)
    # Calling allow(args, kwargs) (line 275)
    allow_call_result_27058 = invoke(stypy.reporting.localization.Localization(__file__, 275, 7), allow_27055, *[str_27056], **kwargs_27057)
    
    # Testing the type of an if condition (line 275)
    if_condition_27059 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 275, 4), allow_call_result_27058)
    # Assigning a type to the variable 'if_condition_27059' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'if_condition_27059', if_condition_27059)
    # SSA begins for if statement (line 275)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to set_link_objects(...): (line 276)
    # Processing the call arguments (line 276)
    # Getting the type of 'cmd' (line 276)
    cmd_27062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 30), 'cmd', False)
    # Obtaining the member 'link_objects' of a type (line 276)
    link_objects_27063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 30), cmd_27062, 'link_objects')
    # Processing the call keyword arguments (line 276)
    kwargs_27064 = {}
    # Getting the type of 'self' (line 276)
    self_27060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'self', False)
    # Obtaining the member 'set_link_objects' of a type (line 276)
    set_link_objects_27061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 8), self_27060, 'set_link_objects')
    # Calling set_link_objects(args, kwargs) (line 276)
    set_link_objects_call_result_27065 = invoke(stypy.reporting.localization.Localization(__file__, 276, 8), set_link_objects_27061, *[link_objects_27063], **kwargs_27064)
    
    # SSA join for if statement (line 275)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'CCompiler_customize_cmd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'CCompiler_customize_cmd' in the type store
    # Getting the type of 'stypy_return_type' (line 237)
    stypy_return_type_27066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_27066)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'CCompiler_customize_cmd'
    return stypy_return_type_27066

# Assigning a type to the variable 'CCompiler_customize_cmd' (line 237)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 0), 'CCompiler_customize_cmd', CCompiler_customize_cmd)

# Call to replace_method(...): (line 278)
# Processing the call arguments (line 278)
# Getting the type of 'CCompiler' (line 278)
CCompiler_27068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 15), 'CCompiler', False)
str_27069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 26), 'str', 'customize_cmd')
# Getting the type of 'CCompiler_customize_cmd' (line 278)
CCompiler_customize_cmd_27070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 43), 'CCompiler_customize_cmd', False)
# Processing the call keyword arguments (line 278)
kwargs_27071 = {}
# Getting the type of 'replace_method' (line 278)
replace_method_27067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 0), 'replace_method', False)
# Calling replace_method(args, kwargs) (line 278)
replace_method_call_result_27072 = invoke(stypy.reporting.localization.Localization(__file__, 278, 0), replace_method_27067, *[CCompiler_27068, str_27069, CCompiler_customize_cmd_27070], **kwargs_27071)


@norecursion
def _compiler_to_string(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_compiler_to_string'
    module_type_store = module_type_store.open_function_context('_compiler_to_string', 280, 0, False)
    
    # Passed parameters checking function
    _compiler_to_string.stypy_localization = localization
    _compiler_to_string.stypy_type_of_self = None
    _compiler_to_string.stypy_type_store = module_type_store
    _compiler_to_string.stypy_function_name = '_compiler_to_string'
    _compiler_to_string.stypy_param_names_list = ['compiler']
    _compiler_to_string.stypy_varargs_param_name = None
    _compiler_to_string.stypy_kwargs_param_name = None
    _compiler_to_string.stypy_call_defaults = defaults
    _compiler_to_string.stypy_call_varargs = varargs
    _compiler_to_string.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_compiler_to_string', ['compiler'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_compiler_to_string', localization, ['compiler'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_compiler_to_string(...)' code ##################

    
    # Assigning a List to a Name (line 281):
    
    # Assigning a List to a Name (line 281):
    
    # Assigning a List to a Name (line 281):
    
    # Obtaining an instance of the builtin type 'list' (line 281)
    list_27073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 281)
    
    # Assigning a type to the variable 'props' (line 281)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'props', list_27073)
    
    # Assigning a Num to a Name (line 282):
    
    # Assigning a Num to a Name (line 282):
    
    # Assigning a Num to a Name (line 282):
    int_27074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 9), 'int')
    # Assigning a type to the variable 'mx' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'mx', int_27074)
    
    # Assigning a Call to a Name (line 283):
    
    # Assigning a Call to a Name (line 283):
    
    # Assigning a Call to a Name (line 283):
    
    # Call to list(...): (line 283)
    # Processing the call arguments (line 283)
    
    # Call to keys(...): (line 283)
    # Processing the call keyword arguments (line 283)
    kwargs_27079 = {}
    # Getting the type of 'compiler' (line 283)
    compiler_27076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'compiler', False)
    # Obtaining the member 'executables' of a type (line 283)
    executables_27077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 16), compiler_27076, 'executables')
    # Obtaining the member 'keys' of a type (line 283)
    keys_27078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 16), executables_27077, 'keys')
    # Calling keys(args, kwargs) (line 283)
    keys_call_result_27080 = invoke(stypy.reporting.localization.Localization(__file__, 283, 16), keys_27078, *[], **kwargs_27079)
    
    # Processing the call keyword arguments (line 283)
    kwargs_27081 = {}
    # Getting the type of 'list' (line 283)
    list_27075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 11), 'list', False)
    # Calling list(args, kwargs) (line 283)
    list_call_result_27082 = invoke(stypy.reporting.localization.Localization(__file__, 283, 11), list_27075, *[keys_call_result_27080], **kwargs_27081)
    
    # Assigning a type to the variable 'keys' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'keys', list_call_result_27082)
    
    
    # Obtaining an instance of the builtin type 'list' (line 284)
    list_27083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 284)
    # Adding element type (line 284)
    str_27084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 16), 'str', 'version')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 15), list_27083, str_27084)
    # Adding element type (line 284)
    str_27085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 27), 'str', 'libraries')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 15), list_27083, str_27085)
    # Adding element type (line 284)
    str_27086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 40), 'str', 'library_dirs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 15), list_27083, str_27086)
    # Adding element type (line 284)
    str_27087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 16), 'str', 'object_switch')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 15), list_27083, str_27087)
    # Adding element type (line 284)
    str_27088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 33), 'str', 'compile_switch')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 15), list_27083, str_27088)
    # Adding element type (line 284)
    str_27089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 16), 'str', 'include_dirs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 15), list_27083, str_27089)
    # Adding element type (line 284)
    str_27090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 32), 'str', 'define')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 15), list_27083, str_27090)
    # Adding element type (line 284)
    str_27091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 42), 'str', 'undef')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 15), list_27083, str_27091)
    # Adding element type (line 284)
    str_27092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 51), 'str', 'rpath')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 15), list_27083, str_27092)
    # Adding element type (line 284)
    str_27093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 60), 'str', 'link_objects')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 15), list_27083, str_27093)
    
    # Testing the type of a for loop iterable (line 284)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 284, 4), list_27083)
    # Getting the type of the for loop variable (line 284)
    for_loop_var_27094 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 284, 4), list_27083)
    # Assigning a type to the variable 'key' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'key', for_loop_var_27094)
    # SSA begins for a for statement (line 284)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'key' (line 287)
    key_27095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 11), 'key')
    # Getting the type of 'keys' (line 287)
    keys_27096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 22), 'keys')
    # Applying the binary operator 'notin' (line 287)
    result_contains_27097 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 11), 'notin', key_27095, keys_27096)
    
    # Testing the type of an if condition (line 287)
    if_condition_27098 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 287, 8), result_contains_27097)
    # Assigning a type to the variable 'if_condition_27098' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'if_condition_27098', if_condition_27098)
    # SSA begins for if statement (line 287)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 288)
    # Processing the call arguments (line 288)
    # Getting the type of 'key' (line 288)
    key_27101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 24), 'key', False)
    # Processing the call keyword arguments (line 288)
    kwargs_27102 = {}
    # Getting the type of 'keys' (line 288)
    keys_27099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'keys', False)
    # Obtaining the member 'append' of a type (line 288)
    append_27100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 12), keys_27099, 'append')
    # Calling append(args, kwargs) (line 288)
    append_call_result_27103 = invoke(stypy.reporting.localization.Localization(__file__, 288, 12), append_27100, *[key_27101], **kwargs_27102)
    
    # SSA join for if statement (line 287)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'keys' (line 289)
    keys_27104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 15), 'keys')
    # Testing the type of a for loop iterable (line 289)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 289, 4), keys_27104)
    # Getting the type of the for loop variable (line 289)
    for_loop_var_27105 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 289, 4), keys_27104)
    # Assigning a type to the variable 'key' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'key', for_loop_var_27105)
    # SSA begins for a for statement (line 289)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to hasattr(...): (line 290)
    # Processing the call arguments (line 290)
    # Getting the type of 'compiler' (line 290)
    compiler_27107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 19), 'compiler', False)
    # Getting the type of 'key' (line 290)
    key_27108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 29), 'key', False)
    # Processing the call keyword arguments (line 290)
    kwargs_27109 = {}
    # Getting the type of 'hasattr' (line 290)
    hasattr_27106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 11), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 290)
    hasattr_call_result_27110 = invoke(stypy.reporting.localization.Localization(__file__, 290, 11), hasattr_27106, *[compiler_27107, key_27108], **kwargs_27109)
    
    # Testing the type of an if condition (line 290)
    if_condition_27111 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 290, 8), hasattr_call_result_27110)
    # Assigning a type to the variable 'if_condition_27111' (line 290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'if_condition_27111', if_condition_27111)
    # SSA begins for if statement (line 290)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 291):
    
    # Assigning a Call to a Name (line 291):
    
    # Assigning a Call to a Name (line 291):
    
    # Call to getattr(...): (line 291)
    # Processing the call arguments (line 291)
    # Getting the type of 'compiler' (line 291)
    compiler_27113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 24), 'compiler', False)
    # Getting the type of 'key' (line 291)
    key_27114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 34), 'key', False)
    # Processing the call keyword arguments (line 291)
    kwargs_27115 = {}
    # Getting the type of 'getattr' (line 291)
    getattr_27112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 16), 'getattr', False)
    # Calling getattr(args, kwargs) (line 291)
    getattr_call_result_27116 = invoke(stypy.reporting.localization.Localization(__file__, 291, 16), getattr_27112, *[compiler_27113, key_27114], **kwargs_27115)
    
    # Assigning a type to the variable 'v' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'v', getattr_call_result_27116)
    
    # Assigning a Call to a Name (line 292):
    
    # Assigning a Call to a Name (line 292):
    
    # Assigning a Call to a Name (line 292):
    
    # Call to max(...): (line 292)
    # Processing the call arguments (line 292)
    # Getting the type of 'mx' (line 292)
    mx_27118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 21), 'mx', False)
    
    # Call to len(...): (line 292)
    # Processing the call arguments (line 292)
    # Getting the type of 'key' (line 292)
    key_27120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 29), 'key', False)
    # Processing the call keyword arguments (line 292)
    kwargs_27121 = {}
    # Getting the type of 'len' (line 292)
    len_27119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 25), 'len', False)
    # Calling len(args, kwargs) (line 292)
    len_call_result_27122 = invoke(stypy.reporting.localization.Localization(__file__, 292, 25), len_27119, *[key_27120], **kwargs_27121)
    
    # Processing the call keyword arguments (line 292)
    kwargs_27123 = {}
    # Getting the type of 'max' (line 292)
    max_27117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 17), 'max', False)
    # Calling max(args, kwargs) (line 292)
    max_call_result_27124 = invoke(stypy.reporting.localization.Localization(__file__, 292, 17), max_27117, *[mx_27118, len_call_result_27122], **kwargs_27123)
    
    # Assigning a type to the variable 'mx' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'mx', max_call_result_27124)
    
    # Call to append(...): (line 293)
    # Processing the call arguments (line 293)
    
    # Obtaining an instance of the builtin type 'tuple' (line 293)
    tuple_27127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 293)
    # Adding element type (line 293)
    # Getting the type of 'key' (line 293)
    key_27128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 26), 'key', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 26), tuple_27127, key_27128)
    # Adding element type (line 293)
    
    # Call to repr(...): (line 293)
    # Processing the call arguments (line 293)
    # Getting the type of 'v' (line 293)
    v_27130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 36), 'v', False)
    # Processing the call keyword arguments (line 293)
    kwargs_27131 = {}
    # Getting the type of 'repr' (line 293)
    repr_27129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 31), 'repr', False)
    # Calling repr(args, kwargs) (line 293)
    repr_call_result_27132 = invoke(stypy.reporting.localization.Localization(__file__, 293, 31), repr_27129, *[v_27130], **kwargs_27131)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 26), tuple_27127, repr_call_result_27132)
    
    # Processing the call keyword arguments (line 293)
    kwargs_27133 = {}
    # Getting the type of 'props' (line 293)
    props_27125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'props', False)
    # Obtaining the member 'append' of a type (line 293)
    append_27126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 12), props_27125, 'append')
    # Calling append(args, kwargs) (line 293)
    append_call_result_27134 = invoke(stypy.reporting.localization.Localization(__file__, 293, 12), append_27126, *[tuple_27127], **kwargs_27133)
    
    # SSA join for if statement (line 290)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 294):
    
    # Assigning a List to a Name (line 294):
    
    # Assigning a List to a Name (line 294):
    
    # Obtaining an instance of the builtin type 'list' (line 294)
    list_27135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 294)
    
    # Assigning a type to the variable 'lines' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'lines', list_27135)
    
    # Assigning a BinOp to a Name (line 295):
    
    # Assigning a BinOp to a Name (line 295):
    
    # Assigning a BinOp to a Name (line 295):
    str_27136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 13), 'str', '%-')
    
    # Call to repr(...): (line 295)
    # Processing the call arguments (line 295)
    # Getting the type of 'mx' (line 295)
    mx_27138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 25), 'mx', False)
    int_27139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 28), 'int')
    # Applying the binary operator '+' (line 295)
    result_add_27140 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 25), '+', mx_27138, int_27139)
    
    # Processing the call keyword arguments (line 295)
    kwargs_27141 = {}
    # Getting the type of 'repr' (line 295)
    repr_27137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 20), 'repr', False)
    # Calling repr(args, kwargs) (line 295)
    repr_call_result_27142 = invoke(stypy.reporting.localization.Localization(__file__, 295, 20), repr_27137, *[result_add_27140], **kwargs_27141)
    
    # Applying the binary operator '+' (line 295)
    result_add_27143 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 13), '+', str_27136, repr_call_result_27142)
    
    str_27144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 33), 'str', 's = %s')
    # Applying the binary operator '+' (line 295)
    result_add_27145 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 31), '+', result_add_27143, str_27144)
    
    # Assigning a type to the variable 'format' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'format', result_add_27145)
    
    # Getting the type of 'props' (line 296)
    props_27146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 16), 'props')
    # Testing the type of a for loop iterable (line 296)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 296, 4), props_27146)
    # Getting the type of the for loop variable (line 296)
    for_loop_var_27147 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 296, 4), props_27146)
    # Assigning a type to the variable 'prop' (line 296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 'prop', for_loop_var_27147)
    # SSA begins for a for statement (line 296)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 297)
    # Processing the call arguments (line 297)
    # Getting the type of 'format' (line 297)
    format_27150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 21), 'format', False)
    # Getting the type of 'prop' (line 297)
    prop_27151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 30), 'prop', False)
    # Applying the binary operator '%' (line 297)
    result_mod_27152 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 21), '%', format_27150, prop_27151)
    
    # Processing the call keyword arguments (line 297)
    kwargs_27153 = {}
    # Getting the type of 'lines' (line 297)
    lines_27148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'lines', False)
    # Obtaining the member 'append' of a type (line 297)
    append_27149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 8), lines_27148, 'append')
    # Calling append(args, kwargs) (line 297)
    append_call_result_27154 = invoke(stypy.reporting.localization.Localization(__file__, 297, 8), append_27149, *[result_mod_27152], **kwargs_27153)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to join(...): (line 298)
    # Processing the call arguments (line 298)
    # Getting the type of 'lines' (line 298)
    lines_27157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 21), 'lines', False)
    # Processing the call keyword arguments (line 298)
    kwargs_27158 = {}
    str_27155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 11), 'str', '\n')
    # Obtaining the member 'join' of a type (line 298)
    join_27156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 11), str_27155, 'join')
    # Calling join(args, kwargs) (line 298)
    join_call_result_27159 = invoke(stypy.reporting.localization.Localization(__file__, 298, 11), join_27156, *[lines_27157], **kwargs_27158)
    
    # Assigning a type to the variable 'stypy_return_type' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'stypy_return_type', join_call_result_27159)
    
    # ################# End of '_compiler_to_string(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_compiler_to_string' in the type store
    # Getting the type of 'stypy_return_type' (line 280)
    stypy_return_type_27160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_27160)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_compiler_to_string'
    return stypy_return_type_27160

# Assigning a type to the variable '_compiler_to_string' (line 280)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 0), '_compiler_to_string', _compiler_to_string)

@norecursion
def CCompiler_show_customization(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'CCompiler_show_customization'
    module_type_store = module_type_store.open_function_context('CCompiler_show_customization', 300, 0, False)
    
    # Passed parameters checking function
    CCompiler_show_customization.stypy_localization = localization
    CCompiler_show_customization.stypy_type_of_self = None
    CCompiler_show_customization.stypy_type_store = module_type_store
    CCompiler_show_customization.stypy_function_name = 'CCompiler_show_customization'
    CCompiler_show_customization.stypy_param_names_list = ['self']
    CCompiler_show_customization.stypy_varargs_param_name = None
    CCompiler_show_customization.stypy_kwargs_param_name = None
    CCompiler_show_customization.stypy_call_defaults = defaults
    CCompiler_show_customization.stypy_call_varargs = varargs
    CCompiler_show_customization.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'CCompiler_show_customization', ['self'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'CCompiler_show_customization', localization, ['self'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'CCompiler_show_customization(...)' code ##################

    str_27161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, (-1)), 'str', '\n    Print the compiler customizations to stdout.\n\n    Parameters\n    ----------\n    None\n\n    Returns\n    -------\n    None\n\n    Notes\n    -----\n    Printing is only done if the distutils log threshold is < 2.\n\n    ')
    
    int_27162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 7), 'int')
    # Testing the type of an if condition (line 317)
    if_condition_27163 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 317, 4), int_27162)
    # Assigning a type to the variable 'if_condition_27163' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'if_condition_27163', if_condition_27163)
    # SSA begins for if statement (line 317)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Obtaining an instance of the builtin type 'list' (line 318)
    list_27164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 318)
    # Adding element type (line 318)
    str_27165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 25), 'str', 'include_dirs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 24), list_27164, str_27165)
    # Adding element type (line 318)
    str_27166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 41), 'str', 'define')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 24), list_27164, str_27166)
    # Adding element type (line 318)
    str_27167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 51), 'str', 'undef')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 24), list_27164, str_27167)
    # Adding element type (line 318)
    str_27168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 25), 'str', 'libraries')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 24), list_27164, str_27168)
    # Adding element type (line 318)
    str_27169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 38), 'str', 'library_dirs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 24), list_27164, str_27169)
    # Adding element type (line 318)
    str_27170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 25), 'str', 'rpath')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 24), list_27164, str_27170)
    # Adding element type (line 318)
    str_27171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 34), 'str', 'link_objects')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 24), list_27164, str_27171)
    
    # Testing the type of a for loop iterable (line 318)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 318, 8), list_27164)
    # Getting the type of the for loop variable (line 318)
    for_loop_var_27172 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 318, 8), list_27164)
    # Assigning a type to the variable 'attrname' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'attrname', for_loop_var_27172)
    # SSA begins for a for statement (line 318)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 321):
    
    # Assigning a Call to a Name (line 321):
    
    # Assigning a Call to a Name (line 321):
    
    # Call to getattr(...): (line 321)
    # Processing the call arguments (line 321)
    # Getting the type of 'self' (line 321)
    self_27174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 27), 'self', False)
    # Getting the type of 'attrname' (line 321)
    attrname_27175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 33), 'attrname', False)
    # Getting the type of 'None' (line 321)
    None_27176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 43), 'None', False)
    # Processing the call keyword arguments (line 321)
    kwargs_27177 = {}
    # Getting the type of 'getattr' (line 321)
    getattr_27173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 19), 'getattr', False)
    # Calling getattr(args, kwargs) (line 321)
    getattr_call_result_27178 = invoke(stypy.reporting.localization.Localization(__file__, 321, 19), getattr_27173, *[self_27174, attrname_27175, None_27176], **kwargs_27177)
    
    # Assigning a type to the variable 'attr' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'attr', getattr_call_result_27178)
    
    
    # Getting the type of 'attr' (line 322)
    attr_27179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 19), 'attr')
    # Applying the 'not' unary operator (line 322)
    result_not__27180 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 15), 'not', attr_27179)
    
    # Testing the type of an if condition (line 322)
    if_condition_27181 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 322, 12), result_not__27180)
    # Assigning a type to the variable 'if_condition_27181' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'if_condition_27181', if_condition_27181)
    # SSA begins for if statement (line 322)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 322)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to info(...): (line 324)
    # Processing the call arguments (line 324)
    str_27184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 21), 'str', "compiler '%s' is set to %s")
    
    # Obtaining an instance of the builtin type 'tuple' (line 324)
    tuple_27185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 53), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 324)
    # Adding element type (line 324)
    # Getting the type of 'attrname' (line 324)
    attrname_27186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 53), 'attrname', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 53), tuple_27185, attrname_27186)
    # Adding element type (line 324)
    # Getting the type of 'attr' (line 324)
    attr_27187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 63), 'attr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 53), tuple_27185, attr_27187)
    
    # Applying the binary operator '%' (line 324)
    result_mod_27188 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 21), '%', str_27184, tuple_27185)
    
    # Processing the call keyword arguments (line 324)
    kwargs_27189 = {}
    # Getting the type of 'log' (line 324)
    log_27182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'log', False)
    # Obtaining the member 'info' of a type (line 324)
    info_27183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 12), log_27182, 'info')
    # Calling info(args, kwargs) (line 324)
    info_call_result_27190 = invoke(stypy.reporting.localization.Localization(__file__, 324, 12), info_27183, *[result_mod_27188], **kwargs_27189)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 317)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 325)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to get_version(...): (line 326)
    # Processing the call keyword arguments (line 326)
    kwargs_27193 = {}
    # Getting the type of 'self' (line 326)
    self_27191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'self', False)
    # Obtaining the member 'get_version' of a type (line 326)
    get_version_27192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 8), self_27191, 'get_version')
    # Calling get_version(args, kwargs) (line 326)
    get_version_call_result_27194 = invoke(stypy.reporting.localization.Localization(__file__, 326, 8), get_version_27192, *[], **kwargs_27193)
    
    # SSA branch for the except part of a try statement (line 325)
    # SSA branch for the except '<any exception>' branch of a try statement (line 325)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 325)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'log' (line 329)
    log_27195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 7), 'log')
    # Obtaining the member '_global_log' of a type (line 329)
    _global_log_27196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 7), log_27195, '_global_log')
    # Obtaining the member 'threshold' of a type (line 329)
    threshold_27197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 7), _global_log_27196, 'threshold')
    int_27198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 33), 'int')
    # Applying the binary operator '<' (line 329)
    result_lt_27199 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 7), '<', threshold_27197, int_27198)
    
    # Testing the type of an if condition (line 329)
    if_condition_27200 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 329, 4), result_lt_27199)
    # Assigning a type to the variable 'if_condition_27200' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'if_condition_27200', if_condition_27200)
    # SSA begins for if statement (line 329)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 330)
    # Processing the call arguments (line 330)
    str_27202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 14), 'str', '*')
    int_27203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 18), 'int')
    # Applying the binary operator '*' (line 330)
    result_mul_27204 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 14), '*', str_27202, int_27203)
    
    # Processing the call keyword arguments (line 330)
    kwargs_27205 = {}
    # Getting the type of 'print' (line 330)
    print_27201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'print', False)
    # Calling print(args, kwargs) (line 330)
    print_call_result_27206 = invoke(stypy.reporting.localization.Localization(__file__, 330, 8), print_27201, *[result_mul_27204], **kwargs_27205)
    
    
    # Call to print(...): (line 331)
    # Processing the call arguments (line 331)
    # Getting the type of 'self' (line 331)
    self_27208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 14), 'self', False)
    # Obtaining the member '__class__' of a type (line 331)
    class___27209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 14), self_27208, '__class__')
    # Processing the call keyword arguments (line 331)
    kwargs_27210 = {}
    # Getting the type of 'print' (line 331)
    print_27207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'print', False)
    # Calling print(args, kwargs) (line 331)
    print_call_result_27211 = invoke(stypy.reporting.localization.Localization(__file__, 331, 8), print_27207, *[class___27209], **kwargs_27210)
    
    
    # Call to print(...): (line 332)
    # Processing the call arguments (line 332)
    
    # Call to _compiler_to_string(...): (line 332)
    # Processing the call arguments (line 332)
    # Getting the type of 'self' (line 332)
    self_27214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 34), 'self', False)
    # Processing the call keyword arguments (line 332)
    kwargs_27215 = {}
    # Getting the type of '_compiler_to_string' (line 332)
    _compiler_to_string_27213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 14), '_compiler_to_string', False)
    # Calling _compiler_to_string(args, kwargs) (line 332)
    _compiler_to_string_call_result_27216 = invoke(stypy.reporting.localization.Localization(__file__, 332, 14), _compiler_to_string_27213, *[self_27214], **kwargs_27215)
    
    # Processing the call keyword arguments (line 332)
    kwargs_27217 = {}
    # Getting the type of 'print' (line 332)
    print_27212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'print', False)
    # Calling print(args, kwargs) (line 332)
    print_call_result_27218 = invoke(stypy.reporting.localization.Localization(__file__, 332, 8), print_27212, *[_compiler_to_string_call_result_27216], **kwargs_27217)
    
    
    # Call to print(...): (line 333)
    # Processing the call arguments (line 333)
    str_27220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 14), 'str', '*')
    int_27221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 18), 'int')
    # Applying the binary operator '*' (line 333)
    result_mul_27222 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 14), '*', str_27220, int_27221)
    
    # Processing the call keyword arguments (line 333)
    kwargs_27223 = {}
    # Getting the type of 'print' (line 333)
    print_27219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'print', False)
    # Calling print(args, kwargs) (line 333)
    print_call_result_27224 = invoke(stypy.reporting.localization.Localization(__file__, 333, 8), print_27219, *[result_mul_27222], **kwargs_27223)
    
    # SSA join for if statement (line 329)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'CCompiler_show_customization(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'CCompiler_show_customization' in the type store
    # Getting the type of 'stypy_return_type' (line 300)
    stypy_return_type_27225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_27225)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'CCompiler_show_customization'
    return stypy_return_type_27225

# Assigning a type to the variable 'CCompiler_show_customization' (line 300)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 0), 'CCompiler_show_customization', CCompiler_show_customization)

# Call to replace_method(...): (line 335)
# Processing the call arguments (line 335)
# Getting the type of 'CCompiler' (line 335)
CCompiler_27227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 15), 'CCompiler', False)
str_27228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 26), 'str', 'show_customization')
# Getting the type of 'CCompiler_show_customization' (line 335)
CCompiler_show_customization_27229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 48), 'CCompiler_show_customization', False)
# Processing the call keyword arguments (line 335)
kwargs_27230 = {}
# Getting the type of 'replace_method' (line 335)
replace_method_27226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 0), 'replace_method', False)
# Calling replace_method(args, kwargs) (line 335)
replace_method_call_result_27231 = invoke(stypy.reporting.localization.Localization(__file__, 335, 0), replace_method_27226, *[CCompiler_27227, str_27228, CCompiler_show_customization_27229], **kwargs_27230)


@norecursion
def CCompiler_customize(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_27232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 45), 'int')
    defaults = [int_27232]
    # Create a new context for function 'CCompiler_customize'
    module_type_store = module_type_store.open_function_context('CCompiler_customize', 337, 0, False)
    
    # Passed parameters checking function
    CCompiler_customize.stypy_localization = localization
    CCompiler_customize.stypy_type_of_self = None
    CCompiler_customize.stypy_type_store = module_type_store
    CCompiler_customize.stypy_function_name = 'CCompiler_customize'
    CCompiler_customize.stypy_param_names_list = ['self', 'dist', 'need_cxx']
    CCompiler_customize.stypy_varargs_param_name = None
    CCompiler_customize.stypy_kwargs_param_name = None
    CCompiler_customize.stypy_call_defaults = defaults
    CCompiler_customize.stypy_call_varargs = varargs
    CCompiler_customize.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'CCompiler_customize', ['self', 'dist', 'need_cxx'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'CCompiler_customize', localization, ['self', 'dist', 'need_cxx'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'CCompiler_customize(...)' code ##################

    str_27233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, (-1)), 'str', '\n    Do any platform-specific customization of a compiler instance.\n\n    This method calls `distutils.sysconfig.customize_compiler` for\n    platform-specific customization, as well as optionally remove a flag\n    to suppress spurious warnings in case C++ code is being compiled.\n\n    Parameters\n    ----------\n    dist : object\n        This parameter is not used for anything.\n    need_cxx : bool, optional\n        Whether or not C++ has to be compiled. If so (True), the\n        ``"-Wstrict-prototypes"`` option is removed to prevent spurious\n        warnings. Default is False.\n\n    Returns\n    -------\n    None\n\n    Notes\n    -----\n    All the default options used by distutils can be extracted with::\n\n      from distutils import sysconfig\n      sysconfig.get_config_vars(\'CC\', \'CXX\', \'OPT\', \'BASECFLAGS\',\n                                \'CCSHARED\', \'LDSHARED\', \'SO\')\n\n    ')
    
    # Call to info(...): (line 368)
    # Processing the call arguments (line 368)
    str_27236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 13), 'str', 'customize %s')
    # Getting the type of 'self' (line 368)
    self_27237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 31), 'self', False)
    # Obtaining the member '__class__' of a type (line 368)
    class___27238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 31), self_27237, '__class__')
    # Obtaining the member '__name__' of a type (line 368)
    name___27239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 31), class___27238, '__name__')
    # Applying the binary operator '%' (line 368)
    result_mod_27240 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 13), '%', str_27236, name___27239)
    
    # Processing the call keyword arguments (line 368)
    kwargs_27241 = {}
    # Getting the type of 'log' (line 368)
    log_27234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'log', False)
    # Obtaining the member 'info' of a type (line 368)
    info_27235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 4), log_27234, 'info')
    # Calling info(args, kwargs) (line 368)
    info_call_result_27242 = invoke(stypy.reporting.localization.Localization(__file__, 368, 4), info_27235, *[result_mod_27240], **kwargs_27241)
    
    
    # Call to customize_compiler(...): (line 369)
    # Processing the call arguments (line 369)
    # Getting the type of 'self' (line 369)
    self_27244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 23), 'self', False)
    # Processing the call keyword arguments (line 369)
    kwargs_27245 = {}
    # Getting the type of 'customize_compiler' (line 369)
    customize_compiler_27243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'customize_compiler', False)
    # Calling customize_compiler(args, kwargs) (line 369)
    customize_compiler_call_result_27246 = invoke(stypy.reporting.localization.Localization(__file__, 369, 4), customize_compiler_27243, *[self_27244], **kwargs_27245)
    
    
    # Getting the type of 'need_cxx' (line 370)
    need_cxx_27247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 7), 'need_cxx')
    # Testing the type of an if condition (line 370)
    if_condition_27248 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 370, 4), need_cxx_27247)
    # Assigning a type to the variable 'if_condition_27248' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'if_condition_27248', if_condition_27248)
    # SSA begins for if statement (line 370)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 374)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to remove(...): (line 375)
    # Processing the call arguments (line 375)
    str_27252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 36), 'str', '-Wstrict-prototypes')
    # Processing the call keyword arguments (line 375)
    kwargs_27253 = {}
    # Getting the type of 'self' (line 375)
    self_27249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 12), 'self', False)
    # Obtaining the member 'compiler_so' of a type (line 375)
    compiler_so_27250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 12), self_27249, 'compiler_so')
    # Obtaining the member 'remove' of a type (line 375)
    remove_27251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 12), compiler_so_27250, 'remove')
    # Calling remove(args, kwargs) (line 375)
    remove_call_result_27254 = invoke(stypy.reporting.localization.Localization(__file__, 375, 12), remove_27251, *[str_27252], **kwargs_27253)
    
    # SSA branch for the except part of a try statement (line 374)
    # SSA branch for the except 'Tuple' branch of a try statement (line 374)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 374)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Call to hasattr(...): (line 379)
    # Processing the call arguments (line 379)
    # Getting the type of 'self' (line 379)
    self_27256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 19), 'self', False)
    str_27257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 25), 'str', 'compiler')
    # Processing the call keyword arguments (line 379)
    kwargs_27258 = {}
    # Getting the type of 'hasattr' (line 379)
    hasattr_27255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 11), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 379)
    hasattr_call_result_27259 = invoke(stypy.reporting.localization.Localization(__file__, 379, 11), hasattr_27255, *[self_27256, str_27257], **kwargs_27258)
    
    
    str_27260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 41), 'str', 'cc')
    
    # Obtaining the type of the subscript
    int_27261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 63), 'int')
    # Getting the type of 'self' (line 379)
    self_27262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 49), 'self')
    # Obtaining the member 'compiler' of a type (line 379)
    compiler_27263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 49), self_27262, 'compiler')
    # Obtaining the member '__getitem__' of a type (line 379)
    getitem___27264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 49), compiler_27263, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 379)
    subscript_call_result_27265 = invoke(stypy.reporting.localization.Localization(__file__, 379, 49), getitem___27264, int_27261)
    
    # Applying the binary operator 'in' (line 379)
    result_contains_27266 = python_operator(stypy.reporting.localization.Localization(__file__, 379, 41), 'in', str_27260, subscript_call_result_27265)
    
    # Applying the binary operator 'and' (line 379)
    result_and_keyword_27267 = python_operator(stypy.reporting.localization.Localization(__file__, 379, 11), 'and', hasattr_call_result_27259, result_contains_27266)
    
    # Testing the type of an if condition (line 379)
    if_condition_27268 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 379, 8), result_and_keyword_27267)
    # Assigning a type to the variable 'if_condition_27268' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'if_condition_27268', if_condition_27268)
    # SSA begins for if statement (line 379)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'self' (line 380)
    self_27269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 19), 'self')
    # Obtaining the member 'compiler_cxx' of a type (line 380)
    compiler_cxx_27270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 19), self_27269, 'compiler_cxx')
    # Applying the 'not' unary operator (line 380)
    result_not__27271 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 15), 'not', compiler_cxx_27270)
    
    # Testing the type of an if condition (line 380)
    if_condition_27272 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 380, 12), result_not__27271)
    # Assigning a type to the variable 'if_condition_27272' (line 380)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 12), 'if_condition_27272', if_condition_27272)
    # SSA begins for if statement (line 380)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to startswith(...): (line 381)
    # Processing the call arguments (line 381)
    str_27279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 47), 'str', 'gcc')
    # Processing the call keyword arguments (line 381)
    kwargs_27280 = {}
    
    # Obtaining the type of the subscript
    int_27273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 33), 'int')
    # Getting the type of 'self' (line 381)
    self_27274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 19), 'self', False)
    # Obtaining the member 'compiler' of a type (line 381)
    compiler_27275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 19), self_27274, 'compiler')
    # Obtaining the member '__getitem__' of a type (line 381)
    getitem___27276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 19), compiler_27275, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 381)
    subscript_call_result_27277 = invoke(stypy.reporting.localization.Localization(__file__, 381, 19), getitem___27276, int_27273)
    
    # Obtaining the member 'startswith' of a type (line 381)
    startswith_27278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 19), subscript_call_result_27277, 'startswith')
    # Calling startswith(args, kwargs) (line 381)
    startswith_call_result_27281 = invoke(stypy.reporting.localization.Localization(__file__, 381, 19), startswith_27278, *[str_27279], **kwargs_27280)
    
    # Testing the type of an if condition (line 381)
    if_condition_27282 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 381, 16), startswith_call_result_27281)
    # Assigning a type to the variable 'if_condition_27282' (line 381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 16), 'if_condition_27282', if_condition_27282)
    # SSA begins for if statement (line 381)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Tuple (line 382):
    
    # Assigning a Str to a Name (line 382):
    
    # Assigning a Str to a Name (line 382):
    str_27283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 27), 'str', 'gcc')
    # Assigning a type to the variable 'tuple_assignment_26353' (line 382)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 20), 'tuple_assignment_26353', str_27283)
    
    # Assigning a Str to a Name (line 382):
    
    # Assigning a Str to a Name (line 382):
    str_27284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 34), 'str', 'g++')
    # Assigning a type to the variable 'tuple_assignment_26354' (line 382)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 20), 'tuple_assignment_26354', str_27284)
    
    # Assigning a Name to a Name (line 382):
    
    # Assigning a Name to a Name (line 382):
    # Getting the type of 'tuple_assignment_26353' (line 382)
    tuple_assignment_26353_27285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 20), 'tuple_assignment_26353')
    # Assigning a type to the variable 'a' (line 382)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 20), 'a', tuple_assignment_26353_27285)
    
    # Assigning a Name to a Name (line 382):
    
    # Assigning a Name to a Name (line 382):
    # Getting the type of 'tuple_assignment_26354' (line 382)
    tuple_assignment_26354_27286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 20), 'tuple_assignment_26354')
    # Assigning a type to the variable 'b' (line 382)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 23), 'b', tuple_assignment_26354_27286)
    # SSA branch for the else part of an if statement (line 381)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Tuple to a Tuple (line 384):
    
    # Assigning a Str to a Name (line 384):
    
    # Assigning a Str to a Name (line 384):
    str_27287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 27), 'str', 'cc')
    # Assigning a type to the variable 'tuple_assignment_26355' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 20), 'tuple_assignment_26355', str_27287)
    
    # Assigning a Str to a Name (line 384):
    
    # Assigning a Str to a Name (line 384):
    str_27288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 33), 'str', 'c++')
    # Assigning a type to the variable 'tuple_assignment_26356' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 20), 'tuple_assignment_26356', str_27288)
    
    # Assigning a Name to a Name (line 384):
    
    # Assigning a Name to a Name (line 384):
    # Getting the type of 'tuple_assignment_26355' (line 384)
    tuple_assignment_26355_27289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 20), 'tuple_assignment_26355')
    # Assigning a type to the variable 'a' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 20), 'a', tuple_assignment_26355_27289)
    
    # Assigning a Name to a Name (line 384):
    
    # Assigning a Name to a Name (line 384):
    # Getting the type of 'tuple_assignment_26356' (line 384)
    tuple_assignment_26356_27290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 20), 'tuple_assignment_26356')
    # Assigning a type to the variable 'b' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 23), 'b', tuple_assignment_26356_27290)
    # SSA join for if statement (line 381)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Attribute (line 385):
    
    # Assigning a BinOp to a Attribute (line 385):
    
    # Assigning a BinOp to a Attribute (line 385):
    
    # Obtaining an instance of the builtin type 'list' (line 385)
    list_27291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 385)
    # Adding element type (line 385)
    
    # Call to replace(...): (line 385)
    # Processing the call arguments (line 385)
    # Getting the type of 'a' (line 385)
    a_27298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 62), 'a', False)
    # Getting the type of 'b' (line 385)
    b_27299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 65), 'b', False)
    # Processing the call keyword arguments (line 385)
    kwargs_27300 = {}
    
    # Obtaining the type of the subscript
    int_27292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 51), 'int')
    # Getting the type of 'self' (line 385)
    self_27293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 37), 'self', False)
    # Obtaining the member 'compiler' of a type (line 385)
    compiler_27294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 37), self_27293, 'compiler')
    # Obtaining the member '__getitem__' of a type (line 385)
    getitem___27295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 37), compiler_27294, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 385)
    subscript_call_result_27296 = invoke(stypy.reporting.localization.Localization(__file__, 385, 37), getitem___27295, int_27292)
    
    # Obtaining the member 'replace' of a type (line 385)
    replace_27297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 37), subscript_call_result_27296, 'replace')
    # Calling replace(args, kwargs) (line 385)
    replace_call_result_27301 = invoke(stypy.reporting.localization.Localization(__file__, 385, 37), replace_27297, *[a_27298, b_27299], **kwargs_27300)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 36), list_27291, replace_call_result_27301)
    
    
    # Obtaining the type of the subscript
    int_27302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 52), 'int')
    slice_27303 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 386, 38), int_27302, None, None)
    # Getting the type of 'self' (line 386)
    self_27304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 38), 'self')
    # Obtaining the member 'compiler' of a type (line 386)
    compiler_27305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 38), self_27304, 'compiler')
    # Obtaining the member '__getitem__' of a type (line 386)
    getitem___27306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 38), compiler_27305, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 386)
    subscript_call_result_27307 = invoke(stypy.reporting.localization.Localization(__file__, 386, 38), getitem___27306, slice_27303)
    
    # Applying the binary operator '+' (line 385)
    result_add_27308 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 36), '+', list_27291, subscript_call_result_27307)
    
    # Getting the type of 'self' (line 385)
    self_27309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 16), 'self')
    # Setting the type of the member 'compiler_cxx' of a type (line 385)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 16), self_27309, 'compiler_cxx', result_add_27308)
    # SSA join for if statement (line 380)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 379)
    module_type_store.open_ssa_branch('else')
    
    # Type idiom detected: calculating its left and rigth part (line 388)
    str_27310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 29), 'str', 'compiler')
    # Getting the type of 'self' (line 388)
    self_27311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 23), 'self')
    
    (may_be_27312, more_types_in_union_27313) = may_provide_member(str_27310, self_27311)

    if may_be_27312:

        if more_types_in_union_27313:
            # Runtime conditional SSA (line 388)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'self' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'self', remove_not_member_provider_from_union(self_27311, 'compiler'))
        
        # Call to warn(...): (line 389)
        # Processing the call arguments (line 389)
        str_27316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 25), 'str', '#### %s #######')
        
        # Obtaining an instance of the builtin type 'tuple' (line 389)
        tuple_27317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 389)
        # Adding element type (line 389)
        # Getting the type of 'self' (line 389)
        self_27318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 46), 'self', False)
        # Obtaining the member 'compiler' of a type (line 389)
        compiler_27319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 46), self_27318, 'compiler')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 46), tuple_27317, compiler_27319)
        
        # Applying the binary operator '%' (line 389)
        result_mod_27320 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 25), '%', str_27316, tuple_27317)
        
        # Processing the call keyword arguments (line 389)
        kwargs_27321 = {}
        # Getting the type of 'log' (line 389)
        log_27314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 16), 'log', False)
        # Obtaining the member 'warn' of a type (line 389)
        warn_27315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 16), log_27314, 'warn')
        # Calling warn(args, kwargs) (line 389)
        warn_call_result_27322 = invoke(stypy.reporting.localization.Localization(__file__, 389, 16), warn_27315, *[result_mod_27320], **kwargs_27321)
        

        if more_types_in_union_27313:
            # SSA join for if statement (line 388)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 390)
    str_27323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 33), 'str', 'compiler_cxx')
    # Getting the type of 'self' (line 390)
    self_27324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 27), 'self')
    
    (may_be_27325, more_types_in_union_27326) = may_not_provide_member(str_27323, self_27324)

    if may_be_27325:

        if more_types_in_union_27326:
            # Runtime conditional SSA (line 390)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'self' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'self', remove_member_provider_from_union(self_27324, 'compiler_cxx'))
        
        # Call to warn(...): (line 391)
        # Processing the call arguments (line 391)
        str_27329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 25), 'str', 'Missing compiler_cxx fix for ')
        # Getting the type of 'self' (line 391)
        self_27330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 59), 'self', False)
        # Obtaining the member '__class__' of a type (line 391)
        class___27331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 59), self_27330, '__class__')
        # Obtaining the member '__name__' of a type (line 391)
        name___27332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 59), class___27331, '__name__')
        # Applying the binary operator '+' (line 391)
        result_add_27333 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 25), '+', str_27329, name___27332)
        
        # Processing the call keyword arguments (line 391)
        kwargs_27334 = {}
        # Getting the type of 'log' (line 391)
        log_27327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 16), 'log', False)
        # Obtaining the member 'warn' of a type (line 391)
        warn_27328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 16), log_27327, 'warn')
        # Calling warn(args, kwargs) (line 391)
        warn_call_result_27335 = invoke(stypy.reporting.localization.Localization(__file__, 391, 16), warn_27328, *[result_add_27333], **kwargs_27334)
        

        if more_types_in_union_27326:
            # SSA join for if statement (line 390)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 379)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 370)
    module_type_store = module_type_store.join_ssa_context()
    
    # Assigning a type to the variable 'stypy_return_type' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'stypy_return_type', types.NoneType)
    
    # ################# End of 'CCompiler_customize(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'CCompiler_customize' in the type store
    # Getting the type of 'stypy_return_type' (line 337)
    stypy_return_type_27336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_27336)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'CCompiler_customize'
    return stypy_return_type_27336

# Assigning a type to the variable 'CCompiler_customize' (line 337)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 0), 'CCompiler_customize', CCompiler_customize)

# Call to replace_method(...): (line 394)
# Processing the call arguments (line 394)
# Getting the type of 'CCompiler' (line 394)
CCompiler_27338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 15), 'CCompiler', False)
str_27339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 26), 'str', 'customize')
# Getting the type of 'CCompiler_customize' (line 394)
CCompiler_customize_27340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 39), 'CCompiler_customize', False)
# Processing the call keyword arguments (line 394)
kwargs_27341 = {}
# Getting the type of 'replace_method' (line 394)
replace_method_27337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 0), 'replace_method', False)
# Calling replace_method(args, kwargs) (line 394)
replace_method_call_result_27342 = invoke(stypy.reporting.localization.Localization(__file__, 394, 0), replace_method_27337, *[CCompiler_27338, str_27339, CCompiler_customize_27340], **kwargs_27341)


@norecursion
def simple_version_match(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_27343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 29), 'str', '[-.\\d]+')
    str_27344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 48), 'str', '')
    str_27345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 58), 'str', '')
    defaults = [str_27343, str_27344, str_27345]
    # Create a new context for function 'simple_version_match'
    module_type_store = module_type_store.open_function_context('simple_version_match', 396, 0, False)
    
    # Passed parameters checking function
    simple_version_match.stypy_localization = localization
    simple_version_match.stypy_type_of_self = None
    simple_version_match.stypy_type_store = module_type_store
    simple_version_match.stypy_function_name = 'simple_version_match'
    simple_version_match.stypy_param_names_list = ['pat', 'ignore', 'start']
    simple_version_match.stypy_varargs_param_name = None
    simple_version_match.stypy_kwargs_param_name = None
    simple_version_match.stypy_call_defaults = defaults
    simple_version_match.stypy_call_varargs = varargs
    simple_version_match.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_version_match', ['pat', 'ignore', 'start'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_version_match', localization, ['pat', 'ignore', 'start'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_version_match(...)' code ##################

    str_27346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, (-1)), 'str', "\n    Simple matching of version numbers, for use in CCompiler and FCompiler.\n\n    Parameters\n    ----------\n    pat : str, optional\n        A regular expression matching version numbers.\n        Default is ``r'[-.\\d]+'``.\n    ignore : str, optional\n        A regular expression matching patterns to skip.\n        Default is ``''``, in which case nothing is skipped.\n    start : str, optional\n        A regular expression matching the start of where to start looking\n        for version numbers.\n        Default is ``''``, in which case searching is started at the\n        beginning of the version string given to `matcher`.\n\n    Returns\n    -------\n    matcher : callable\n        A function that is appropriate to use as the ``.version_match``\n        attribute of a `CCompiler` class. `matcher` takes a single parameter,\n        a version string.\n\n    ")

    @norecursion
    def matcher(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'matcher'
        module_type_store = module_type_store.open_function_context('matcher', 422, 4, False)
        
        # Passed parameters checking function
        matcher.stypy_localization = localization
        matcher.stypy_type_of_self = None
        matcher.stypy_type_store = module_type_store
        matcher.stypy_function_name = 'matcher'
        matcher.stypy_param_names_list = ['self', 'version_string']
        matcher.stypy_varargs_param_name = None
        matcher.stypy_kwargs_param_name = None
        matcher.stypy_call_defaults = defaults
        matcher.stypy_call_varargs = varargs
        matcher.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'matcher', ['self', 'version_string'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'matcher', localization, ['self', 'version_string'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'matcher(...)' code ##################

        
        # Assigning a Call to a Name (line 425):
        
        # Assigning a Call to a Name (line 425):
        
        # Assigning a Call to a Name (line 425):
        
        # Call to replace(...): (line 425)
        # Processing the call arguments (line 425)
        str_27349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 48), 'str', '\n')
        str_27350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 54), 'str', ' ')
        # Processing the call keyword arguments (line 425)
        kwargs_27351 = {}
        # Getting the type of 'version_string' (line 425)
        version_string_27347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 25), 'version_string', False)
        # Obtaining the member 'replace' of a type (line 425)
        replace_27348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 25), version_string_27347, 'replace')
        # Calling replace(args, kwargs) (line 425)
        replace_call_result_27352 = invoke(stypy.reporting.localization.Localization(__file__, 425, 25), replace_27348, *[str_27349, str_27350], **kwargs_27351)
        
        # Assigning a type to the variable 'version_string' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'version_string', replace_call_result_27352)
        
        # Assigning a Num to a Name (line 426):
        
        # Assigning a Num to a Name (line 426):
        
        # Assigning a Num to a Name (line 426):
        int_27353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 14), 'int')
        # Assigning a type to the variable 'pos' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'pos', int_27353)
        
        # Getting the type of 'start' (line 427)
        start_27354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 11), 'start')
        # Testing the type of an if condition (line 427)
        if_condition_27355 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 427, 8), start_27354)
        # Assigning a type to the variable 'if_condition_27355' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'if_condition_27355', if_condition_27355)
        # SSA begins for if statement (line 427)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 428):
        
        # Assigning a Call to a Name (line 428):
        
        # Assigning a Call to a Name (line 428):
        
        # Call to match(...): (line 428)
        # Processing the call arguments (line 428)
        # Getting the type of 'start' (line 428)
        start_27358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 25), 'start', False)
        # Getting the type of 'version_string' (line 428)
        version_string_27359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 32), 'version_string', False)
        # Processing the call keyword arguments (line 428)
        kwargs_27360 = {}
        # Getting the type of 're' (line 428)
        re_27356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 16), 're', False)
        # Obtaining the member 'match' of a type (line 428)
        match_27357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 16), re_27356, 'match')
        # Calling match(args, kwargs) (line 428)
        match_call_result_27361 = invoke(stypy.reporting.localization.Localization(__file__, 428, 16), match_27357, *[start_27358, version_string_27359], **kwargs_27360)
        
        # Assigning a type to the variable 'm' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 12), 'm', match_call_result_27361)
        
        
        # Getting the type of 'm' (line 429)
        m_27362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 19), 'm')
        # Applying the 'not' unary operator (line 429)
        result_not__27363 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 15), 'not', m_27362)
        
        # Testing the type of an if condition (line 429)
        if_condition_27364 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 429, 12), result_not__27363)
        # Assigning a type to the variable 'if_condition_27364' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'if_condition_27364', if_condition_27364)
        # SSA begins for if statement (line 429)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'None' (line 430)
        None_27365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 23), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 16), 'stypy_return_type', None_27365)
        # SSA join for if statement (line 429)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 431):
        
        # Assigning a Call to a Name (line 431):
        
        # Assigning a Call to a Name (line 431):
        
        # Call to end(...): (line 431)
        # Processing the call keyword arguments (line 431)
        kwargs_27368 = {}
        # Getting the type of 'm' (line 431)
        m_27366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 18), 'm', False)
        # Obtaining the member 'end' of a type (line 431)
        end_27367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 18), m_27366, 'end')
        # Calling end(args, kwargs) (line 431)
        end_call_result_27369 = invoke(stypy.reporting.localization.Localization(__file__, 431, 18), end_27367, *[], **kwargs_27368)
        
        # Assigning a type to the variable 'pos' (line 431)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 12), 'pos', end_call_result_27369)
        # SSA join for if statement (line 427)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'True' (line 432)
        True_27370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 14), 'True')
        # Testing the type of an if condition (line 432)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 432, 8), True_27370)
        # SSA begins for while statement (line 432)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 433):
        
        # Assigning a Call to a Name (line 433):
        
        # Assigning a Call to a Name (line 433):
        
        # Call to search(...): (line 433)
        # Processing the call arguments (line 433)
        # Getting the type of 'pat' (line 433)
        pat_27373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 26), 'pat', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'pos' (line 433)
        pos_27374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 46), 'pos', False)
        slice_27375 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 433, 31), pos_27374, None, None)
        # Getting the type of 'version_string' (line 433)
        version_string_27376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 31), 'version_string', False)
        # Obtaining the member '__getitem__' of a type (line 433)
        getitem___27377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 31), version_string_27376, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 433)
        subscript_call_result_27378 = invoke(stypy.reporting.localization.Localization(__file__, 433, 31), getitem___27377, slice_27375)
        
        # Processing the call keyword arguments (line 433)
        kwargs_27379 = {}
        # Getting the type of 're' (line 433)
        re_27371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 16), 're', False)
        # Obtaining the member 'search' of a type (line 433)
        search_27372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 16), re_27371, 'search')
        # Calling search(args, kwargs) (line 433)
        search_call_result_27380 = invoke(stypy.reporting.localization.Localization(__file__, 433, 16), search_27372, *[pat_27373, subscript_call_result_27378], **kwargs_27379)
        
        # Assigning a type to the variable 'm' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 12), 'm', search_call_result_27380)
        
        
        # Getting the type of 'm' (line 434)
        m_27381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 19), 'm')
        # Applying the 'not' unary operator (line 434)
        result_not__27382 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 15), 'not', m_27381)
        
        # Testing the type of an if condition (line 434)
        if_condition_27383 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 434, 12), result_not__27382)
        # Assigning a type to the variable 'if_condition_27383' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'if_condition_27383', if_condition_27383)
        # SSA begins for if statement (line 434)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'None' (line 435)
        None_27384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 23), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 16), 'stypy_return_type', None_27384)
        # SSA join for if statement (line 434)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'ignore' (line 436)
        ignore_27385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 15), 'ignore')
        
        # Call to match(...): (line 436)
        # Processing the call arguments (line 436)
        # Getting the type of 'ignore' (line 436)
        ignore_27388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 35), 'ignore', False)
        
        # Call to group(...): (line 436)
        # Processing the call arguments (line 436)
        int_27391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 51), 'int')
        # Processing the call keyword arguments (line 436)
        kwargs_27392 = {}
        # Getting the type of 'm' (line 436)
        m_27389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 43), 'm', False)
        # Obtaining the member 'group' of a type (line 436)
        group_27390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 43), m_27389, 'group')
        # Calling group(args, kwargs) (line 436)
        group_call_result_27393 = invoke(stypy.reporting.localization.Localization(__file__, 436, 43), group_27390, *[int_27391], **kwargs_27392)
        
        # Processing the call keyword arguments (line 436)
        kwargs_27394 = {}
        # Getting the type of 're' (line 436)
        re_27386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 26), 're', False)
        # Obtaining the member 'match' of a type (line 436)
        match_27387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 26), re_27386, 'match')
        # Calling match(args, kwargs) (line 436)
        match_call_result_27395 = invoke(stypy.reporting.localization.Localization(__file__, 436, 26), match_27387, *[ignore_27388, group_call_result_27393], **kwargs_27394)
        
        # Applying the binary operator 'and' (line 436)
        result_and_keyword_27396 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 15), 'and', ignore_27385, match_call_result_27395)
        
        # Testing the type of an if condition (line 436)
        if_condition_27397 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 436, 12), result_and_keyword_27396)
        # Assigning a type to the variable 'if_condition_27397' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'if_condition_27397', if_condition_27397)
        # SSA begins for if statement (line 436)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 437):
        
        # Assigning a Call to a Name (line 437):
        
        # Assigning a Call to a Name (line 437):
        
        # Call to end(...): (line 437)
        # Processing the call keyword arguments (line 437)
        kwargs_27400 = {}
        # Getting the type of 'm' (line 437)
        m_27398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 22), 'm', False)
        # Obtaining the member 'end' of a type (line 437)
        end_27399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 22), m_27398, 'end')
        # Calling end(args, kwargs) (line 437)
        end_call_result_27401 = invoke(stypy.reporting.localization.Localization(__file__, 437, 22), end_27399, *[], **kwargs_27400)
        
        # Assigning a type to the variable 'pos' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 16), 'pos', end_call_result_27401)
        # SSA join for if statement (line 436)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 432)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to group(...): (line 440)
        # Processing the call arguments (line 440)
        int_27404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 23), 'int')
        # Processing the call keyword arguments (line 440)
        kwargs_27405 = {}
        # Getting the type of 'm' (line 440)
        m_27402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 15), 'm', False)
        # Obtaining the member 'group' of a type (line 440)
        group_27403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 15), m_27402, 'group')
        # Calling group(args, kwargs) (line 440)
        group_call_result_27406 = invoke(stypy.reporting.localization.Localization(__file__, 440, 15), group_27403, *[int_27404], **kwargs_27405)
        
        # Assigning a type to the variable 'stypy_return_type' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'stypy_return_type', group_call_result_27406)
        
        # ################# End of 'matcher(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'matcher' in the type store
        # Getting the type of 'stypy_return_type' (line 422)
        stypy_return_type_27407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27407)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'matcher'
        return stypy_return_type_27407

    # Assigning a type to the variable 'matcher' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'matcher', matcher)
    # Getting the type of 'matcher' (line 441)
    matcher_27408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 11), 'matcher')
    # Assigning a type to the variable 'stypy_return_type' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'stypy_return_type', matcher_27408)
    
    # ################# End of 'simple_version_match(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_version_match' in the type store
    # Getting the type of 'stypy_return_type' (line 396)
    stypy_return_type_27409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_27409)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_version_match'
    return stypy_return_type_27409

# Assigning a type to the variable 'simple_version_match' (line 396)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 0), 'simple_version_match', simple_version_match)

@norecursion
def CCompiler_get_version(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 443)
    False_27410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 38), 'False')
    
    # Obtaining an instance of the builtin type 'list' (line 443)
    list_27411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 55), 'list')
    # Adding type elements to the builtin type 'list' instance (line 443)
    # Adding element type (line 443)
    int_27412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 56), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 55), list_27411, int_27412)
    
    defaults = [False_27410, list_27411]
    # Create a new context for function 'CCompiler_get_version'
    module_type_store = module_type_store.open_function_context('CCompiler_get_version', 443, 0, False)
    
    # Passed parameters checking function
    CCompiler_get_version.stypy_localization = localization
    CCompiler_get_version.stypy_type_of_self = None
    CCompiler_get_version.stypy_type_store = module_type_store
    CCompiler_get_version.stypy_function_name = 'CCompiler_get_version'
    CCompiler_get_version.stypy_param_names_list = ['self', 'force', 'ok_status']
    CCompiler_get_version.stypy_varargs_param_name = None
    CCompiler_get_version.stypy_kwargs_param_name = None
    CCompiler_get_version.stypy_call_defaults = defaults
    CCompiler_get_version.stypy_call_varargs = varargs
    CCompiler_get_version.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'CCompiler_get_version', ['self', 'force', 'ok_status'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'CCompiler_get_version', localization, ['self', 'force', 'ok_status'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'CCompiler_get_version(...)' code ##################

    str_27413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, (-1)), 'str', '\n    Return compiler version, or None if compiler is not available.\n\n    Parameters\n    ----------\n    force : bool, optional\n        If True, force a new determination of the version, even if the\n        compiler already has a version attribute. Default is False.\n    ok_status : list of int, optional\n        The list of status values returned by the version look-up process\n        for which a version string is returned. If the status value is not\n        in `ok_status`, None is returned. Default is ``[0]``.\n\n    Returns\n    -------\n    version : str or None\n        Version string, in the format of `distutils.version.LooseVersion`.\n\n    ')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'force' (line 463)
    force_27414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 11), 'force')
    # Applying the 'not' unary operator (line 463)
    result_not__27415 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 7), 'not', force_27414)
    
    
    # Call to hasattr(...): (line 463)
    # Processing the call arguments (line 463)
    # Getting the type of 'self' (line 463)
    self_27417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 29), 'self', False)
    str_27418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 35), 'str', 'version')
    # Processing the call keyword arguments (line 463)
    kwargs_27419 = {}
    # Getting the type of 'hasattr' (line 463)
    hasattr_27416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 21), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 463)
    hasattr_call_result_27420 = invoke(stypy.reporting.localization.Localization(__file__, 463, 21), hasattr_27416, *[self_27417, str_27418], **kwargs_27419)
    
    # Applying the binary operator 'and' (line 463)
    result_and_keyword_27421 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 7), 'and', result_not__27415, hasattr_call_result_27420)
    
    # Testing the type of an if condition (line 463)
    if_condition_27422 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 463, 4), result_and_keyword_27421)
    # Assigning a type to the variable 'if_condition_27422' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'if_condition_27422', if_condition_27422)
    # SSA begins for if statement (line 463)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'self' (line 464)
    self_27423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 15), 'self')
    # Obtaining the member 'version' of a type (line 464)
    version_27424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 15), self_27423, 'version')
    # Assigning a type to the variable 'stypy_return_type' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'stypy_return_type', version_27424)
    # SSA join for if statement (line 463)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to find_executables(...): (line 465)
    # Processing the call keyword arguments (line 465)
    kwargs_27427 = {}
    # Getting the type of 'self' (line 465)
    self_27425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 4), 'self', False)
    # Obtaining the member 'find_executables' of a type (line 465)
    find_executables_27426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 4), self_27425, 'find_executables')
    # Calling find_executables(args, kwargs) (line 465)
    find_executables_call_result_27428 = invoke(stypy.reporting.localization.Localization(__file__, 465, 4), find_executables_27426, *[], **kwargs_27427)
    
    
    
    # SSA begins for try-except statement (line 466)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 467):
    
    # Assigning a Attribute to a Name (line 467):
    
    # Assigning a Attribute to a Name (line 467):
    # Getting the type of 'self' (line 467)
    self_27429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 22), 'self')
    # Obtaining the member 'version_cmd' of a type (line 467)
    version_cmd_27430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 22), self_27429, 'version_cmd')
    # Assigning a type to the variable 'version_cmd' (line 467)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'version_cmd', version_cmd_27430)
    # SSA branch for the except part of a try statement (line 466)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 466)
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'None' (line 469)
    None_27431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 469)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'stypy_return_type', None_27431)
    # SSA join for try-except statement (line 466)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'version_cmd' (line 470)
    version_cmd_27432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 11), 'version_cmd')
    # Applying the 'not' unary operator (line 470)
    result_not__27433 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 7), 'not', version_cmd_27432)
    
    
    
    # Obtaining the type of the subscript
    int_27434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 42), 'int')
    # Getting the type of 'version_cmd' (line 470)
    version_cmd_27435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 30), 'version_cmd')
    # Obtaining the member '__getitem__' of a type (line 470)
    getitem___27436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 30), version_cmd_27435, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 470)
    subscript_call_result_27437 = invoke(stypy.reporting.localization.Localization(__file__, 470, 30), getitem___27436, int_27434)
    
    # Applying the 'not' unary operator (line 470)
    result_not__27438 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 26), 'not', subscript_call_result_27437)
    
    # Applying the binary operator 'or' (line 470)
    result_or_keyword_27439 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 7), 'or', result_not__27433, result_not__27438)
    
    # Testing the type of an if condition (line 470)
    if_condition_27440 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 470, 4), result_or_keyword_27439)
    # Assigning a type to the variable 'if_condition_27440' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'if_condition_27440', if_condition_27440)
    # SSA begins for if statement (line 470)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'None' (line 471)
    None_27441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'stypy_return_type', None_27441)
    # SSA join for if statement (line 470)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 472)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 473):
    
    # Assigning a Attribute to a Name (line 473):
    
    # Assigning a Attribute to a Name (line 473):
    # Getting the type of 'self' (line 473)
    self_27442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 18), 'self')
    # Obtaining the member 'version_match' of a type (line 473)
    version_match_27443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 18), self_27442, 'version_match')
    # Assigning a type to the variable 'matcher' (line 473)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'matcher', version_match_27443)
    # SSA branch for the except part of a try statement (line 472)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 472)
    module_type_store.open_ssa_branch('except')
    
    
    # SSA begins for try-except statement (line 475)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 476):
    
    # Assigning a Attribute to a Name (line 476):
    
    # Assigning a Attribute to a Name (line 476):
    # Getting the type of 'self' (line 476)
    self_27444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 18), 'self')
    # Obtaining the member 'version_pattern' of a type (line 476)
    version_pattern_27445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 18), self_27444, 'version_pattern')
    # Assigning a type to the variable 'pat' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 12), 'pat', version_pattern_27445)
    # SSA branch for the except part of a try statement (line 475)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 475)
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'None' (line 478)
    None_27446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 19), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 12), 'stypy_return_type', None_27446)
    # SSA join for try-except statement (line 475)
    module_type_store = module_type_store.join_ssa_context()
    

    @norecursion
    def matcher(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'matcher'
        module_type_store = module_type_store.open_function_context('matcher', 479, 8, False)
        
        # Passed parameters checking function
        matcher.stypy_localization = localization
        matcher.stypy_type_of_self = None
        matcher.stypy_type_store = module_type_store
        matcher.stypy_function_name = 'matcher'
        matcher.stypy_param_names_list = ['version_string']
        matcher.stypy_varargs_param_name = None
        matcher.stypy_kwargs_param_name = None
        matcher.stypy_call_defaults = defaults
        matcher.stypy_call_varargs = varargs
        matcher.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'matcher', ['version_string'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'matcher', localization, ['version_string'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'matcher(...)' code ##################

        
        # Assigning a Call to a Name (line 480):
        
        # Assigning a Call to a Name (line 480):
        
        # Assigning a Call to a Name (line 480):
        
        # Call to match(...): (line 480)
        # Processing the call arguments (line 480)
        # Getting the type of 'pat' (line 480)
        pat_27449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 25), 'pat', False)
        # Getting the type of 'version_string' (line 480)
        version_string_27450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 30), 'version_string', False)
        # Processing the call keyword arguments (line 480)
        kwargs_27451 = {}
        # Getting the type of 're' (line 480)
        re_27447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 16), 're', False)
        # Obtaining the member 'match' of a type (line 480)
        match_27448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 16), re_27447, 'match')
        # Calling match(args, kwargs) (line 480)
        match_call_result_27452 = invoke(stypy.reporting.localization.Localization(__file__, 480, 16), match_27448, *[pat_27449, version_string_27450], **kwargs_27451)
        
        # Assigning a type to the variable 'm' (line 480)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 12), 'm', match_call_result_27452)
        
        
        # Getting the type of 'm' (line 481)
        m_27453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 19), 'm')
        # Applying the 'not' unary operator (line 481)
        result_not__27454 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 15), 'not', m_27453)
        
        # Testing the type of an if condition (line 481)
        if_condition_27455 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 481, 12), result_not__27454)
        # Assigning a type to the variable 'if_condition_27455' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 12), 'if_condition_27455', if_condition_27455)
        # SSA begins for if statement (line 481)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'None' (line 482)
        None_27456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 23), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 16), 'stypy_return_type', None_27456)
        # SSA join for if statement (line 481)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 483):
        
        # Assigning a Call to a Name (line 483):
        
        # Assigning a Call to a Name (line 483):
        
        # Call to group(...): (line 483)
        # Processing the call arguments (line 483)
        str_27459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 30), 'str', 'version')
        # Processing the call keyword arguments (line 483)
        kwargs_27460 = {}
        # Getting the type of 'm' (line 483)
        m_27457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 22), 'm', False)
        # Obtaining the member 'group' of a type (line 483)
        group_27458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 22), m_27457, 'group')
        # Calling group(args, kwargs) (line 483)
        group_call_result_27461 = invoke(stypy.reporting.localization.Localization(__file__, 483, 22), group_27458, *[str_27459], **kwargs_27460)
        
        # Assigning a type to the variable 'version' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 12), 'version', group_call_result_27461)
        # Getting the type of 'version' (line 484)
        version_27462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 19), 'version')
        # Assigning a type to the variable 'stypy_return_type' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 12), 'stypy_return_type', version_27462)
        
        # ################# End of 'matcher(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'matcher' in the type store
        # Getting the type of 'stypy_return_type' (line 479)
        stypy_return_type_27463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27463)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'matcher'
        return stypy_return_type_27463

    # Assigning a type to the variable 'matcher' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'matcher', matcher)
    # SSA join for try-except statement (line 472)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 486):
    
    # Assigning a Call to a Name:
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 486)
    # Processing the call arguments (line 486)
    # Getting the type of 'version_cmd' (line 486)
    version_cmd_27465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 34), 'version_cmd', False)
    # Processing the call keyword arguments (line 486)
    int_27466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 55), 'int')
    keyword_27467 = int_27466
    kwargs_27468 = {'use_tee': keyword_27467}
    # Getting the type of 'exec_command' (line 486)
    exec_command_27464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 21), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 486)
    exec_command_call_result_27469 = invoke(stypy.reporting.localization.Localization(__file__, 486, 21), exec_command_27464, *[version_cmd_27465], **kwargs_27468)
    
    # Assigning a type to the variable 'call_assignment_26357' (line 486)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'call_assignment_26357', exec_command_call_result_27469)
    
    # Assigning a Call to a Name (line 486):
    
    # Assigning a Call to a Name (line 486):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_27472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 4), 'int')
    # Processing the call keyword arguments
    kwargs_27473 = {}
    # Getting the type of 'call_assignment_26357' (line 486)
    call_assignment_26357_27470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'call_assignment_26357', False)
    # Obtaining the member '__getitem__' of a type (line 486)
    getitem___27471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 4), call_assignment_26357_27470, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_27474 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___27471, *[int_27472], **kwargs_27473)
    
    # Assigning a type to the variable 'call_assignment_26358' (line 486)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'call_assignment_26358', getitem___call_result_27474)
    
    # Assigning a Name to a Name (line 486):
    
    # Assigning a Name to a Name (line 486):
    # Getting the type of 'call_assignment_26358' (line 486)
    call_assignment_26358_27475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'call_assignment_26358')
    # Assigning a type to the variable 'status' (line 486)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'status', call_assignment_26358_27475)
    
    # Assigning a Call to a Name (line 486):
    
    # Assigning a Call to a Name (line 486):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_27478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 4), 'int')
    # Processing the call keyword arguments
    kwargs_27479 = {}
    # Getting the type of 'call_assignment_26357' (line 486)
    call_assignment_26357_27476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'call_assignment_26357', False)
    # Obtaining the member '__getitem__' of a type (line 486)
    getitem___27477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 4), call_assignment_26357_27476, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_27480 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___27477, *[int_27478], **kwargs_27479)
    
    # Assigning a type to the variable 'call_assignment_26359' (line 486)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'call_assignment_26359', getitem___call_result_27480)
    
    # Assigning a Name to a Name (line 486):
    
    # Assigning a Name to a Name (line 486):
    # Getting the type of 'call_assignment_26359' (line 486)
    call_assignment_26359_27481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'call_assignment_26359')
    # Assigning a type to the variable 'output' (line 486)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 12), 'output', call_assignment_26359_27481)
    
    # Assigning a Name to a Name (line 488):
    
    # Assigning a Name to a Name (line 488):
    
    # Assigning a Name to a Name (line 488):
    # Getting the type of 'None' (line 488)
    None_27482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 14), 'None')
    # Assigning a type to the variable 'version' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 4), 'version', None_27482)
    
    
    # Getting the type of 'status' (line 489)
    status_27483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 7), 'status')
    # Getting the type of 'ok_status' (line 489)
    ok_status_27484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 17), 'ok_status')
    # Applying the binary operator 'in' (line 489)
    result_contains_27485 = python_operator(stypy.reporting.localization.Localization(__file__, 489, 7), 'in', status_27483, ok_status_27484)
    
    # Testing the type of an if condition (line 489)
    if_condition_27486 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 489, 4), result_contains_27485)
    # Assigning a type to the variable 'if_condition_27486' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 4), 'if_condition_27486', if_condition_27486)
    # SSA begins for if statement (line 489)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 490):
    
    # Assigning a Call to a Name (line 490):
    
    # Assigning a Call to a Name (line 490):
    
    # Call to matcher(...): (line 490)
    # Processing the call arguments (line 490)
    # Getting the type of 'output' (line 490)
    output_27488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 26), 'output', False)
    # Processing the call keyword arguments (line 490)
    kwargs_27489 = {}
    # Getting the type of 'matcher' (line 490)
    matcher_27487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 18), 'matcher', False)
    # Calling matcher(args, kwargs) (line 490)
    matcher_call_result_27490 = invoke(stypy.reporting.localization.Localization(__file__, 490, 18), matcher_27487, *[output_27488], **kwargs_27489)
    
    # Assigning a type to the variable 'version' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'version', matcher_call_result_27490)
    
    # Getting the type of 'version' (line 491)
    version_27491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 11), 'version')
    # Testing the type of an if condition (line 491)
    if_condition_27492 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 491, 8), version_27491)
    # Assigning a type to the variable 'if_condition_27492' (line 491)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'if_condition_27492', if_condition_27492)
    # SSA begins for if statement (line 491)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 492):
    
    # Assigning a Call to a Name (line 492):
    
    # Assigning a Call to a Name (line 492):
    
    # Call to LooseVersion(...): (line 492)
    # Processing the call arguments (line 492)
    # Getting the type of 'version' (line 492)
    version_27494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 35), 'version', False)
    # Processing the call keyword arguments (line 492)
    kwargs_27495 = {}
    # Getting the type of 'LooseVersion' (line 492)
    LooseVersion_27493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 22), 'LooseVersion', False)
    # Calling LooseVersion(args, kwargs) (line 492)
    LooseVersion_call_result_27496 = invoke(stypy.reporting.localization.Localization(__file__, 492, 22), LooseVersion_27493, *[version_27494], **kwargs_27495)
    
    # Assigning a type to the variable 'version' (line 492)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 12), 'version', LooseVersion_call_result_27496)
    # SSA join for if statement (line 491)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 489)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Attribute (line 493):
    
    # Assigning a Name to a Attribute (line 493):
    
    # Assigning a Name to a Attribute (line 493):
    # Getting the type of 'version' (line 493)
    version_27497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 19), 'version')
    # Getting the type of 'self' (line 493)
    self_27498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'self')
    # Setting the type of the member 'version' of a type (line 493)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 4), self_27498, 'version', version_27497)
    # Getting the type of 'version' (line 494)
    version_27499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 11), 'version')
    # Assigning a type to the variable 'stypy_return_type' (line 494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 4), 'stypy_return_type', version_27499)
    
    # ################# End of 'CCompiler_get_version(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'CCompiler_get_version' in the type store
    # Getting the type of 'stypy_return_type' (line 443)
    stypy_return_type_27500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_27500)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'CCompiler_get_version'
    return stypy_return_type_27500

# Assigning a type to the variable 'CCompiler_get_version' (line 443)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 0), 'CCompiler_get_version', CCompiler_get_version)

# Call to replace_method(...): (line 496)
# Processing the call arguments (line 496)
# Getting the type of 'CCompiler' (line 496)
CCompiler_27502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 15), 'CCompiler', False)
str_27503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 26), 'str', 'get_version')
# Getting the type of 'CCompiler_get_version' (line 496)
CCompiler_get_version_27504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 41), 'CCompiler_get_version', False)
# Processing the call keyword arguments (line 496)
kwargs_27505 = {}
# Getting the type of 'replace_method' (line 496)
replace_method_27501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 0), 'replace_method', False)
# Calling replace_method(args, kwargs) (line 496)
replace_method_call_result_27506 = invoke(stypy.reporting.localization.Localization(__file__, 496, 0), replace_method_27501, *[CCompiler_27502, str_27503, CCompiler_get_version_27504], **kwargs_27505)


@norecursion
def CCompiler_cxx_compiler(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'CCompiler_cxx_compiler'
    module_type_store = module_type_store.open_function_context('CCompiler_cxx_compiler', 498, 0, False)
    
    # Passed parameters checking function
    CCompiler_cxx_compiler.stypy_localization = localization
    CCompiler_cxx_compiler.stypy_type_of_self = None
    CCompiler_cxx_compiler.stypy_type_store = module_type_store
    CCompiler_cxx_compiler.stypy_function_name = 'CCompiler_cxx_compiler'
    CCompiler_cxx_compiler.stypy_param_names_list = ['self']
    CCompiler_cxx_compiler.stypy_varargs_param_name = None
    CCompiler_cxx_compiler.stypy_kwargs_param_name = None
    CCompiler_cxx_compiler.stypy_call_defaults = defaults
    CCompiler_cxx_compiler.stypy_call_varargs = varargs
    CCompiler_cxx_compiler.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'CCompiler_cxx_compiler', ['self'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'CCompiler_cxx_compiler', localization, ['self'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'CCompiler_cxx_compiler(...)' code ##################

    str_27507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, (-1)), 'str', '\n    Return the C++ compiler.\n\n    Parameters\n    ----------\n    None\n\n    Returns\n    -------\n    cxx : class instance\n        The C++ compiler, as a `CCompiler` instance.\n\n    ')
    
    
    # Getting the type of 'self' (line 512)
    self_27508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 7), 'self')
    # Obtaining the member 'compiler_type' of a type (line 512)
    compiler_type_27509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 7), self_27508, 'compiler_type')
    
    # Obtaining an instance of the builtin type 'tuple' (line 512)
    tuple_27510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 512)
    # Adding element type (line 512)
    str_27511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 30), 'str', 'msvc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 30), tuple_27510, str_27511)
    # Adding element type (line 512)
    str_27512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 38), 'str', 'intelw')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 30), tuple_27510, str_27512)
    # Adding element type (line 512)
    str_27513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 48), 'str', 'intelemw')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 30), tuple_27510, str_27513)
    
    # Applying the binary operator 'in' (line 512)
    result_contains_27514 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 7), 'in', compiler_type_27509, tuple_27510)
    
    # Testing the type of an if condition (line 512)
    if_condition_27515 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 512, 4), result_contains_27514)
    # Assigning a type to the variable 'if_condition_27515' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), 'if_condition_27515', if_condition_27515)
    # SSA begins for if statement (line 512)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'self' (line 513)
    self_27516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 15), 'self')
    # Assigning a type to the variable 'stypy_return_type' (line 513)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'stypy_return_type', self_27516)
    # SSA join for if statement (line 512)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 515):
    
    # Assigning a Call to a Name (line 515):
    
    # Assigning a Call to a Name (line 515):
    
    # Call to copy(...): (line 515)
    # Processing the call arguments (line 515)
    # Getting the type of 'self' (line 515)
    self_27518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 15), 'self', False)
    # Processing the call keyword arguments (line 515)
    kwargs_27519 = {}
    # Getting the type of 'copy' (line 515)
    copy_27517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 10), 'copy', False)
    # Calling copy(args, kwargs) (line 515)
    copy_call_result_27520 = invoke(stypy.reporting.localization.Localization(__file__, 515, 10), copy_27517, *[self_27518], **kwargs_27519)
    
    # Assigning a type to the variable 'cxx' (line 515)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'cxx', copy_call_result_27520)
    
    # Assigning a BinOp to a Attribute (line 516):
    
    # Assigning a BinOp to a Attribute (line 516):
    
    # Assigning a BinOp to a Attribute (line 516):
    
    # Obtaining an instance of the builtin type 'list' (line 516)
    list_27521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 516)
    # Adding element type (line 516)
    
    # Obtaining the type of the subscript
    int_27522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 40), 'int')
    # Getting the type of 'cxx' (line 516)
    cxx_27523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 23), 'cxx')
    # Obtaining the member 'compiler_cxx' of a type (line 516)
    compiler_cxx_27524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 23), cxx_27523, 'compiler_cxx')
    # Obtaining the member '__getitem__' of a type (line 516)
    getitem___27525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 23), compiler_cxx_27524, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 516)
    subscript_call_result_27526 = invoke(stypy.reporting.localization.Localization(__file__, 516, 23), getitem___27525, int_27522)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 22), list_27521, subscript_call_result_27526)
    
    
    # Obtaining the type of the subscript
    int_27527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 62), 'int')
    slice_27528 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 516, 46), int_27527, None, None)
    # Getting the type of 'cxx' (line 516)
    cxx_27529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 46), 'cxx')
    # Obtaining the member 'compiler_so' of a type (line 516)
    compiler_so_27530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 46), cxx_27529, 'compiler_so')
    # Obtaining the member '__getitem__' of a type (line 516)
    getitem___27531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 46), compiler_so_27530, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 516)
    subscript_call_result_27532 = invoke(stypy.reporting.localization.Localization(__file__, 516, 46), getitem___27531, slice_27528)
    
    # Applying the binary operator '+' (line 516)
    result_add_27533 = python_operator(stypy.reporting.localization.Localization(__file__, 516, 22), '+', list_27521, subscript_call_result_27532)
    
    # Getting the type of 'cxx' (line 516)
    cxx_27534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), 'cxx')
    # Setting the type of the member 'compiler_so' of a type (line 516)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 4), cxx_27534, 'compiler_so', result_add_27533)
    
    
    # Evaluating a boolean operation
    
    # Call to startswith(...): (line 517)
    # Processing the call arguments (line 517)
    str_27538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 31), 'str', 'aix')
    # Processing the call keyword arguments (line 517)
    kwargs_27539 = {}
    # Getting the type of 'sys' (line 517)
    sys_27535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 7), 'sys', False)
    # Obtaining the member 'platform' of a type (line 517)
    platform_27536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 7), sys_27535, 'platform')
    # Obtaining the member 'startswith' of a type (line 517)
    startswith_27537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 7), platform_27536, 'startswith')
    # Calling startswith(args, kwargs) (line 517)
    startswith_call_result_27540 = invoke(stypy.reporting.localization.Localization(__file__, 517, 7), startswith_27537, *[str_27538], **kwargs_27539)
    
    
    str_27541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 42), 'str', 'ld_so_aix')
    
    # Obtaining the type of the subscript
    int_27542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 71), 'int')
    # Getting the type of 'cxx' (line 517)
    cxx_27543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 57), 'cxx')
    # Obtaining the member 'linker_so' of a type (line 517)
    linker_so_27544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 57), cxx_27543, 'linker_so')
    # Obtaining the member '__getitem__' of a type (line 517)
    getitem___27545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 57), linker_so_27544, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 517)
    subscript_call_result_27546 = invoke(stypy.reporting.localization.Localization(__file__, 517, 57), getitem___27545, int_27542)
    
    # Applying the binary operator 'in' (line 517)
    result_contains_27547 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 42), 'in', str_27541, subscript_call_result_27546)
    
    # Applying the binary operator 'and' (line 517)
    result_and_keyword_27548 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 7), 'and', startswith_call_result_27540, result_contains_27547)
    
    # Testing the type of an if condition (line 517)
    if_condition_27549 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 517, 4), result_and_keyword_27548)
    # Assigning a type to the variable 'if_condition_27549' (line 517)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 4), 'if_condition_27549', if_condition_27549)
    # SSA begins for if statement (line 517)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Attribute (line 519):
    
    # Assigning a BinOp to a Attribute (line 519):
    
    # Assigning a BinOp to a Attribute (line 519):
    
    # Obtaining an instance of the builtin type 'list' (line 519)
    list_27550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 519)
    # Adding element type (line 519)
    
    # Obtaining the type of the subscript
    int_27551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 39), 'int')
    # Getting the type of 'cxx' (line 519)
    cxx_27552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 25), 'cxx')
    # Obtaining the member 'linker_so' of a type (line 519)
    linker_so_27553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 25), cxx_27552, 'linker_so')
    # Obtaining the member '__getitem__' of a type (line 519)
    getitem___27554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 25), linker_so_27553, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 519)
    subscript_call_result_27555 = invoke(stypy.reporting.localization.Localization(__file__, 519, 25), getitem___27554, int_27551)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 24), list_27550, subscript_call_result_27555)
    # Adding element type (line 519)
    
    # Obtaining the type of the subscript
    int_27556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 60), 'int')
    # Getting the type of 'cxx' (line 519)
    cxx_27557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 43), 'cxx')
    # Obtaining the member 'compiler_cxx' of a type (line 519)
    compiler_cxx_27558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 43), cxx_27557, 'compiler_cxx')
    # Obtaining the member '__getitem__' of a type (line 519)
    getitem___27559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 43), compiler_cxx_27558, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 519)
    subscript_call_result_27560 = invoke(stypy.reporting.localization.Localization(__file__, 519, 43), getitem___27559, int_27556)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 24), list_27550, subscript_call_result_27560)
    
    
    # Obtaining the type of the subscript
    int_27561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 40), 'int')
    slice_27562 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 520, 26), int_27561, None, None)
    # Getting the type of 'cxx' (line 520)
    cxx_27563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 26), 'cxx')
    # Obtaining the member 'linker_so' of a type (line 520)
    linker_so_27564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 26), cxx_27563, 'linker_so')
    # Obtaining the member '__getitem__' of a type (line 520)
    getitem___27565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 26), linker_so_27564, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 520)
    subscript_call_result_27566 = invoke(stypy.reporting.localization.Localization(__file__, 520, 26), getitem___27565, slice_27562)
    
    # Applying the binary operator '+' (line 519)
    result_add_27567 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 24), '+', list_27550, subscript_call_result_27566)
    
    # Getting the type of 'cxx' (line 519)
    cxx_27568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'cxx')
    # Setting the type of the member 'linker_so' of a type (line 519)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 8), cxx_27568, 'linker_so', result_add_27567)
    # SSA branch for the else part of an if statement (line 517)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Attribute (line 522):
    
    # Assigning a BinOp to a Attribute (line 522):
    
    # Assigning a BinOp to a Attribute (line 522):
    
    # Obtaining an instance of the builtin type 'list' (line 522)
    list_27569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 522)
    # Adding element type (line 522)
    
    # Obtaining the type of the subscript
    int_27570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 42), 'int')
    # Getting the type of 'cxx' (line 522)
    cxx_27571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 25), 'cxx')
    # Obtaining the member 'compiler_cxx' of a type (line 522)
    compiler_cxx_27572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 25), cxx_27571, 'compiler_cxx')
    # Obtaining the member '__getitem__' of a type (line 522)
    getitem___27573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 25), compiler_cxx_27572, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 522)
    subscript_call_result_27574 = invoke(stypy.reporting.localization.Localization(__file__, 522, 25), getitem___27573, int_27570)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 24), list_27569, subscript_call_result_27574)
    
    
    # Obtaining the type of the subscript
    int_27575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 62), 'int')
    slice_27576 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 522, 48), int_27575, None, None)
    # Getting the type of 'cxx' (line 522)
    cxx_27577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 48), 'cxx')
    # Obtaining the member 'linker_so' of a type (line 522)
    linker_so_27578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 48), cxx_27577, 'linker_so')
    # Obtaining the member '__getitem__' of a type (line 522)
    getitem___27579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 48), linker_so_27578, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 522)
    subscript_call_result_27580 = invoke(stypy.reporting.localization.Localization(__file__, 522, 48), getitem___27579, slice_27576)
    
    # Applying the binary operator '+' (line 522)
    result_add_27581 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 24), '+', list_27569, subscript_call_result_27580)
    
    # Getting the type of 'cxx' (line 522)
    cxx_27582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'cxx')
    # Setting the type of the member 'linker_so' of a type (line 522)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 8), cxx_27582, 'linker_so', result_add_27581)
    # SSA join for if statement (line 517)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'cxx' (line 523)
    cxx_27583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 11), 'cxx')
    # Assigning a type to the variable 'stypy_return_type' (line 523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 4), 'stypy_return_type', cxx_27583)
    
    # ################# End of 'CCompiler_cxx_compiler(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'CCompiler_cxx_compiler' in the type store
    # Getting the type of 'stypy_return_type' (line 498)
    stypy_return_type_27584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_27584)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'CCompiler_cxx_compiler'
    return stypy_return_type_27584

# Assigning a type to the variable 'CCompiler_cxx_compiler' (line 498)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 0), 'CCompiler_cxx_compiler', CCompiler_cxx_compiler)

# Call to replace_method(...): (line 525)
# Processing the call arguments (line 525)
# Getting the type of 'CCompiler' (line 525)
CCompiler_27586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 15), 'CCompiler', False)
str_27587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 26), 'str', 'cxx_compiler')
# Getting the type of 'CCompiler_cxx_compiler' (line 525)
CCompiler_cxx_compiler_27588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 42), 'CCompiler_cxx_compiler', False)
# Processing the call keyword arguments (line 525)
kwargs_27589 = {}
# Getting the type of 'replace_method' (line 525)
replace_method_27585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 0), 'replace_method', False)
# Calling replace_method(args, kwargs) (line 525)
replace_method_call_result_27590 = invoke(stypy.reporting.localization.Localization(__file__, 525, 0), replace_method_27585, *[CCompiler_27586, str_27587, CCompiler_cxx_compiler_27588], **kwargs_27589)


# Assigning a Tuple to a Subscript (line 527):

# Assigning a Tuple to a Subscript (line 527):

# Assigning a Tuple to a Subscript (line 527):

# Obtaining an instance of the builtin type 'tuple' (line 527)
tuple_27591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 27), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 527)
# Adding element type (line 527)
str_27592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 27), 'str', 'intelccompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 27), tuple_27591, str_27592)
# Adding element type (line 527)
str_27593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 45), 'str', 'IntelCCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 27), tuple_27591, str_27593)
# Adding element type (line 527)
str_27594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 27), 'str', 'Intel C Compiler for 32-bit applications')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 27), tuple_27591, str_27594)

# Getting the type of 'compiler_class' (line 527)
compiler_class_27595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 0), 'compiler_class')
str_27596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 15), 'str', 'intel')
# Storing an element on a container (line 527)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 0), compiler_class_27595, (str_27596, tuple_27591))

# Assigning a Tuple to a Subscript (line 529):

# Assigning a Tuple to a Subscript (line 529):

# Assigning a Tuple to a Subscript (line 529):

# Obtaining an instance of the builtin type 'tuple' (line 529)
tuple_27597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 529)
# Adding element type (line 529)
str_27598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 28), 'str', 'intelccompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 28), tuple_27597, str_27598)
# Adding element type (line 529)
str_27599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 46), 'str', 'IntelItaniumCCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 28), tuple_27597, str_27599)
# Adding element type (line 529)
str_27600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 28), 'str', 'Intel C Itanium Compiler for Itanium-based applications')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 28), tuple_27597, str_27600)

# Getting the type of 'compiler_class' (line 529)
compiler_class_27601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 0), 'compiler_class')
str_27602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 15), 'str', 'intele')
# Storing an element on a container (line 529)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 0), compiler_class_27601, (str_27602, tuple_27597))

# Assigning a Tuple to a Subscript (line 531):

# Assigning a Tuple to a Subscript (line 531):

# Assigning a Tuple to a Subscript (line 531):

# Obtaining an instance of the builtin type 'tuple' (line 531)
tuple_27603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 29), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 531)
# Adding element type (line 531)
str_27604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 29), 'str', 'intelccompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 29), tuple_27603, str_27604)
# Adding element type (line 531)
str_27605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 47), 'str', 'IntelEM64TCCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 29), tuple_27603, str_27605)
# Adding element type (line 531)
str_27606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 29), 'str', 'Intel C Compiler for 64-bit applications')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 29), tuple_27603, str_27606)

# Getting the type of 'compiler_class' (line 531)
compiler_class_27607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 0), 'compiler_class')
str_27608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 15), 'str', 'intelem')
# Storing an element on a container (line 531)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 0), compiler_class_27607, (str_27608, tuple_27603))

# Assigning a Tuple to a Subscript (line 533):

# Assigning a Tuple to a Subscript (line 533):

# Assigning a Tuple to a Subscript (line 533):

# Obtaining an instance of the builtin type 'tuple' (line 533)
tuple_27609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 533)
# Adding element type (line 533)
str_27610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 28), 'str', 'intelccompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 28), tuple_27609, str_27610)
# Adding element type (line 533)
str_27611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 46), 'str', 'IntelCCompilerW')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 28), tuple_27609, str_27611)
# Adding element type (line 533)
str_27612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 28), 'str', 'Intel C Compiler for 32-bit applications on Windows')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 28), tuple_27609, str_27612)

# Getting the type of 'compiler_class' (line 533)
compiler_class_27613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 0), 'compiler_class')
str_27614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 15), 'str', 'intelw')
# Storing an element on a container (line 533)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 0), compiler_class_27613, (str_27614, tuple_27609))

# Assigning a Tuple to a Subscript (line 535):

# Assigning a Tuple to a Subscript (line 535):

# Assigning a Tuple to a Subscript (line 535):

# Obtaining an instance of the builtin type 'tuple' (line 535)
tuple_27615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 30), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 535)
# Adding element type (line 535)
str_27616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 30), 'str', 'intelccompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 30), tuple_27615, str_27616)
# Adding element type (line 535)
str_27617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 48), 'str', 'IntelEM64TCCompilerW')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 30), tuple_27615, str_27617)
# Adding element type (line 535)
str_27618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 30), 'str', 'Intel C Compiler for 64-bit applications on Windows')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 30), tuple_27615, str_27618)

# Getting the type of 'compiler_class' (line 535)
compiler_class_27619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 0), 'compiler_class')
str_27620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 15), 'str', 'intelemw')
# Storing an element on a container (line 535)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 0), compiler_class_27619, (str_27620, tuple_27615))

# Assigning a Tuple to a Subscript (line 537):

# Assigning a Tuple to a Subscript (line 537):

# Assigning a Tuple to a Subscript (line 537):

# Obtaining an instance of the builtin type 'tuple' (line 537)
tuple_27621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 537)
# Adding element type (line 537)
str_27622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 28), 'str', 'pathccompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 28), tuple_27621, str_27622)
# Adding element type (line 537)
str_27623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 45), 'str', 'PathScaleCCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 28), tuple_27621, str_27623)
# Adding element type (line 537)
str_27624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 28), 'str', 'PathScale Compiler for SiCortex-based applications')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 28), tuple_27621, str_27624)

# Getting the type of 'compiler_class' (line 537)
compiler_class_27625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 0), 'compiler_class')
str_27626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 15), 'str', 'pathcc')
# Storing an element on a container (line 537)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 0), compiler_class_27625, (str_27626, tuple_27621))

# Getting the type of 'ccompiler' (line 539)
ccompiler_27627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 0), 'ccompiler')
# Obtaining the member '_default_compilers' of a type (line 539)
_default_compilers_27628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 0), ccompiler_27627, '_default_compilers')

# Obtaining an instance of the builtin type 'tuple' (line 539)
tuple_27629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 33), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 539)
# Adding element type (line 539)

# Obtaining an instance of the builtin type 'tuple' (line 539)
tuple_27630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 34), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 539)
# Adding element type (line 539)
str_27631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 34), 'str', 'linux.*')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 539, 34), tuple_27630, str_27631)
# Adding element type (line 539)
str_27632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 45), 'str', 'intel')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 539, 34), tuple_27630, str_27632)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 539, 33), tuple_27629, tuple_27630)
# Adding element type (line 539)

# Obtaining an instance of the builtin type 'tuple' (line 540)
tuple_27633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 34), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 540)
# Adding element type (line 540)
str_27634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 34), 'str', 'linux.*')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 540, 34), tuple_27633, str_27634)
# Adding element type (line 540)
str_27635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 45), 'str', 'intele')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 540, 34), tuple_27633, str_27635)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 539, 33), tuple_27629, tuple_27633)
# Adding element type (line 539)

# Obtaining an instance of the builtin type 'tuple' (line 541)
tuple_27636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 34), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 541)
# Adding element type (line 541)
str_27637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 34), 'str', 'linux.*')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 541, 34), tuple_27636, str_27637)
# Adding element type (line 541)
str_27638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 45), 'str', 'intelem')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 541, 34), tuple_27636, str_27638)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 539, 33), tuple_27629, tuple_27636)
# Adding element type (line 539)

# Obtaining an instance of the builtin type 'tuple' (line 542)
tuple_27639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 34), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 542)
# Adding element type (line 542)
str_27640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 34), 'str', 'linux.*')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 34), tuple_27639, str_27640)
# Adding element type (line 542)
str_27641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 45), 'str', 'pathcc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 34), tuple_27639, str_27641)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 539, 33), tuple_27629, tuple_27639)
# Adding element type (line 539)

# Obtaining an instance of the builtin type 'tuple' (line 543)
tuple_27642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 34), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 543)
# Adding element type (line 543)
str_27643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 34), 'str', 'nt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 543, 34), tuple_27642, str_27643)
# Adding element type (line 543)
str_27644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 40), 'str', 'intelw')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 543, 34), tuple_27642, str_27644)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 539, 33), tuple_27629, tuple_27642)
# Adding element type (line 539)

# Obtaining an instance of the builtin type 'tuple' (line 544)
tuple_27645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 34), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 544)
# Adding element type (line 544)
str_27646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 34), 'str', 'nt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 544, 34), tuple_27645, str_27646)
# Adding element type (line 544)
str_27647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 40), 'str', 'intelemw')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 544, 34), tuple_27645, str_27647)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 539, 33), tuple_27629, tuple_27645)

# Applying the binary operator '+=' (line 539)
result_iadd_27648 = python_operator(stypy.reporting.localization.Localization(__file__, 539, 0), '+=', _default_compilers_27628, tuple_27629)
# Getting the type of 'ccompiler' (line 539)
ccompiler_27649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 0), 'ccompiler')
# Setting the type of the member '_default_compilers' of a type (line 539)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 0), ccompiler_27649, '_default_compilers', result_iadd_27648)



# Getting the type of 'sys' (line 546)
sys_27650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 3), 'sys')
# Obtaining the member 'platform' of a type (line 546)
platform_27651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 3), sys_27650, 'platform')
str_27652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 19), 'str', 'win32')
# Applying the binary operator '==' (line 546)
result_eq_27653 = python_operator(stypy.reporting.localization.Localization(__file__, 546, 3), '==', platform_27651, str_27652)

# Testing the type of an if condition (line 546)
if_condition_27654 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 546, 0), result_eq_27653)
# Assigning a type to the variable 'if_condition_27654' (line 546)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 0), 'if_condition_27654', if_condition_27654)
# SSA begins for if statement (line 546)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Tuple to a Subscript (line 547):

# Assigning a Tuple to a Subscript (line 547):

# Assigning a Tuple to a Subscript (line 547):

# Obtaining an instance of the builtin type 'tuple' (line 547)
tuple_27655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 33), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 547)
# Adding element type (line 547)
str_27656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 33), 'str', 'mingw32ccompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 547, 33), tuple_27655, str_27656)
# Adding element type (line 547)
str_27657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 53), 'str', 'Mingw32CCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 547, 33), tuple_27655, str_27657)
# Adding element type (line 547)
str_27658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 33), 'str', 'Mingw32 port of GNU C Compiler for Win32(for MSC built Python)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 547, 33), tuple_27655, str_27658)

# Getting the type of 'compiler_class' (line 547)
compiler_class_27659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 4), 'compiler_class')
str_27660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 19), 'str', 'mingw32')
# Storing an element on a container (line 547)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 547, 4), compiler_class_27659, (str_27660, tuple_27655))


# Call to mingw32(...): (line 550)
# Processing the call keyword arguments (line 550)
kwargs_27662 = {}
# Getting the type of 'mingw32' (line 550)
mingw32_27661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 7), 'mingw32', False)
# Calling mingw32(args, kwargs) (line 550)
mingw32_call_result_27663 = invoke(stypy.reporting.localization.Localization(__file__, 550, 7), mingw32_27661, *[], **kwargs_27662)

# Testing the type of an if condition (line 550)
if_condition_27664 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 550, 4), mingw32_call_result_27663)
# Assigning a type to the variable 'if_condition_27664' (line 550)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), 'if_condition_27664', if_condition_27664)
# SSA begins for if statement (line 550)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to info(...): (line 553)
# Processing the call arguments (line 553)
str_27667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 17), 'str', 'Setting mingw32 as default compiler for nt.')
# Processing the call keyword arguments (line 553)
kwargs_27668 = {}
# Getting the type of 'log' (line 553)
log_27665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'log', False)
# Obtaining the member 'info' of a type (line 553)
info_27666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 8), log_27665, 'info')
# Calling info(args, kwargs) (line 553)
info_call_result_27669 = invoke(stypy.reporting.localization.Localization(__file__, 553, 8), info_27666, *[str_27667], **kwargs_27668)


# Assigning a BinOp to a Attribute (line 554):

# Assigning a BinOp to a Attribute (line 554):

# Assigning a BinOp to a Attribute (line 554):

# Obtaining an instance of the builtin type 'tuple' (line 554)
tuple_27670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 40), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 554)
# Adding element type (line 554)

# Obtaining an instance of the builtin type 'tuple' (line 554)
tuple_27671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 41), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 554)
# Adding element type (line 554)
str_27672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 41), 'str', 'nt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 554, 41), tuple_27671, str_27672)
# Adding element type (line 554)
str_27673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 47), 'str', 'mingw32')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 554, 41), tuple_27671, str_27673)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 554, 40), tuple_27670, tuple_27671)

# Getting the type of 'ccompiler' (line 555)
ccompiler_27674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 41), 'ccompiler')
# Obtaining the member '_default_compilers' of a type (line 555)
_default_compilers_27675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 41), ccompiler_27674, '_default_compilers')
# Applying the binary operator '+' (line 554)
result_add_27676 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 39), '+', tuple_27670, _default_compilers_27675)

# Getting the type of 'ccompiler' (line 554)
ccompiler_27677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 8), 'ccompiler')
# Setting the type of the member '_default_compilers' of a type (line 554)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 8), ccompiler_27677, '_default_compilers', result_add_27676)
# SSA join for if statement (line 550)
module_type_store = module_type_store.join_ssa_context()

# SSA join for if statement (line 546)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Name to a Name (line 558):

# Assigning a Name to a Name (line 558):

# Assigning a Name to a Name (line 558):
# Getting the type of 'new_compiler' (line 558)
new_compiler_27678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 26), 'new_compiler')
# Assigning a type to the variable '_distutils_new_compiler' (line 558)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 0), '_distutils_new_compiler', new_compiler_27678)

@norecursion
def new_compiler(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 559)
    None_27679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 23), 'None')
    # Getting the type of 'None' (line 560)
    None_27680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 27), 'None')
    int_27681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 26), 'int')
    int_27682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 26), 'int')
    int_27683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 24), 'int')
    defaults = [None_27679, None_27680, int_27681, int_27682, int_27683]
    # Create a new context for function 'new_compiler'
    module_type_store = module_type_store.open_function_context('new_compiler', 559, 0, False)
    
    # Passed parameters checking function
    new_compiler.stypy_localization = localization
    new_compiler.stypy_type_of_self = None
    new_compiler.stypy_type_store = module_type_store
    new_compiler.stypy_function_name = 'new_compiler'
    new_compiler.stypy_param_names_list = ['plat', 'compiler', 'verbose', 'dry_run', 'force']
    new_compiler.stypy_varargs_param_name = None
    new_compiler.stypy_kwargs_param_name = None
    new_compiler.stypy_call_defaults = defaults
    new_compiler.stypy_call_varargs = varargs
    new_compiler.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'new_compiler', ['plat', 'compiler', 'verbose', 'dry_run', 'force'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'new_compiler', localization, ['plat', 'compiler', 'verbose', 'dry_run', 'force'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'new_compiler(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 565)
    # Getting the type of 'plat' (line 565)
    plat_27684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 7), 'plat')
    # Getting the type of 'None' (line 565)
    None_27685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 15), 'None')
    
    (may_be_27686, more_types_in_union_27687) = may_be_none(plat_27684, None_27685)

    if may_be_27686:

        if more_types_in_union_27687:
            # Runtime conditional SSA (line 565)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 566):
        
        # Assigning a Attribute to a Name (line 566):
        
        # Assigning a Attribute to a Name (line 566):
        # Getting the type of 'os' (line 566)
        os_27688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 15), 'os')
        # Obtaining the member 'name' of a type (line 566)
        name_27689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 15), os_27688, 'name')
        # Assigning a type to the variable 'plat' (line 566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'plat', name_27689)

        if more_types_in_union_27687:
            # SSA join for if statement (line 565)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # SSA begins for try-except statement (line 567)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Type idiom detected: calculating its left and rigth part (line 568)
    # Getting the type of 'compiler' (line 568)
    compiler_27690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 11), 'compiler')
    # Getting the type of 'None' (line 568)
    None_27691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 23), 'None')
    
    (may_be_27692, more_types_in_union_27693) = may_be_none(compiler_27690, None_27691)

    if may_be_27692:

        if more_types_in_union_27693:
            # Runtime conditional SSA (line 568)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 569):
        
        # Assigning a Call to a Name (line 569):
        
        # Assigning a Call to a Name (line 569):
        
        # Call to get_default_compiler(...): (line 569)
        # Processing the call arguments (line 569)
        # Getting the type of 'plat' (line 569)
        plat_27695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 44), 'plat', False)
        # Processing the call keyword arguments (line 569)
        kwargs_27696 = {}
        # Getting the type of 'get_default_compiler' (line 569)
        get_default_compiler_27694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 23), 'get_default_compiler', False)
        # Calling get_default_compiler(args, kwargs) (line 569)
        get_default_compiler_call_result_27697 = invoke(stypy.reporting.localization.Localization(__file__, 569, 23), get_default_compiler_27694, *[plat_27695], **kwargs_27696)
        
        # Assigning a type to the variable 'compiler' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 12), 'compiler', get_default_compiler_call_result_27697)

        if more_types_in_union_27693:
            # SSA join for if statement (line 568)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Subscript to a Tuple (line 570):
    
    # Assigning a Subscript to a Name (line 570):
    
    # Assigning a Subscript to a Name (line 570):
    
    # Obtaining the type of the subscript
    int_27698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 8), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'compiler' (line 570)
    compiler_27699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 69), 'compiler')
    # Getting the type of 'compiler_class' (line 570)
    compiler_class_27700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 54), 'compiler_class')
    # Obtaining the member '__getitem__' of a type (line 570)
    getitem___27701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 54), compiler_class_27700, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 570)
    subscript_call_result_27702 = invoke(stypy.reporting.localization.Localization(__file__, 570, 54), getitem___27701, compiler_27699)
    
    # Obtaining the member '__getitem__' of a type (line 570)
    getitem___27703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 8), subscript_call_result_27702, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 570)
    subscript_call_result_27704 = invoke(stypy.reporting.localization.Localization(__file__, 570, 8), getitem___27703, int_27698)
    
    # Assigning a type to the variable 'tuple_var_assignment_26360' (line 570)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'tuple_var_assignment_26360', subscript_call_result_27704)
    
    # Assigning a Subscript to a Name (line 570):
    
    # Assigning a Subscript to a Name (line 570):
    
    # Obtaining the type of the subscript
    int_27705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 8), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'compiler' (line 570)
    compiler_27706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 69), 'compiler')
    # Getting the type of 'compiler_class' (line 570)
    compiler_class_27707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 54), 'compiler_class')
    # Obtaining the member '__getitem__' of a type (line 570)
    getitem___27708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 54), compiler_class_27707, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 570)
    subscript_call_result_27709 = invoke(stypy.reporting.localization.Localization(__file__, 570, 54), getitem___27708, compiler_27706)
    
    # Obtaining the member '__getitem__' of a type (line 570)
    getitem___27710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 8), subscript_call_result_27709, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 570)
    subscript_call_result_27711 = invoke(stypy.reporting.localization.Localization(__file__, 570, 8), getitem___27710, int_27705)
    
    # Assigning a type to the variable 'tuple_var_assignment_26361' (line 570)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'tuple_var_assignment_26361', subscript_call_result_27711)
    
    # Assigning a Subscript to a Name (line 570):
    
    # Assigning a Subscript to a Name (line 570):
    
    # Obtaining the type of the subscript
    int_27712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 8), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'compiler' (line 570)
    compiler_27713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 69), 'compiler')
    # Getting the type of 'compiler_class' (line 570)
    compiler_class_27714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 54), 'compiler_class')
    # Obtaining the member '__getitem__' of a type (line 570)
    getitem___27715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 54), compiler_class_27714, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 570)
    subscript_call_result_27716 = invoke(stypy.reporting.localization.Localization(__file__, 570, 54), getitem___27715, compiler_27713)
    
    # Obtaining the member '__getitem__' of a type (line 570)
    getitem___27717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 8), subscript_call_result_27716, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 570)
    subscript_call_result_27718 = invoke(stypy.reporting.localization.Localization(__file__, 570, 8), getitem___27717, int_27712)
    
    # Assigning a type to the variable 'tuple_var_assignment_26362' (line 570)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'tuple_var_assignment_26362', subscript_call_result_27718)
    
    # Assigning a Name to a Name (line 570):
    
    # Assigning a Name to a Name (line 570):
    # Getting the type of 'tuple_var_assignment_26360' (line 570)
    tuple_var_assignment_26360_27719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'tuple_var_assignment_26360')
    # Assigning a type to the variable 'module_name' (line 570)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 9), 'module_name', tuple_var_assignment_26360_27719)
    
    # Assigning a Name to a Name (line 570):
    
    # Assigning a Name to a Name (line 570):
    # Getting the type of 'tuple_var_assignment_26361' (line 570)
    tuple_var_assignment_26361_27720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'tuple_var_assignment_26361')
    # Assigning a type to the variable 'class_name' (line 570)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 22), 'class_name', tuple_var_assignment_26361_27720)
    
    # Assigning a Name to a Name (line 570):
    
    # Assigning a Name to a Name (line 570):
    # Getting the type of 'tuple_var_assignment_26362' (line 570)
    tuple_var_assignment_26362_27721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'tuple_var_assignment_26362')
    # Assigning a type to the variable 'long_description' (line 570)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 34), 'long_description', tuple_var_assignment_26362_27721)
    # SSA branch for the except part of a try statement (line 567)
    # SSA branch for the except 'KeyError' branch of a try statement (line 567)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a BinOp to a Name (line 572):
    
    # Assigning a BinOp to a Name (line 572):
    
    # Assigning a BinOp to a Name (line 572):
    str_27722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 14), 'str', "don't know how to compile C/C++ code on platform '%s'")
    # Getting the type of 'plat' (line 572)
    plat_27723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 72), 'plat')
    # Applying the binary operator '%' (line 572)
    result_mod_27724 = python_operator(stypy.reporting.localization.Localization(__file__, 572, 14), '%', str_27722, plat_27723)
    
    # Assigning a type to the variable 'msg' (line 572)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'msg', result_mod_27724)
    
    # Type idiom detected: calculating its left and rigth part (line 573)
    # Getting the type of 'compiler' (line 573)
    compiler_27725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'compiler')
    # Getting the type of 'None' (line 573)
    None_27726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 27), 'None')
    
    (may_be_27727, more_types_in_union_27728) = may_not_be_none(compiler_27725, None_27726)

    if may_be_27727:

        if more_types_in_union_27728:
            # Runtime conditional SSA (line 573)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 574):
        
        # Assigning a BinOp to a Name (line 574):
        
        # Assigning a BinOp to a Name (line 574):
        # Getting the type of 'msg' (line 574)
        msg_27729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 18), 'msg')
        str_27730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 24), 'str', " with '%s' compiler")
        # Getting the type of 'compiler' (line 574)
        compiler_27731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 48), 'compiler')
        # Applying the binary operator '%' (line 574)
        result_mod_27732 = python_operator(stypy.reporting.localization.Localization(__file__, 574, 24), '%', str_27730, compiler_27731)
        
        # Applying the binary operator '+' (line 574)
        result_add_27733 = python_operator(stypy.reporting.localization.Localization(__file__, 574, 18), '+', msg_27729, result_mod_27732)
        
        # Assigning a type to the variable 'msg' (line 574)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 12), 'msg', result_add_27733)

        if more_types_in_union_27728:
            # SSA join for if statement (line 573)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to DistutilsPlatformError(...): (line 575)
    # Processing the call arguments (line 575)
    # Getting the type of 'msg' (line 575)
    msg_27735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 37), 'msg', False)
    # Processing the call keyword arguments (line 575)
    kwargs_27736 = {}
    # Getting the type of 'DistutilsPlatformError' (line 575)
    DistutilsPlatformError_27734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 14), 'DistutilsPlatformError', False)
    # Calling DistutilsPlatformError(args, kwargs) (line 575)
    DistutilsPlatformError_call_result_27737 = invoke(stypy.reporting.localization.Localization(__file__, 575, 14), DistutilsPlatformError_27734, *[msg_27735], **kwargs_27736)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 575, 8), DistutilsPlatformError_call_result_27737, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 567)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 576):
    
    # Assigning a BinOp to a Name (line 576):
    
    # Assigning a BinOp to a Name (line 576):
    str_27738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 18), 'str', 'numpy.distutils.')
    # Getting the type of 'module_name' (line 576)
    module_name_27739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 39), 'module_name')
    # Applying the binary operator '+' (line 576)
    result_add_27740 = python_operator(stypy.reporting.localization.Localization(__file__, 576, 18), '+', str_27738, module_name_27739)
    
    # Assigning a type to the variable 'module_name' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 4), 'module_name', result_add_27740)
    
    
    # SSA begins for try-except statement (line 577)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to __import__(...): (line 578)
    # Processing the call arguments (line 578)
    # Getting the type of 'module_name' (line 578)
    module_name_27742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 20), 'module_name', False)
    # Processing the call keyword arguments (line 578)
    kwargs_27743 = {}
    # Getting the type of '__import__' (line 578)
    import___27741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 8), '__import__', False)
    # Calling __import__(args, kwargs) (line 578)
    import___call_result_27744 = invoke(stypy.reporting.localization.Localization(__file__, 578, 8), import___27741, *[module_name_27742], **kwargs_27743)
    
    # SSA branch for the except part of a try statement (line 577)
    # SSA branch for the except 'ImportError' branch of a try statement (line 577)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Name (line 580):
    
    # Assigning a Call to a Name (line 580):
    
    # Assigning a Call to a Name (line 580):
    
    # Call to str(...): (line 580)
    # Processing the call arguments (line 580)
    
    # Call to get_exception(...): (line 580)
    # Processing the call keyword arguments (line 580)
    kwargs_27747 = {}
    # Getting the type of 'get_exception' (line 580)
    get_exception_27746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 18), 'get_exception', False)
    # Calling get_exception(args, kwargs) (line 580)
    get_exception_call_result_27748 = invoke(stypy.reporting.localization.Localization(__file__, 580, 18), get_exception_27746, *[], **kwargs_27747)
    
    # Processing the call keyword arguments (line 580)
    kwargs_27749 = {}
    # Getting the type of 'str' (line 580)
    str_27745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 14), 'str', False)
    # Calling str(args, kwargs) (line 580)
    str_call_result_27750 = invoke(stypy.reporting.localization.Localization(__file__, 580, 14), str_27745, *[get_exception_call_result_27748], **kwargs_27749)
    
    # Assigning a type to the variable 'msg' (line 580)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 8), 'msg', str_call_result_27750)
    
    # Call to info(...): (line 581)
    # Processing the call arguments (line 581)
    str_27753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 17), 'str', '%s in numpy.distutils; trying from distutils')
    
    # Call to str(...): (line 582)
    # Processing the call arguments (line 582)
    # Getting the type of 'msg' (line 582)
    msg_27755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 21), 'msg', False)
    # Processing the call keyword arguments (line 582)
    kwargs_27756 = {}
    # Getting the type of 'str' (line 582)
    str_27754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 17), 'str', False)
    # Calling str(args, kwargs) (line 582)
    str_call_result_27757 = invoke(stypy.reporting.localization.Localization(__file__, 582, 17), str_27754, *[msg_27755], **kwargs_27756)
    
    # Processing the call keyword arguments (line 581)
    kwargs_27758 = {}
    # Getting the type of 'log' (line 581)
    log_27751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'log', False)
    # Obtaining the member 'info' of a type (line 581)
    info_27752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 8), log_27751, 'info')
    # Calling info(args, kwargs) (line 581)
    info_call_result_27759 = invoke(stypy.reporting.localization.Localization(__file__, 581, 8), info_27752, *[str_27753, str_call_result_27757], **kwargs_27758)
    
    
    # Assigning a Subscript to a Name (line 583):
    
    # Assigning a Subscript to a Name (line 583):
    
    # Assigning a Subscript to a Name (line 583):
    
    # Obtaining the type of the subscript
    int_27760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 34), 'int')
    slice_27761 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 583, 22), int_27760, None, None)
    # Getting the type of 'module_name' (line 583)
    module_name_27762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 22), 'module_name')
    # Obtaining the member '__getitem__' of a type (line 583)
    getitem___27763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 22), module_name_27762, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 583)
    subscript_call_result_27764 = invoke(stypy.reporting.localization.Localization(__file__, 583, 22), getitem___27763, slice_27761)
    
    # Assigning a type to the variable 'module_name' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 8), 'module_name', subscript_call_result_27764)
    
    
    # SSA begins for try-except statement (line 584)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to __import__(...): (line 585)
    # Processing the call arguments (line 585)
    # Getting the type of 'module_name' (line 585)
    module_name_27766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 23), 'module_name', False)
    # Processing the call keyword arguments (line 585)
    kwargs_27767 = {}
    # Getting the type of '__import__' (line 585)
    import___27765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 12), '__import__', False)
    # Calling __import__(args, kwargs) (line 585)
    import___call_result_27768 = invoke(stypy.reporting.localization.Localization(__file__, 585, 12), import___27765, *[module_name_27766], **kwargs_27767)
    
    # SSA branch for the except part of a try statement (line 584)
    # SSA branch for the except 'ImportError' branch of a try statement (line 584)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Name (line 587):
    
    # Assigning a Call to a Name (line 587):
    
    # Assigning a Call to a Name (line 587):
    
    # Call to str(...): (line 587)
    # Processing the call arguments (line 587)
    
    # Call to get_exception(...): (line 587)
    # Processing the call keyword arguments (line 587)
    kwargs_27771 = {}
    # Getting the type of 'get_exception' (line 587)
    get_exception_27770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 22), 'get_exception', False)
    # Calling get_exception(args, kwargs) (line 587)
    get_exception_call_result_27772 = invoke(stypy.reporting.localization.Localization(__file__, 587, 22), get_exception_27770, *[], **kwargs_27771)
    
    # Processing the call keyword arguments (line 587)
    kwargs_27773 = {}
    # Getting the type of 'str' (line 587)
    str_27769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 18), 'str', False)
    # Calling str(args, kwargs) (line 587)
    str_call_result_27774 = invoke(stypy.reporting.localization.Localization(__file__, 587, 18), str_27769, *[get_exception_call_result_27772], **kwargs_27773)
    
    # Assigning a type to the variable 'msg' (line 587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 12), 'msg', str_call_result_27774)
    
    # Call to DistutilsModuleError(...): (line 588)
    # Processing the call arguments (line 588)
    str_27776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 39), 'str', "can't compile C/C++ code: unable to load module '%s'")
    # Getting the type of 'module_name' (line 589)
    module_name_27777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 18), 'module_name', False)
    # Applying the binary operator '%' (line 588)
    result_mod_27778 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 39), '%', str_27776, module_name_27777)
    
    # Processing the call keyword arguments (line 588)
    kwargs_27779 = {}
    # Getting the type of 'DistutilsModuleError' (line 588)
    DistutilsModuleError_27775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 18), 'DistutilsModuleError', False)
    # Calling DistutilsModuleError(args, kwargs) (line 588)
    DistutilsModuleError_call_result_27780 = invoke(stypy.reporting.localization.Localization(__file__, 588, 18), DistutilsModuleError_27775, *[result_mod_27778], **kwargs_27779)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 588, 12), DistutilsModuleError_call_result_27780, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 584)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 577)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 590)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 591):
    
    # Assigning a Subscript to a Name (line 591):
    
    # Assigning a Subscript to a Name (line 591):
    
    # Obtaining the type of the subscript
    # Getting the type of 'module_name' (line 591)
    module_name_27781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 29), 'module_name')
    # Getting the type of 'sys' (line 591)
    sys_27782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 17), 'sys')
    # Obtaining the member 'modules' of a type (line 591)
    modules_27783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 17), sys_27782, 'modules')
    # Obtaining the member '__getitem__' of a type (line 591)
    getitem___27784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 17), modules_27783, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 591)
    subscript_call_result_27785 = invoke(stypy.reporting.localization.Localization(__file__, 591, 17), getitem___27784, module_name_27781)
    
    # Assigning a type to the variable 'module' (line 591)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 8), 'module', subscript_call_result_27785)
    
    # Assigning a Subscript to a Name (line 592):
    
    # Assigning a Subscript to a Name (line 592):
    
    # Assigning a Subscript to a Name (line 592):
    
    # Obtaining the type of the subscript
    # Getting the type of 'class_name' (line 592)
    class_name_27786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 29), 'class_name')
    
    # Call to vars(...): (line 592)
    # Processing the call arguments (line 592)
    # Getting the type of 'module' (line 592)
    module_27788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 21), 'module', False)
    # Processing the call keyword arguments (line 592)
    kwargs_27789 = {}
    # Getting the type of 'vars' (line 592)
    vars_27787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 16), 'vars', False)
    # Calling vars(args, kwargs) (line 592)
    vars_call_result_27790 = invoke(stypy.reporting.localization.Localization(__file__, 592, 16), vars_27787, *[module_27788], **kwargs_27789)
    
    # Obtaining the member '__getitem__' of a type (line 592)
    getitem___27791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 16), vars_call_result_27790, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 592)
    subscript_call_result_27792 = invoke(stypy.reporting.localization.Localization(__file__, 592, 16), getitem___27791, class_name_27786)
    
    # Assigning a type to the variable 'klass' (line 592)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'klass', subscript_call_result_27792)
    # SSA branch for the except part of a try statement (line 590)
    # SSA branch for the except 'KeyError' branch of a try statement (line 590)
    module_type_store.open_ssa_branch('except')
    
    # Call to DistutilsModuleError(...): (line 594)
    # Processing the call arguments (line 594)
    str_27794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 36), 'str', "can't compile C/C++ code: unable to find class '%s' ")
    str_27795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 15), 'str', "in module '%s'")
    # Applying the binary operator '+' (line 594)
    result_add_27796 = python_operator(stypy.reporting.localization.Localization(__file__, 594, 36), '+', str_27794, str_27795)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 595)
    tuple_27797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 595)
    # Adding element type (line 595)
    # Getting the type of 'class_name' (line 595)
    class_name_27798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 36), 'class_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 595, 36), tuple_27797, class_name_27798)
    # Adding element type (line 595)
    # Getting the type of 'module_name' (line 595)
    module_name_27799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 48), 'module_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 595, 36), tuple_27797, module_name_27799)
    
    # Applying the binary operator '%' (line 594)
    result_mod_27800 = python_operator(stypy.reporting.localization.Localization(__file__, 594, 35), '%', result_add_27796, tuple_27797)
    
    # Processing the call keyword arguments (line 594)
    kwargs_27801 = {}
    # Getting the type of 'DistutilsModuleError' (line 594)
    DistutilsModuleError_27793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 14), 'DistutilsModuleError', False)
    # Calling DistutilsModuleError(args, kwargs) (line 594)
    DistutilsModuleError_call_result_27802 = invoke(stypy.reporting.localization.Localization(__file__, 594, 14), DistutilsModuleError_27793, *[result_mod_27800], **kwargs_27801)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 594, 8), DistutilsModuleError_call_result_27802, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 590)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 596):
    
    # Assigning a Call to a Name (line 596):
    
    # Assigning a Call to a Name (line 596):
    
    # Call to klass(...): (line 596)
    # Processing the call arguments (line 596)
    # Getting the type of 'None' (line 596)
    None_27804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 21), 'None', False)
    # Getting the type of 'dry_run' (line 596)
    dry_run_27805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 27), 'dry_run', False)
    # Getting the type of 'force' (line 596)
    force_27806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 36), 'force', False)
    # Processing the call keyword arguments (line 596)
    kwargs_27807 = {}
    # Getting the type of 'klass' (line 596)
    klass_27803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 15), 'klass', False)
    # Calling klass(args, kwargs) (line 596)
    klass_call_result_27808 = invoke(stypy.reporting.localization.Localization(__file__, 596, 15), klass_27803, *[None_27804, dry_run_27805, force_27806], **kwargs_27807)
    
    # Assigning a type to the variable 'compiler' (line 596)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 4), 'compiler', klass_call_result_27808)
    
    # Call to debug(...): (line 597)
    # Processing the call arguments (line 597)
    str_27811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 14), 'str', 'new_compiler returns %s')
    # Getting the type of 'klass' (line 597)
    klass_27812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 43), 'klass', False)
    # Applying the binary operator '%' (line 597)
    result_mod_27813 = python_operator(stypy.reporting.localization.Localization(__file__, 597, 14), '%', str_27811, klass_27812)
    
    # Processing the call keyword arguments (line 597)
    kwargs_27814 = {}
    # Getting the type of 'log' (line 597)
    log_27809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 4), 'log', False)
    # Obtaining the member 'debug' of a type (line 597)
    debug_27810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 4), log_27809, 'debug')
    # Calling debug(args, kwargs) (line 597)
    debug_call_result_27815 = invoke(stypy.reporting.localization.Localization(__file__, 597, 4), debug_27810, *[result_mod_27813], **kwargs_27814)
    
    # Getting the type of 'compiler' (line 598)
    compiler_27816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 11), 'compiler')
    # Assigning a type to the variable 'stypy_return_type' (line 598)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'stypy_return_type', compiler_27816)
    
    # ################# End of 'new_compiler(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'new_compiler' in the type store
    # Getting the type of 'stypy_return_type' (line 559)
    stypy_return_type_27817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_27817)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'new_compiler'
    return stypy_return_type_27817

# Assigning a type to the variable 'new_compiler' (line 559)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 0), 'new_compiler', new_compiler)

# Assigning a Name to a Attribute (line 600):

# Assigning a Name to a Attribute (line 600):

# Assigning a Name to a Attribute (line 600):
# Getting the type of 'new_compiler' (line 600)
new_compiler_27818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 25), 'new_compiler')
# Getting the type of 'ccompiler' (line 600)
ccompiler_27819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 0), 'ccompiler')
# Setting the type of the member 'new_compiler' of a type (line 600)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 0), ccompiler_27819, 'new_compiler', new_compiler_27818)

# Assigning a Name to a Name (line 602):

# Assigning a Name to a Name (line 602):

# Assigning a Name to a Name (line 602):
# Getting the type of 'gen_lib_options' (line 602)
gen_lib_options_27820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 29), 'gen_lib_options')
# Assigning a type to the variable '_distutils_gen_lib_options' (line 602)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 0), '_distutils_gen_lib_options', gen_lib_options_27820)

@norecursion
def gen_lib_options(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'gen_lib_options'
    module_type_store = module_type_store.open_function_context('gen_lib_options', 603, 0, False)
    
    # Passed parameters checking function
    gen_lib_options.stypy_localization = localization
    gen_lib_options.stypy_type_of_self = None
    gen_lib_options.stypy_type_store = module_type_store
    gen_lib_options.stypy_function_name = 'gen_lib_options'
    gen_lib_options.stypy_param_names_list = ['compiler', 'library_dirs', 'runtime_library_dirs', 'libraries']
    gen_lib_options.stypy_varargs_param_name = None
    gen_lib_options.stypy_kwargs_param_name = None
    gen_lib_options.stypy_call_defaults = defaults
    gen_lib_options.stypy_call_varargs = varargs
    gen_lib_options.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gen_lib_options', ['compiler', 'library_dirs', 'runtime_library_dirs', 'libraries'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gen_lib_options', localization, ['compiler', 'library_dirs', 'runtime_library_dirs', 'libraries'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gen_lib_options(...)' code ##################

    
    # Assigning a Call to a Name (line 604):
    
    # Assigning a Call to a Name (line 604):
    
    # Assigning a Call to a Name (line 604):
    
    # Call to quote_args(...): (line 604)
    # Processing the call arguments (line 604)
    # Getting the type of 'library_dirs' (line 604)
    library_dirs_27822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 30), 'library_dirs', False)
    # Processing the call keyword arguments (line 604)
    kwargs_27823 = {}
    # Getting the type of 'quote_args' (line 604)
    quote_args_27821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 19), 'quote_args', False)
    # Calling quote_args(args, kwargs) (line 604)
    quote_args_call_result_27824 = invoke(stypy.reporting.localization.Localization(__file__, 604, 19), quote_args_27821, *[library_dirs_27822], **kwargs_27823)
    
    # Assigning a type to the variable 'library_dirs' (line 604)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 4), 'library_dirs', quote_args_call_result_27824)
    
    # Assigning a Call to a Name (line 605):
    
    # Assigning a Call to a Name (line 605):
    
    # Assigning a Call to a Name (line 605):
    
    # Call to quote_args(...): (line 605)
    # Processing the call arguments (line 605)
    # Getting the type of 'runtime_library_dirs' (line 605)
    runtime_library_dirs_27826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 38), 'runtime_library_dirs', False)
    # Processing the call keyword arguments (line 605)
    kwargs_27827 = {}
    # Getting the type of 'quote_args' (line 605)
    quote_args_27825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 27), 'quote_args', False)
    # Calling quote_args(args, kwargs) (line 605)
    quote_args_call_result_27828 = invoke(stypy.reporting.localization.Localization(__file__, 605, 27), quote_args_27825, *[runtime_library_dirs_27826], **kwargs_27827)
    
    # Assigning a type to the variable 'runtime_library_dirs' (line 605)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 4), 'runtime_library_dirs', quote_args_call_result_27828)
    
    # Assigning a Call to a Name (line 606):
    
    # Assigning a Call to a Name (line 606):
    
    # Assigning a Call to a Name (line 606):
    
    # Call to _distutils_gen_lib_options(...): (line 606)
    # Processing the call arguments (line 606)
    # Getting the type of 'compiler' (line 606)
    compiler_27830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 35), 'compiler', False)
    # Getting the type of 'library_dirs' (line 606)
    library_dirs_27831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 45), 'library_dirs', False)
    # Getting the type of 'runtime_library_dirs' (line 607)
    runtime_library_dirs_27832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 35), 'runtime_library_dirs', False)
    # Getting the type of 'libraries' (line 607)
    libraries_27833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 57), 'libraries', False)
    # Processing the call keyword arguments (line 606)
    kwargs_27834 = {}
    # Getting the type of '_distutils_gen_lib_options' (line 606)
    _distutils_gen_lib_options_27829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 8), '_distutils_gen_lib_options', False)
    # Calling _distutils_gen_lib_options(args, kwargs) (line 606)
    _distutils_gen_lib_options_call_result_27835 = invoke(stypy.reporting.localization.Localization(__file__, 606, 8), _distutils_gen_lib_options_27829, *[compiler_27830, library_dirs_27831, runtime_library_dirs_27832, libraries_27833], **kwargs_27834)
    
    # Assigning a type to the variable 'r' (line 606)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 4), 'r', _distutils_gen_lib_options_call_result_27835)
    
    # Assigning a List to a Name (line 608):
    
    # Assigning a List to a Name (line 608):
    
    # Assigning a List to a Name (line 608):
    
    # Obtaining an instance of the builtin type 'list' (line 608)
    list_27836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 608)
    
    # Assigning a type to the variable 'lib_opts' (line 608)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 4), 'lib_opts', list_27836)
    
    # Getting the type of 'r' (line 609)
    r_27837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 13), 'r')
    # Testing the type of a for loop iterable (line 609)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 609, 4), r_27837)
    # Getting the type of the for loop variable (line 609)
    for_loop_var_27838 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 609, 4), r_27837)
    # Assigning a type to the variable 'i' (line 609)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 4), 'i', for_loop_var_27838)
    # SSA begins for a for statement (line 609)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to is_sequence(...): (line 610)
    # Processing the call arguments (line 610)
    # Getting the type of 'i' (line 610)
    i_27840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 23), 'i', False)
    # Processing the call keyword arguments (line 610)
    kwargs_27841 = {}
    # Getting the type of 'is_sequence' (line 610)
    is_sequence_27839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 11), 'is_sequence', False)
    # Calling is_sequence(args, kwargs) (line 610)
    is_sequence_call_result_27842 = invoke(stypy.reporting.localization.Localization(__file__, 610, 11), is_sequence_27839, *[i_27840], **kwargs_27841)
    
    # Testing the type of an if condition (line 610)
    if_condition_27843 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 610, 8), is_sequence_call_result_27842)
    # Assigning a type to the variable 'if_condition_27843' (line 610)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 8), 'if_condition_27843', if_condition_27843)
    # SSA begins for if statement (line 610)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to extend(...): (line 611)
    # Processing the call arguments (line 611)
    
    # Call to list(...): (line 611)
    # Processing the call arguments (line 611)
    # Getting the type of 'i' (line 611)
    i_27847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 33), 'i', False)
    # Processing the call keyword arguments (line 611)
    kwargs_27848 = {}
    # Getting the type of 'list' (line 611)
    list_27846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 28), 'list', False)
    # Calling list(args, kwargs) (line 611)
    list_call_result_27849 = invoke(stypy.reporting.localization.Localization(__file__, 611, 28), list_27846, *[i_27847], **kwargs_27848)
    
    # Processing the call keyword arguments (line 611)
    kwargs_27850 = {}
    # Getting the type of 'lib_opts' (line 611)
    lib_opts_27844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 12), 'lib_opts', False)
    # Obtaining the member 'extend' of a type (line 611)
    extend_27845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 12), lib_opts_27844, 'extend')
    # Calling extend(args, kwargs) (line 611)
    extend_call_result_27851 = invoke(stypy.reporting.localization.Localization(__file__, 611, 12), extend_27845, *[list_call_result_27849], **kwargs_27850)
    
    # SSA branch for the else part of an if statement (line 610)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 613)
    # Processing the call arguments (line 613)
    # Getting the type of 'i' (line 613)
    i_27854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 28), 'i', False)
    # Processing the call keyword arguments (line 613)
    kwargs_27855 = {}
    # Getting the type of 'lib_opts' (line 613)
    lib_opts_27852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 12), 'lib_opts', False)
    # Obtaining the member 'append' of a type (line 613)
    append_27853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 12), lib_opts_27852, 'append')
    # Calling append(args, kwargs) (line 613)
    append_call_result_27856 = invoke(stypy.reporting.localization.Localization(__file__, 613, 12), append_27853, *[i_27854], **kwargs_27855)
    
    # SSA join for if statement (line 610)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'lib_opts' (line 614)
    lib_opts_27857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 11), 'lib_opts')
    # Assigning a type to the variable 'stypy_return_type' (line 614)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 4), 'stypy_return_type', lib_opts_27857)
    
    # ################# End of 'gen_lib_options(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gen_lib_options' in the type store
    # Getting the type of 'stypy_return_type' (line 603)
    stypy_return_type_27858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_27858)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gen_lib_options'
    return stypy_return_type_27858

# Assigning a type to the variable 'gen_lib_options' (line 603)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 0), 'gen_lib_options', gen_lib_options)

# Assigning a Name to a Attribute (line 615):

# Assigning a Name to a Attribute (line 615):

# Assigning a Name to a Attribute (line 615):
# Getting the type of 'gen_lib_options' (line 615)
gen_lib_options_27859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 28), 'gen_lib_options')
# Getting the type of 'ccompiler' (line 615)
ccompiler_27860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 0), 'ccompiler')
# Setting the type of the member 'gen_lib_options' of a type (line 615)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 0), ccompiler_27860, 'gen_lib_options', gen_lib_options_27859)


# Obtaining an instance of the builtin type 'list' (line 620)
list_27861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 11), 'list')
# Adding type elements to the builtin type 'list' instance (line 620)
# Adding element type (line 620)
str_27862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 12), 'str', 'msvc9')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 620, 11), list_27861, str_27862)
# Adding element type (line 620)
str_27863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 21), 'str', 'msvc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 620, 11), list_27861, str_27863)
# Adding element type (line 620)
str_27864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 29), 'str', 'bcpp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 620, 11), list_27861, str_27864)
# Adding element type (line 620)
str_27865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 37), 'str', 'cygwinc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 620, 11), list_27861, str_27865)
# Adding element type (line 620)
str_27866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 48), 'str', 'emxc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 620, 11), list_27861, str_27866)
# Adding element type (line 620)
str_27867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 56), 'str', 'unixc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 620, 11), list_27861, str_27867)

# Testing the type of a for loop iterable (line 620)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 620, 0), list_27861)
# Getting the type of the for loop variable (line 620)
for_loop_var_27868 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 620, 0), list_27861)
# Assigning a type to the variable '_cc' (line 620)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 0), '_cc', for_loop_var_27868)
# SSA begins for a for statement (line 620)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Assigning a Call to a Name (line 621):

# Assigning a Call to a Name (line 621):

# Assigning a Call to a Name (line 621):

# Call to get(...): (line 621)
# Processing the call arguments (line 621)
str_27872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 25), 'str', 'distutils.')
# Getting the type of '_cc' (line 621)
_cc_27873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 40), '_cc', False)
# Applying the binary operator '+' (line 621)
result_add_27874 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 25), '+', str_27872, _cc_27873)

str_27875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 46), 'str', 'compiler')
# Applying the binary operator '+' (line 621)
result_add_27876 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 44), '+', result_add_27874, str_27875)

# Processing the call keyword arguments (line 621)
kwargs_27877 = {}
# Getting the type of 'sys' (line 621)
sys_27869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 9), 'sys', False)
# Obtaining the member 'modules' of a type (line 621)
modules_27870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 9), sys_27869, 'modules')
# Obtaining the member 'get' of a type (line 621)
get_27871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 9), modules_27870, 'get')
# Calling get(args, kwargs) (line 621)
get_call_result_27878 = invoke(stypy.reporting.localization.Localization(__file__, 621, 9), get_27871, *[result_add_27876], **kwargs_27877)

# Assigning a type to the variable '_m' (line 621)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 4), '_m', get_call_result_27878)

# Type idiom detected: calculating its left and rigth part (line 622)
# Getting the type of '_m' (line 622)
_m_27879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 4), '_m')
# Getting the type of 'None' (line 622)
None_27880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 17), 'None')

(may_be_27881, more_types_in_union_27882) = may_not_be_none(_m_27879, None_27880)

if may_be_27881:

    if more_types_in_union_27882:
        # Runtime conditional SSA (line 622)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
    else:
        module_type_store = module_type_store

    
    # Call to setattr(...): (line 623)
    # Processing the call arguments (line 623)
    # Getting the type of '_m' (line 623)
    _m_27884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 16), '_m', False)
    str_27885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 20), 'str', 'gen_lib_options')
    # Getting the type of 'gen_lib_options' (line 623)
    gen_lib_options_27886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 39), 'gen_lib_options', False)
    # Processing the call keyword arguments (line 623)
    kwargs_27887 = {}
    # Getting the type of 'setattr' (line 623)
    setattr_27883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 8), 'setattr', False)
    # Calling setattr(args, kwargs) (line 623)
    setattr_call_result_27888 = invoke(stypy.reporting.localization.Localization(__file__, 623, 8), setattr_27883, *[_m_27884, str_27885, gen_lib_options_27886], **kwargs_27887)
    

    if more_types_in_union_27882:
        # SSA join for if statement (line 622)
        module_type_store = module_type_store.join_ssa_context()



# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# Assigning a Name to a Name (line 625):

# Assigning a Name to a Name (line 625):

# Assigning a Name to a Name (line 625):
# Getting the type of 'gen_preprocess_options' (line 625)
gen_preprocess_options_27889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 36), 'gen_preprocess_options')
# Assigning a type to the variable '_distutils_gen_preprocess_options' (line 625)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 0), '_distutils_gen_preprocess_options', gen_preprocess_options_27889)

@norecursion
def gen_preprocess_options(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'gen_preprocess_options'
    module_type_store = module_type_store.open_function_context('gen_preprocess_options', 626, 0, False)
    
    # Passed parameters checking function
    gen_preprocess_options.stypy_localization = localization
    gen_preprocess_options.stypy_type_of_self = None
    gen_preprocess_options.stypy_type_store = module_type_store
    gen_preprocess_options.stypy_function_name = 'gen_preprocess_options'
    gen_preprocess_options.stypy_param_names_list = ['macros', 'include_dirs']
    gen_preprocess_options.stypy_varargs_param_name = None
    gen_preprocess_options.stypy_kwargs_param_name = None
    gen_preprocess_options.stypy_call_defaults = defaults
    gen_preprocess_options.stypy_call_varargs = varargs
    gen_preprocess_options.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gen_preprocess_options', ['macros', 'include_dirs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gen_preprocess_options', localization, ['macros', 'include_dirs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gen_preprocess_options(...)' code ##################

    
    # Assigning a Call to a Name (line 627):
    
    # Assigning a Call to a Name (line 627):
    
    # Assigning a Call to a Name (line 627):
    
    # Call to quote_args(...): (line 627)
    # Processing the call arguments (line 627)
    # Getting the type of 'include_dirs' (line 627)
    include_dirs_27891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 30), 'include_dirs', False)
    # Processing the call keyword arguments (line 627)
    kwargs_27892 = {}
    # Getting the type of 'quote_args' (line 627)
    quote_args_27890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 19), 'quote_args', False)
    # Calling quote_args(args, kwargs) (line 627)
    quote_args_call_result_27893 = invoke(stypy.reporting.localization.Localization(__file__, 627, 19), quote_args_27890, *[include_dirs_27891], **kwargs_27892)
    
    # Assigning a type to the variable 'include_dirs' (line 627)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 4), 'include_dirs', quote_args_call_result_27893)
    
    # Call to _distutils_gen_preprocess_options(...): (line 628)
    # Processing the call arguments (line 628)
    # Getting the type of 'macros' (line 628)
    macros_27895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 45), 'macros', False)
    # Getting the type of 'include_dirs' (line 628)
    include_dirs_27896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 53), 'include_dirs', False)
    # Processing the call keyword arguments (line 628)
    kwargs_27897 = {}
    # Getting the type of '_distutils_gen_preprocess_options' (line 628)
    _distutils_gen_preprocess_options_27894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 11), '_distutils_gen_preprocess_options', False)
    # Calling _distutils_gen_preprocess_options(args, kwargs) (line 628)
    _distutils_gen_preprocess_options_call_result_27898 = invoke(stypy.reporting.localization.Localization(__file__, 628, 11), _distutils_gen_preprocess_options_27894, *[macros_27895, include_dirs_27896], **kwargs_27897)
    
    # Assigning a type to the variable 'stypy_return_type' (line 628)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 4), 'stypy_return_type', _distutils_gen_preprocess_options_call_result_27898)
    
    # ################# End of 'gen_preprocess_options(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gen_preprocess_options' in the type store
    # Getting the type of 'stypy_return_type' (line 626)
    stypy_return_type_27899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_27899)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gen_preprocess_options'
    return stypy_return_type_27899

# Assigning a type to the variable 'gen_preprocess_options' (line 626)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 0), 'gen_preprocess_options', gen_preprocess_options)

# Assigning a Name to a Attribute (line 629):

# Assigning a Name to a Attribute (line 629):

# Assigning a Name to a Attribute (line 629):
# Getting the type of 'gen_preprocess_options' (line 629)
gen_preprocess_options_27900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 35), 'gen_preprocess_options')
# Getting the type of 'ccompiler' (line 629)
ccompiler_27901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 0), 'ccompiler')
# Setting the type of the member 'gen_preprocess_options' of a type (line 629)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 0), ccompiler_27901, 'gen_preprocess_options', gen_preprocess_options_27900)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 636, 0))

# 'import string' statement (line 636)
import string

import_module(stypy.reporting.localization.Localization(__file__, 636, 0), 'string', string, module_type_store)


# Assigning a Call to a Name (line 637):

# Assigning a Call to a Name (line 637):

# Assigning a Call to a Name (line 637):

# Call to compile(...): (line 637)
# Processing the call arguments (line 637)
str_27904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 27), 'str', '[^\\\\\\\'\\"%s ]*')
# Getting the type of 'string' (line 637)
string_27905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 46), 'string', False)
# Obtaining the member 'whitespace' of a type (line 637)
whitespace_27906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 46), string_27905, 'whitespace')
# Applying the binary operator '%' (line 637)
result_mod_27907 = python_operator(stypy.reporting.localization.Localization(__file__, 637, 27), '%', str_27904, whitespace_27906)

# Processing the call keyword arguments (line 637)
kwargs_27908 = {}
# Getting the type of 're' (line 637)
re_27902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 16), 're', False)
# Obtaining the member 'compile' of a type (line 637)
compile_27903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 16), re_27902, 'compile')
# Calling compile(args, kwargs) (line 637)
compile_call_result_27909 = invoke(stypy.reporting.localization.Localization(__file__, 637, 16), compile_27903, *[result_mod_27907], **kwargs_27908)

# Assigning a type to the variable '_wordchars_re' (line 637)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 0), '_wordchars_re', compile_call_result_27909)

# Assigning a Call to a Name (line 638):

# Assigning a Call to a Name (line 638):

# Assigning a Call to a Name (line 638):

# Call to compile(...): (line 638)
# Processing the call arguments (line 638)
str_27912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 24), 'str', "'(?:[^'\\\\]|\\\\.)*'")
# Processing the call keyword arguments (line 638)
kwargs_27913 = {}
# Getting the type of 're' (line 638)
re_27910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 13), 're', False)
# Obtaining the member 'compile' of a type (line 638)
compile_27911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 13), re_27910, 'compile')
# Calling compile(args, kwargs) (line 638)
compile_call_result_27914 = invoke(stypy.reporting.localization.Localization(__file__, 638, 13), compile_27911, *[str_27912], **kwargs_27913)

# Assigning a type to the variable '_squote_re' (line 638)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 0), '_squote_re', compile_call_result_27914)

# Assigning a Call to a Name (line 639):

# Assigning a Call to a Name (line 639):

# Assigning a Call to a Name (line 639):

# Call to compile(...): (line 639)
# Processing the call arguments (line 639)
str_27917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 24), 'str', '"(?:[^"\\\\]|\\\\.)*"')
# Processing the call keyword arguments (line 639)
kwargs_27918 = {}
# Getting the type of 're' (line 639)
re_27915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 13), 're', False)
# Obtaining the member 'compile' of a type (line 639)
compile_27916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 13), re_27915, 'compile')
# Calling compile(args, kwargs) (line 639)
compile_call_result_27919 = invoke(stypy.reporting.localization.Localization(__file__, 639, 13), compile_27916, *[str_27917], **kwargs_27918)

# Assigning a type to the variable '_dquote_re' (line 639)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 0), '_dquote_re', compile_call_result_27919)

# Assigning a Call to a Name (line 640):

# Assigning a Call to a Name (line 640):

# Assigning a Call to a Name (line 640):

# Call to compile(...): (line 640)
# Processing the call arguments (line 640)
str_27922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 27), 'str', '\\s')
# Processing the call keyword arguments (line 640)
kwargs_27923 = {}
# Getting the type of 're' (line 640)
re_27920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 16), 're', False)
# Obtaining the member 'compile' of a type (line 640)
compile_27921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 16), re_27920, 'compile')
# Calling compile(args, kwargs) (line 640)
compile_call_result_27924 = invoke(stypy.reporting.localization.Localization(__file__, 640, 16), compile_27921, *[str_27922], **kwargs_27923)

# Assigning a type to the variable '_has_white_re' (line 640)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 0), '_has_white_re', compile_call_result_27924)

@norecursion
def split_quoted(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'split_quoted'
    module_type_store = module_type_store.open_function_context('split_quoted', 641, 0, False)
    
    # Passed parameters checking function
    split_quoted.stypy_localization = localization
    split_quoted.stypy_type_of_self = None
    split_quoted.stypy_type_store = module_type_store
    split_quoted.stypy_function_name = 'split_quoted'
    split_quoted.stypy_param_names_list = ['s']
    split_quoted.stypy_varargs_param_name = None
    split_quoted.stypy_kwargs_param_name = None
    split_quoted.stypy_call_defaults = defaults
    split_quoted.stypy_call_varargs = varargs
    split_quoted.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'split_quoted', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'split_quoted', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'split_quoted(...)' code ##################

    
    # Assigning a Call to a Name (line 642):
    
    # Assigning a Call to a Name (line 642):
    
    # Assigning a Call to a Name (line 642):
    
    # Call to strip(...): (line 642)
    # Processing the call keyword arguments (line 642)
    kwargs_27927 = {}
    # Getting the type of 's' (line 642)
    s_27925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 8), 's', False)
    # Obtaining the member 'strip' of a type (line 642)
    strip_27926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 8), s_27925, 'strip')
    # Calling strip(args, kwargs) (line 642)
    strip_call_result_27928 = invoke(stypy.reporting.localization.Localization(__file__, 642, 8), strip_27926, *[], **kwargs_27927)
    
    # Assigning a type to the variable 's' (line 642)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 4), 's', strip_call_result_27928)
    
    # Assigning a List to a Name (line 643):
    
    # Assigning a List to a Name (line 643):
    
    # Assigning a List to a Name (line 643):
    
    # Obtaining an instance of the builtin type 'list' (line 643)
    list_27929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 643)
    
    # Assigning a type to the variable 'words' (line 643)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 4), 'words', list_27929)
    
    # Assigning a Num to a Name (line 644):
    
    # Assigning a Num to a Name (line 644):
    
    # Assigning a Num to a Name (line 644):
    int_27930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, 10), 'int')
    # Assigning a type to the variable 'pos' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 4), 'pos', int_27930)
    
    # Getting the type of 's' (line 646)
    s_27931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 10), 's')
    # Testing the type of an if condition (line 646)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 646, 4), s_27931)
    # SSA begins for while statement (line 646)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 647):
    
    # Assigning a Call to a Name (line 647):
    
    # Assigning a Call to a Name (line 647):
    
    # Call to match(...): (line 647)
    # Processing the call arguments (line 647)
    # Getting the type of 's' (line 647)
    s_27934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 32), 's', False)
    # Getting the type of 'pos' (line 647)
    pos_27935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 35), 'pos', False)
    # Processing the call keyword arguments (line 647)
    kwargs_27936 = {}
    # Getting the type of '_wordchars_re' (line 647)
    _wordchars_re_27932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 12), '_wordchars_re', False)
    # Obtaining the member 'match' of a type (line 647)
    match_27933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 12), _wordchars_re_27932, 'match')
    # Calling match(args, kwargs) (line 647)
    match_call_result_27937 = invoke(stypy.reporting.localization.Localization(__file__, 647, 12), match_27933, *[s_27934, pos_27935], **kwargs_27936)
    
    # Assigning a type to the variable 'm' (line 647)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 8), 'm', match_call_result_27937)
    
    # Assigning a Call to a Name (line 648):
    
    # Assigning a Call to a Name (line 648):
    
    # Assigning a Call to a Name (line 648):
    
    # Call to end(...): (line 648)
    # Processing the call keyword arguments (line 648)
    kwargs_27940 = {}
    # Getting the type of 'm' (line 648)
    m_27938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 14), 'm', False)
    # Obtaining the member 'end' of a type (line 648)
    end_27939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 14), m_27938, 'end')
    # Calling end(args, kwargs) (line 648)
    end_call_result_27941 = invoke(stypy.reporting.localization.Localization(__file__, 648, 14), end_27939, *[], **kwargs_27940)
    
    # Assigning a type to the variable 'end' (line 648)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 8), 'end', end_call_result_27941)
    
    
    # Getting the type of 'end' (line 649)
    end_27942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 11), 'end')
    
    # Call to len(...): (line 649)
    # Processing the call arguments (line 649)
    # Getting the type of 's' (line 649)
    s_27944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 22), 's', False)
    # Processing the call keyword arguments (line 649)
    kwargs_27945 = {}
    # Getting the type of 'len' (line 649)
    len_27943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 18), 'len', False)
    # Calling len(args, kwargs) (line 649)
    len_call_result_27946 = invoke(stypy.reporting.localization.Localization(__file__, 649, 18), len_27943, *[s_27944], **kwargs_27945)
    
    # Applying the binary operator '==' (line 649)
    result_eq_27947 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 11), '==', end_27942, len_call_result_27946)
    
    # Testing the type of an if condition (line 649)
    if_condition_27948 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 649, 8), result_eq_27947)
    # Assigning a type to the variable 'if_condition_27948' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 8), 'if_condition_27948', if_condition_27948)
    # SSA begins for if statement (line 649)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 650)
    # Processing the call arguments (line 650)
    
    # Obtaining the type of the subscript
    # Getting the type of 'end' (line 650)
    end_27951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 28), 'end', False)
    slice_27952 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 650, 25), None, end_27951, None)
    # Getting the type of 's' (line 650)
    s_27953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 25), 's', False)
    # Obtaining the member '__getitem__' of a type (line 650)
    getitem___27954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 25), s_27953, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 650)
    subscript_call_result_27955 = invoke(stypy.reporting.localization.Localization(__file__, 650, 25), getitem___27954, slice_27952)
    
    # Processing the call keyword arguments (line 650)
    kwargs_27956 = {}
    # Getting the type of 'words' (line 650)
    words_27949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 12), 'words', False)
    # Obtaining the member 'append' of a type (line 650)
    append_27950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 12), words_27949, 'append')
    # Calling append(args, kwargs) (line 650)
    append_call_result_27957 = invoke(stypy.reporting.localization.Localization(__file__, 650, 12), append_27950, *[subscript_call_result_27955], **kwargs_27956)
    
    # SSA join for if statement (line 649)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'end' (line 653)
    end_27958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 13), 'end')
    # Getting the type of 's' (line 653)
    s_27959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 11), 's')
    # Obtaining the member '__getitem__' of a type (line 653)
    getitem___27960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 11), s_27959, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 653)
    subscript_call_result_27961 = invoke(stypy.reporting.localization.Localization(__file__, 653, 11), getitem___27960, end_27958)
    
    # Getting the type of 'string' (line 653)
    string_27962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 21), 'string')
    # Obtaining the member 'whitespace' of a type (line 653)
    whitespace_27963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 21), string_27962, 'whitespace')
    # Applying the binary operator 'in' (line 653)
    result_contains_27964 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 11), 'in', subscript_call_result_27961, whitespace_27963)
    
    # Testing the type of an if condition (line 653)
    if_condition_27965 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 653, 8), result_contains_27964)
    # Assigning a type to the variable 'if_condition_27965' (line 653)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 8), 'if_condition_27965', if_condition_27965)
    # SSA begins for if statement (line 653)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 654)
    # Processing the call arguments (line 654)
    
    # Obtaining the type of the subscript
    # Getting the type of 'end' (line 654)
    end_27968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 28), 'end', False)
    slice_27969 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 654, 25), None, end_27968, None)
    # Getting the type of 's' (line 654)
    s_27970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 25), 's', False)
    # Obtaining the member '__getitem__' of a type (line 654)
    getitem___27971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 654, 25), s_27970, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 654)
    subscript_call_result_27972 = invoke(stypy.reporting.localization.Localization(__file__, 654, 25), getitem___27971, slice_27969)
    
    # Processing the call keyword arguments (line 654)
    kwargs_27973 = {}
    # Getting the type of 'words' (line 654)
    words_27966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 12), 'words', False)
    # Obtaining the member 'append' of a type (line 654)
    append_27967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 654, 12), words_27966, 'append')
    # Calling append(args, kwargs) (line 654)
    append_call_result_27974 = invoke(stypy.reporting.localization.Localization(__file__, 654, 12), append_27967, *[subscript_call_result_27972], **kwargs_27973)
    
    
    # Assigning a Call to a Name (line 655):
    
    # Assigning a Call to a Name (line 655):
    
    # Assigning a Call to a Name (line 655):
    
    # Call to lstrip(...): (line 655)
    # Processing the call keyword arguments (line 655)
    kwargs_27981 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'end' (line 655)
    end_27975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 18), 'end', False)
    slice_27976 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 655, 16), end_27975, None, None)
    # Getting the type of 's' (line 655)
    s_27977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 16), 's', False)
    # Obtaining the member '__getitem__' of a type (line 655)
    getitem___27978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 16), s_27977, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 655)
    subscript_call_result_27979 = invoke(stypy.reporting.localization.Localization(__file__, 655, 16), getitem___27978, slice_27976)
    
    # Obtaining the member 'lstrip' of a type (line 655)
    lstrip_27980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 16), subscript_call_result_27979, 'lstrip')
    # Calling lstrip(args, kwargs) (line 655)
    lstrip_call_result_27982 = invoke(stypy.reporting.localization.Localization(__file__, 655, 16), lstrip_27980, *[], **kwargs_27981)
    
    # Assigning a type to the variable 's' (line 655)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 12), 's', lstrip_call_result_27982)
    
    # Assigning a Num to a Name (line 656):
    
    # Assigning a Num to a Name (line 656):
    
    # Assigning a Num to a Name (line 656):
    int_27983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 18), 'int')
    # Assigning a type to the variable 'pos' (line 656)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 12), 'pos', int_27983)
    # SSA branch for the else part of an if statement (line 653)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'end' (line 658)
    end_27984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 15), 'end')
    # Getting the type of 's' (line 658)
    s_27985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 13), 's')
    # Obtaining the member '__getitem__' of a type (line 658)
    getitem___27986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 13), s_27985, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 658)
    subscript_call_result_27987 = invoke(stypy.reporting.localization.Localization(__file__, 658, 13), getitem___27986, end_27984)
    
    str_27988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 23), 'str', '\\')
    # Applying the binary operator '==' (line 658)
    result_eq_27989 = python_operator(stypy.reporting.localization.Localization(__file__, 658, 13), '==', subscript_call_result_27987, str_27988)
    
    # Testing the type of an if condition (line 658)
    if_condition_27990 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 658, 13), result_eq_27989)
    # Assigning a type to the variable 'if_condition_27990' (line 658)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 13), 'if_condition_27990', if_condition_27990)
    # SSA begins for if statement (line 658)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 660):
    
    # Assigning a BinOp to a Name (line 660):
    
    # Assigning a BinOp to a Name (line 660):
    
    # Obtaining the type of the subscript
    # Getting the type of 'end' (line 660)
    end_27991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 19), 'end')
    slice_27992 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 660, 16), None, end_27991, None)
    # Getting the type of 's' (line 660)
    s_27993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 16), 's')
    # Obtaining the member '__getitem__' of a type (line 660)
    getitem___27994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 16), s_27993, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 660)
    subscript_call_result_27995 = invoke(stypy.reporting.localization.Localization(__file__, 660, 16), getitem___27994, slice_27992)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'end' (line 660)
    end_27996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 28), 'end')
    int_27997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 32), 'int')
    # Applying the binary operator '+' (line 660)
    result_add_27998 = python_operator(stypy.reporting.localization.Localization(__file__, 660, 28), '+', end_27996, int_27997)
    
    slice_27999 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 660, 26), result_add_27998, None, None)
    # Getting the type of 's' (line 660)
    s_28000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 26), 's')
    # Obtaining the member '__getitem__' of a type (line 660)
    getitem___28001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 26), s_28000, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 660)
    subscript_call_result_28002 = invoke(stypy.reporting.localization.Localization(__file__, 660, 26), getitem___28001, slice_27999)
    
    # Applying the binary operator '+' (line 660)
    result_add_28003 = python_operator(stypy.reporting.localization.Localization(__file__, 660, 16), '+', subscript_call_result_27995, subscript_call_result_28002)
    
    # Assigning a type to the variable 's' (line 660)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 12), 's', result_add_28003)
    
    # Assigning a BinOp to a Name (line 661):
    
    # Assigning a BinOp to a Name (line 661):
    
    # Assigning a BinOp to a Name (line 661):
    # Getting the type of 'end' (line 661)
    end_28004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 18), 'end')
    int_28005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 22), 'int')
    # Applying the binary operator '+' (line 661)
    result_add_28006 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 18), '+', end_28004, int_28005)
    
    # Assigning a type to the variable 'pos' (line 661)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 12), 'pos', result_add_28006)
    # SSA branch for the else part of an if statement (line 658)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'end' (line 664)
    end_28007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 17), 'end')
    # Getting the type of 's' (line 664)
    s_28008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 15), 's')
    # Obtaining the member '__getitem__' of a type (line 664)
    getitem___28009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 15), s_28008, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 664)
    subscript_call_result_28010 = invoke(stypy.reporting.localization.Localization(__file__, 664, 15), getitem___28009, end_28007)
    
    str_28011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 25), 'str', "'")
    # Applying the binary operator '==' (line 664)
    result_eq_28012 = python_operator(stypy.reporting.localization.Localization(__file__, 664, 15), '==', subscript_call_result_28010, str_28011)
    
    # Testing the type of an if condition (line 664)
    if_condition_28013 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 664, 12), result_eq_28012)
    # Assigning a type to the variable 'if_condition_28013' (line 664)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 12), 'if_condition_28013', if_condition_28013)
    # SSA begins for if statement (line 664)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 665):
    
    # Assigning a Call to a Name (line 665):
    
    # Assigning a Call to a Name (line 665):
    
    # Call to match(...): (line 665)
    # Processing the call arguments (line 665)
    # Getting the type of 's' (line 665)
    s_28016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 37), 's', False)
    # Getting the type of 'end' (line 665)
    end_28017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 40), 'end', False)
    # Processing the call keyword arguments (line 665)
    kwargs_28018 = {}
    # Getting the type of '_squote_re' (line 665)
    _squote_re_28014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 20), '_squote_re', False)
    # Obtaining the member 'match' of a type (line 665)
    match_28015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 665, 20), _squote_re_28014, 'match')
    # Calling match(args, kwargs) (line 665)
    match_call_result_28019 = invoke(stypy.reporting.localization.Localization(__file__, 665, 20), match_28015, *[s_28016, end_28017], **kwargs_28018)
    
    # Assigning a type to the variable 'm' (line 665)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 16), 'm', match_call_result_28019)
    # SSA branch for the else part of an if statement (line 664)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'end' (line 666)
    end_28020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 19), 'end')
    # Getting the type of 's' (line 666)
    s_28021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 17), 's')
    # Obtaining the member '__getitem__' of a type (line 666)
    getitem___28022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 17), s_28021, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 666)
    subscript_call_result_28023 = invoke(stypy.reporting.localization.Localization(__file__, 666, 17), getitem___28022, end_28020)
    
    str_28024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 27), 'str', '"')
    # Applying the binary operator '==' (line 666)
    result_eq_28025 = python_operator(stypy.reporting.localization.Localization(__file__, 666, 17), '==', subscript_call_result_28023, str_28024)
    
    # Testing the type of an if condition (line 666)
    if_condition_28026 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 666, 17), result_eq_28025)
    # Assigning a type to the variable 'if_condition_28026' (line 666)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 17), 'if_condition_28026', if_condition_28026)
    # SSA begins for if statement (line 666)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 667):
    
    # Assigning a Call to a Name (line 667):
    
    # Assigning a Call to a Name (line 667):
    
    # Call to match(...): (line 667)
    # Processing the call arguments (line 667)
    # Getting the type of 's' (line 667)
    s_28029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 37), 's', False)
    # Getting the type of 'end' (line 667)
    end_28030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 40), 'end', False)
    # Processing the call keyword arguments (line 667)
    kwargs_28031 = {}
    # Getting the type of '_dquote_re' (line 667)
    _dquote_re_28027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 20), '_dquote_re', False)
    # Obtaining the member 'match' of a type (line 667)
    match_28028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 20), _dquote_re_28027, 'match')
    # Calling match(args, kwargs) (line 667)
    match_call_result_28032 = invoke(stypy.reporting.localization.Localization(__file__, 667, 20), match_28028, *[s_28029, end_28030], **kwargs_28031)
    
    # Assigning a type to the variable 'm' (line 667)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 16), 'm', match_call_result_28032)
    # SSA branch for the else part of an if statement (line 666)
    module_type_store.open_ssa_branch('else')
    
    # Call to RuntimeError(...): (line 669)
    # Processing the call arguments (line 669)
    str_28034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 35), 'str', "this can't happen (bad char '%c')")
    
    # Obtaining the type of the subscript
    # Getting the type of 'end' (line 669)
    end_28035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 75), 'end', False)
    # Getting the type of 's' (line 669)
    s_28036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 73), 's', False)
    # Obtaining the member '__getitem__' of a type (line 669)
    getitem___28037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 73), s_28036, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 669)
    subscript_call_result_28038 = invoke(stypy.reporting.localization.Localization(__file__, 669, 73), getitem___28037, end_28035)
    
    # Applying the binary operator '%' (line 669)
    result_mod_28039 = python_operator(stypy.reporting.localization.Localization(__file__, 669, 35), '%', str_28034, subscript_call_result_28038)
    
    # Processing the call keyword arguments (line 669)
    kwargs_28040 = {}
    # Getting the type of 'RuntimeError' (line 669)
    RuntimeError_28033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 22), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 669)
    RuntimeError_call_result_28041 = invoke(stypy.reporting.localization.Localization(__file__, 669, 22), RuntimeError_28033, *[result_mod_28039], **kwargs_28040)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 669, 16), RuntimeError_call_result_28041, 'raise parameter', BaseException)
    # SSA join for if statement (line 666)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 664)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 671)
    # Getting the type of 'm' (line 671)
    m_28042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 15), 'm')
    # Getting the type of 'None' (line 671)
    None_28043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 20), 'None')
    
    (may_be_28044, more_types_in_union_28045) = may_be_none(m_28042, None_28043)

    if may_be_28044:

        if more_types_in_union_28045:
            # Runtime conditional SSA (line 671)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to ValueError(...): (line 672)
        # Processing the call arguments (line 672)
        str_28047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 33), 'str', 'bad string (mismatched %s quotes?)')
        
        # Obtaining the type of the subscript
        # Getting the type of 'end' (line 672)
        end_28048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 74), 'end', False)
        # Getting the type of 's' (line 672)
        s_28049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 72), 's', False)
        # Obtaining the member '__getitem__' of a type (line 672)
        getitem___28050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 72), s_28049, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 672)
        subscript_call_result_28051 = invoke(stypy.reporting.localization.Localization(__file__, 672, 72), getitem___28050, end_28048)
        
        # Applying the binary operator '%' (line 672)
        result_mod_28052 = python_operator(stypy.reporting.localization.Localization(__file__, 672, 33), '%', str_28047, subscript_call_result_28051)
        
        # Processing the call keyword arguments (line 672)
        kwargs_28053 = {}
        # Getting the type of 'ValueError' (line 672)
        ValueError_28046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 672)
        ValueError_call_result_28054 = invoke(stypy.reporting.localization.Localization(__file__, 672, 22), ValueError_28046, *[result_mod_28052], **kwargs_28053)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 672, 16), ValueError_call_result_28054, 'raise parameter', BaseException)

        if more_types_in_union_28045:
            # SSA join for if statement (line 671)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Tuple (line 674):
    
    # Assigning a Call to a Name:
    
    # Assigning a Call to a Name:
    
    # Call to span(...): (line 674)
    # Processing the call keyword arguments (line 674)
    kwargs_28057 = {}
    # Getting the type of 'm' (line 674)
    m_28055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 25), 'm', False)
    # Obtaining the member 'span' of a type (line 674)
    span_28056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 25), m_28055, 'span')
    # Calling span(args, kwargs) (line 674)
    span_call_result_28058 = invoke(stypy.reporting.localization.Localization(__file__, 674, 25), span_28056, *[], **kwargs_28057)
    
    # Assigning a type to the variable 'call_assignment_26363' (line 674)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 12), 'call_assignment_26363', span_call_result_28058)
    
    # Assigning a Call to a Name (line 674):
    
    # Assigning a Call to a Name (line 674):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_28061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 12), 'int')
    # Processing the call keyword arguments
    kwargs_28062 = {}
    # Getting the type of 'call_assignment_26363' (line 674)
    call_assignment_26363_28059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 12), 'call_assignment_26363', False)
    # Obtaining the member '__getitem__' of a type (line 674)
    getitem___28060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 12), call_assignment_26363_28059, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_28063 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___28060, *[int_28061], **kwargs_28062)
    
    # Assigning a type to the variable 'call_assignment_26364' (line 674)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 12), 'call_assignment_26364', getitem___call_result_28063)
    
    # Assigning a Name to a Name (line 674):
    
    # Assigning a Name to a Name (line 674):
    # Getting the type of 'call_assignment_26364' (line 674)
    call_assignment_26364_28064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 12), 'call_assignment_26364')
    # Assigning a type to the variable 'beg' (line 674)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 13), 'beg', call_assignment_26364_28064)
    
    # Assigning a Call to a Name (line 674):
    
    # Assigning a Call to a Name (line 674):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_28067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 12), 'int')
    # Processing the call keyword arguments
    kwargs_28068 = {}
    # Getting the type of 'call_assignment_26363' (line 674)
    call_assignment_26363_28065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 12), 'call_assignment_26363', False)
    # Obtaining the member '__getitem__' of a type (line 674)
    getitem___28066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 12), call_assignment_26363_28065, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_28069 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___28066, *[int_28067], **kwargs_28068)
    
    # Assigning a type to the variable 'call_assignment_26365' (line 674)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 12), 'call_assignment_26365', getitem___call_result_28069)
    
    # Assigning a Name to a Name (line 674):
    
    # Assigning a Name to a Name (line 674):
    # Getting the type of 'call_assignment_26365' (line 674)
    call_assignment_26365_28070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 12), 'call_assignment_26365')
    # Assigning a type to the variable 'end' (line 674)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 18), 'end', call_assignment_26365_28070)
    
    
    # Call to search(...): (line 675)
    # Processing the call arguments (line 675)
    
    # Obtaining the type of the subscript
    # Getting the type of 'beg' (line 675)
    beg_28073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 38), 'beg', False)
    int_28074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 42), 'int')
    # Applying the binary operator '+' (line 675)
    result_add_28075 = python_operator(stypy.reporting.localization.Localization(__file__, 675, 38), '+', beg_28073, int_28074)
    
    # Getting the type of 'end' (line 675)
    end_28076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 44), 'end', False)
    int_28077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 48), 'int')
    # Applying the binary operator '-' (line 675)
    result_sub_28078 = python_operator(stypy.reporting.localization.Localization(__file__, 675, 44), '-', end_28076, int_28077)
    
    slice_28079 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 675, 36), result_add_28075, result_sub_28078, None)
    # Getting the type of 's' (line 675)
    s_28080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 36), 's', False)
    # Obtaining the member '__getitem__' of a type (line 675)
    getitem___28081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 36), s_28080, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 675)
    subscript_call_result_28082 = invoke(stypy.reporting.localization.Localization(__file__, 675, 36), getitem___28081, slice_28079)
    
    # Processing the call keyword arguments (line 675)
    kwargs_28083 = {}
    # Getting the type of '_has_white_re' (line 675)
    _has_white_re_28071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 15), '_has_white_re', False)
    # Obtaining the member 'search' of a type (line 675)
    search_28072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 15), _has_white_re_28071, 'search')
    # Calling search(args, kwargs) (line 675)
    search_call_result_28084 = invoke(stypy.reporting.localization.Localization(__file__, 675, 15), search_28072, *[subscript_call_result_28082], **kwargs_28083)
    
    # Testing the type of an if condition (line 675)
    if_condition_28085 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 675, 12), search_call_result_28084)
    # Assigning a type to the variable 'if_condition_28085' (line 675)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 12), 'if_condition_28085', if_condition_28085)
    # SSA begins for if statement (line 675)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 676):
    
    # Assigning a BinOp to a Name (line 676):
    
    # Assigning a BinOp to a Name (line 676):
    
    # Obtaining the type of the subscript
    # Getting the type of 'beg' (line 676)
    beg_28086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 23), 'beg')
    slice_28087 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 676, 20), None, beg_28086, None)
    # Getting the type of 's' (line 676)
    s_28088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 20), 's')
    # Obtaining the member '__getitem__' of a type (line 676)
    getitem___28089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 20), s_28088, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 676)
    subscript_call_result_28090 = invoke(stypy.reporting.localization.Localization(__file__, 676, 20), getitem___28089, slice_28087)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'beg' (line 676)
    beg_28091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 32), 'beg')
    int_28092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 36), 'int')
    # Applying the binary operator '+' (line 676)
    result_add_28093 = python_operator(stypy.reporting.localization.Localization(__file__, 676, 32), '+', beg_28091, int_28092)
    
    # Getting the type of 'end' (line 676)
    end_28094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 38), 'end')
    int_28095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 42), 'int')
    # Applying the binary operator '-' (line 676)
    result_sub_28096 = python_operator(stypy.reporting.localization.Localization(__file__, 676, 38), '-', end_28094, int_28095)
    
    slice_28097 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 676, 30), result_add_28093, result_sub_28096, None)
    # Getting the type of 's' (line 676)
    s_28098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 30), 's')
    # Obtaining the member '__getitem__' of a type (line 676)
    getitem___28099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 30), s_28098, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 676)
    subscript_call_result_28100 = invoke(stypy.reporting.localization.Localization(__file__, 676, 30), getitem___28099, slice_28097)
    
    # Applying the binary operator '+' (line 676)
    result_add_28101 = python_operator(stypy.reporting.localization.Localization(__file__, 676, 20), '+', subscript_call_result_28090, subscript_call_result_28100)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'end' (line 676)
    end_28102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 49), 'end')
    slice_28103 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 676, 47), end_28102, None, None)
    # Getting the type of 's' (line 676)
    s_28104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 47), 's')
    # Obtaining the member '__getitem__' of a type (line 676)
    getitem___28105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 47), s_28104, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 676)
    subscript_call_result_28106 = invoke(stypy.reporting.localization.Localization(__file__, 676, 47), getitem___28105, slice_28103)
    
    # Applying the binary operator '+' (line 676)
    result_add_28107 = python_operator(stypy.reporting.localization.Localization(__file__, 676, 45), '+', result_add_28101, subscript_call_result_28106)
    
    # Assigning a type to the variable 's' (line 676)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 's', result_add_28107)
    
    # Assigning a BinOp to a Name (line 677):
    
    # Assigning a BinOp to a Name (line 677):
    
    # Assigning a BinOp to a Name (line 677):
    
    # Call to end(...): (line 677)
    # Processing the call keyword arguments (line 677)
    kwargs_28110 = {}
    # Getting the type of 'm' (line 677)
    m_28108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 22), 'm', False)
    # Obtaining the member 'end' of a type (line 677)
    end_28109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 22), m_28108, 'end')
    # Calling end(args, kwargs) (line 677)
    end_call_result_28111 = invoke(stypy.reporting.localization.Localization(__file__, 677, 22), end_28109, *[], **kwargs_28110)
    
    int_28112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 32), 'int')
    # Applying the binary operator '-' (line 677)
    result_sub_28113 = python_operator(stypy.reporting.localization.Localization(__file__, 677, 22), '-', end_call_result_28111, int_28112)
    
    # Assigning a type to the variable 'pos' (line 677)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 16), 'pos', result_sub_28113)
    # SSA branch for the else part of an if statement (line 675)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 681):
    
    # Assigning a Call to a Name (line 681):
    
    # Assigning a Call to a Name (line 681):
    
    # Call to end(...): (line 681)
    # Processing the call keyword arguments (line 681)
    kwargs_28116 = {}
    # Getting the type of 'm' (line 681)
    m_28114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 22), 'm', False)
    # Obtaining the member 'end' of a type (line 681)
    end_28115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 22), m_28114, 'end')
    # Calling end(args, kwargs) (line 681)
    end_call_result_28117 = invoke(stypy.reporting.localization.Localization(__file__, 681, 22), end_28115, *[], **kwargs_28116)
    
    # Assigning a type to the variable 'pos' (line 681)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 16), 'pos', end_call_result_28117)
    # SSA join for if statement (line 675)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 658)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 653)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'pos' (line 683)
    pos_28118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 11), 'pos')
    
    # Call to len(...): (line 683)
    # Processing the call arguments (line 683)
    # Getting the type of 's' (line 683)
    s_28120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 22), 's', False)
    # Processing the call keyword arguments (line 683)
    kwargs_28121 = {}
    # Getting the type of 'len' (line 683)
    len_28119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 18), 'len', False)
    # Calling len(args, kwargs) (line 683)
    len_call_result_28122 = invoke(stypy.reporting.localization.Localization(__file__, 683, 18), len_28119, *[s_28120], **kwargs_28121)
    
    # Applying the binary operator '>=' (line 683)
    result_ge_28123 = python_operator(stypy.reporting.localization.Localization(__file__, 683, 11), '>=', pos_28118, len_call_result_28122)
    
    # Testing the type of an if condition (line 683)
    if_condition_28124 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 683, 8), result_ge_28123)
    # Assigning a type to the variable 'if_condition_28124' (line 683)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 8), 'if_condition_28124', if_condition_28124)
    # SSA begins for if statement (line 683)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 684)
    # Processing the call arguments (line 684)
    # Getting the type of 's' (line 684)
    s_28127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 25), 's', False)
    # Processing the call keyword arguments (line 684)
    kwargs_28128 = {}
    # Getting the type of 'words' (line 684)
    words_28125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 12), 'words', False)
    # Obtaining the member 'append' of a type (line 684)
    append_28126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 684, 12), words_28125, 'append')
    # Calling append(args, kwargs) (line 684)
    append_call_result_28129 = invoke(stypy.reporting.localization.Localization(__file__, 684, 12), append_28126, *[s_28127], **kwargs_28128)
    
    # SSA join for if statement (line 683)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 646)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'words' (line 687)
    words_28130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 11), 'words')
    # Assigning a type to the variable 'stypy_return_type' (line 687)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 4), 'stypy_return_type', words_28130)
    
    # ################# End of 'split_quoted(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'split_quoted' in the type store
    # Getting the type of 'stypy_return_type' (line 641)
    stypy_return_type_28131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28131)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'split_quoted'
    return stypy_return_type_28131

# Assigning a type to the variable 'split_quoted' (line 641)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 0), 'split_quoted', split_quoted)

# Assigning a Name to a Attribute (line 688):

# Assigning a Name to a Attribute (line 688):

# Assigning a Name to a Attribute (line 688):
# Getting the type of 'split_quoted' (line 688)
split_quoted_28132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 25), 'split_quoted')
# Getting the type of 'ccompiler' (line 688)
ccompiler_28133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 0), 'ccompiler')
# Setting the type of the member 'split_quoted' of a type (line 688)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 0), ccompiler_28133, 'split_quoted', split_quoted_28132)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

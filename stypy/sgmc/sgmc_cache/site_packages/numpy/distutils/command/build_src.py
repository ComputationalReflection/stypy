
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Build swig and f2py sources.
2: '''
3: from __future__ import division, absolute_import, print_function
4: 
5: import os
6: import re
7: import sys
8: import shlex
9: import copy
10: 
11: from distutils.command import build_ext
12: from distutils.dep_util import newer_group, newer
13: from distutils.util import get_platform
14: from distutils.errors import DistutilsError, DistutilsSetupError
15: 
16: 
17: # this import can't be done here, as it uses numpy stuff only available
18: # after it's installed
19: #import numpy.f2py
20: from numpy.distutils import log
21: from numpy.distutils.misc_util import fortran_ext_match, \
22:      appendpath, is_string, is_sequence, get_cmd
23: from numpy.distutils.from_template import process_file as process_f_file
24: from numpy.distutils.conv_template import process_file as process_c_file
25: 
26: def subst_vars(target, source, d):
27:     '''Substitute any occurence of @foo@ by d['foo'] from source file into
28:     target.'''
29:     var = re.compile('@([a-zA-Z_]+)@')
30:     fs = open(source, 'r')
31:     try:
32:         ft = open(target, 'w')
33:         try:
34:             for l in fs:
35:                 m = var.search(l)
36:                 if m:
37:                     ft.write(l.replace('@%s@' % m.group(1), d[m.group(1)]))
38:                 else:
39:                     ft.write(l)
40:         finally:
41:             ft.close()
42:     finally:
43:         fs.close()
44: 
45: class build_src(build_ext.build_ext):
46: 
47:     description = "build sources from SWIG, F2PY files or a function"
48: 
49:     user_options = [
50:         ('build-src=', 'd', "directory to \"build\" sources to"),
51:         ('f2py-opts=', None, "list of f2py command line options"),
52:         ('swig=', None, "path to the SWIG executable"),
53:         ('swig-opts=', None, "list of SWIG command line options"),
54:         ('swig-cpp', None, "make SWIG create C++ files (default is autodetected from sources)"),
55:         ('f2pyflags=', None, "additional flags to f2py (use --f2py-opts= instead)"), # obsolete
56:         ('swigflags=', None, "additional flags to swig (use --swig-opts= instead)"), # obsolete
57:         ('force', 'f', "forcibly build everything (ignore file timestamps)"),
58:         ('inplace', 'i',
59:          "ignore build-lib and put compiled extensions into the source " +
60:          "directory alongside your pure Python modules"),
61:         ]
62: 
63:     boolean_options = ['force', 'inplace']
64: 
65:     help_options = []
66: 
67:     def initialize_options(self):
68:         self.extensions = None
69:         self.package = None
70:         self.py_modules = None
71:         self.py_modules_dict = None
72:         self.build_src = None
73:         self.build_lib = None
74:         self.build_base = None
75:         self.force = None
76:         self.inplace = None
77:         self.package_dir = None
78:         self.f2pyflags = None # obsolete
79:         self.f2py_opts = None
80:         self.swigflags = None # obsolete
81:         self.swig_opts = None
82:         self.swig_cpp = None
83:         self.swig = None
84: 
85:     def finalize_options(self):
86:         self.set_undefined_options('build',
87:                                    ('build_base', 'build_base'),
88:                                    ('build_lib', 'build_lib'),
89:                                    ('force', 'force'))
90:         if self.package is None:
91:             self.package = self.distribution.ext_package
92:         self.extensions = self.distribution.ext_modules
93:         self.libraries = self.distribution.libraries or []
94:         self.py_modules = self.distribution.py_modules or []
95:         self.data_files = self.distribution.data_files or []
96: 
97:         if self.build_src is None:
98:             plat_specifier = ".%s-%s" % (get_platform(), sys.version[0:3])
99:             self.build_src = os.path.join(self.build_base, 'src'+plat_specifier)
100: 
101:         # py_modules_dict is used in build_py.find_package_modules
102:         self.py_modules_dict = {}
103: 
104:         if self.f2pyflags:
105:             if self.f2py_opts:
106:                 log.warn('ignoring --f2pyflags as --f2py-opts already used')
107:             else:
108:                 self.f2py_opts = self.f2pyflags
109:             self.f2pyflags = None
110:         if self.f2py_opts is None:
111:             self.f2py_opts = []
112:         else:
113:             self.f2py_opts = shlex.split(self.f2py_opts)
114: 
115:         if self.swigflags:
116:             if self.swig_opts:
117:                 log.warn('ignoring --swigflags as --swig-opts already used')
118:             else:
119:                 self.swig_opts = self.swigflags
120:             self.swigflags = None
121: 
122:         if self.swig_opts is None:
123:             self.swig_opts = []
124:         else:
125:             self.swig_opts = shlex.split(self.swig_opts)
126: 
127:         # use options from build_ext command
128:         build_ext = self.get_finalized_command('build_ext')
129:         if self.inplace is None:
130:             self.inplace = build_ext.inplace
131:         if self.swig_cpp is None:
132:             self.swig_cpp = build_ext.swig_cpp
133:         for c in ['swig', 'swig_opt']:
134:             o = '--'+c.replace('_', '-')
135:             v = getattr(build_ext, c, None)
136:             if v:
137:                 if getattr(self, c):
138:                     log.warn('both build_src and build_ext define %s option' % (o))
139:                 else:
140:                     log.info('using "%s=%s" option from build_ext command' % (o, v))
141:                     setattr(self, c, v)
142: 
143:     def run(self):
144:         log.info("build_src")
145:         if not (self.extensions or self.libraries):
146:             return
147:         self.build_sources()
148: 
149:     def build_sources(self):
150: 
151:         if self.inplace:
152:             self.get_package_dir = \
153:                      self.get_finalized_command('build_py').get_package_dir
154: 
155:         self.build_py_modules_sources()
156: 
157:         for libname_info in self.libraries:
158:             self.build_library_sources(*libname_info)
159: 
160:         if self.extensions:
161:             self.check_extensions_list(self.extensions)
162: 
163:             for ext in self.extensions:
164:                 self.build_extension_sources(ext)
165: 
166:         self.build_data_files_sources()
167:         self.build_npy_pkg_config()
168: 
169:     def build_data_files_sources(self):
170:         if not self.data_files:
171:             return
172:         log.info('building data_files sources')
173:         from numpy.distutils.misc_util import get_data_files
174:         new_data_files = []
175:         for data in self.data_files:
176:             if isinstance(data, str):
177:                 new_data_files.append(data)
178:             elif isinstance(data, tuple):
179:                 d, files = data
180:                 if self.inplace:
181:                     build_dir = self.get_package_dir('.'.join(d.split(os.sep)))
182:                 else:
183:                     build_dir = os.path.join(self.build_src, d)
184:                 funcs = [f for f in files if hasattr(f, '__call__')]
185:                 files = [f for f in files if not hasattr(f, '__call__')]
186:                 for f in funcs:
187:                     if f.__code__.co_argcount==1:
188:                         s = f(build_dir)
189:                     else:
190:                         s = f()
191:                     if s is not None:
192:                         if isinstance(s, list):
193:                             files.extend(s)
194:                         elif isinstance(s, str):
195:                             files.append(s)
196:                         else:
197:                             raise TypeError(repr(s))
198:                 filenames = get_data_files((d, files))
199:                 new_data_files.append((d, filenames))
200:             else:
201:                 raise TypeError(repr(data))
202:         self.data_files[:] = new_data_files
203: 
204: 
205:     def _build_npy_pkg_config(self, info, gd):
206:         import shutil
207:         template, install_dir, subst_dict = info
208:         template_dir = os.path.dirname(template)
209:         for k, v in gd.items():
210:             subst_dict[k] = v
211: 
212:         if self.inplace == 1:
213:             generated_dir = os.path.join(template_dir, install_dir)
214:         else:
215:             generated_dir = os.path.join(self.build_src, template_dir,
216:                     install_dir)
217:         generated = os.path.basename(os.path.splitext(template)[0])
218:         generated_path = os.path.join(generated_dir, generated)
219:         if not os.path.exists(generated_dir):
220:             os.makedirs(generated_dir)
221: 
222:         subst_vars(generated_path, template, subst_dict)
223: 
224:         # Where to install relatively to install prefix
225:         full_install_dir = os.path.join(template_dir, install_dir)
226:         return full_install_dir, generated_path
227: 
228:     def build_npy_pkg_config(self):
229:         log.info('build_src: building npy-pkg config files')
230: 
231:         # XXX: another ugly workaround to circumvent distutils brain damage. We
232:         # need the install prefix here, but finalizing the options of the
233:         # install command when only building sources cause error. Instead, we
234:         # copy the install command instance, and finalize the copy so that it
235:         # does not disrupt how distutils want to do things when with the
236:         # original install command instance.
237:         install_cmd = copy.copy(get_cmd('install'))
238:         if not install_cmd.finalized == 1:
239:             install_cmd.finalize_options()
240:         build_npkg = False
241:         gd = {}
242:         if self.inplace == 1:
243:             top_prefix = '.'
244:             build_npkg = True
245:         elif hasattr(install_cmd, 'install_libbase'):
246:             top_prefix = install_cmd.install_libbase
247:             build_npkg = True
248: 
249:         if build_npkg:
250:             for pkg, infos in self.distribution.installed_pkg_config.items():
251:                 pkg_path = self.distribution.package_dir[pkg]
252:                 prefix = os.path.join(os.path.abspath(top_prefix), pkg_path)
253:                 d = {'prefix': prefix}
254:                 for info in infos:
255:                     install_dir, generated = self._build_npy_pkg_config(info, d)
256:                     self.distribution.data_files.append((install_dir,
257:                         [generated]))
258: 
259:     def build_py_modules_sources(self):
260:         if not self.py_modules:
261:             return
262:         log.info('building py_modules sources')
263:         new_py_modules = []
264:         for source in self.py_modules:
265:             if is_sequence(source) and len(source)==3:
266:                 package, module_base, source = source
267:                 if self.inplace:
268:                     build_dir = self.get_package_dir(package)
269:                 else:
270:                     build_dir = os.path.join(self.build_src,
271:                                              os.path.join(*package.split('.')))
272:                 if hasattr(source, '__call__'):
273:                     target = os.path.join(build_dir, module_base + '.py')
274:                     source = source(target)
275:                 if source is None:
276:                     continue
277:                 modules = [(package, module_base, source)]
278:                 if package not in self.py_modules_dict:
279:                     self.py_modules_dict[package] = []
280:                 self.py_modules_dict[package] += modules
281:             else:
282:                 new_py_modules.append(source)
283:         self.py_modules[:] = new_py_modules
284: 
285:     def build_library_sources(self, lib_name, build_info):
286:         sources = list(build_info.get('sources', []))
287: 
288:         if not sources:
289:             return
290: 
291:         log.info('building library "%s" sources' % (lib_name))
292: 
293:         sources = self.generate_sources(sources, (lib_name, build_info))
294: 
295:         sources = self.template_sources(sources, (lib_name, build_info))
296: 
297:         sources, h_files = self.filter_h_files(sources)
298: 
299:         if h_files:
300:             log.info('%s - nothing done with h_files = %s',
301:                      self.package, h_files)
302: 
303:         #for f in h_files:
304:         #    self.distribution.headers.append((lib_name,f))
305: 
306:         build_info['sources'] = sources
307:         return
308: 
309:     def build_extension_sources(self, ext):
310: 
311:         sources = list(ext.sources)
312: 
313:         log.info('building extension "%s" sources' % (ext.name))
314: 
315:         fullname = self.get_ext_fullname(ext.name)
316: 
317:         modpath = fullname.split('.')
318:         package = '.'.join(modpath[0:-1])
319: 
320:         if self.inplace:
321:             self.ext_target_dir = self.get_package_dir(package)
322: 
323:         sources = self.generate_sources(sources, ext)
324:         sources = self.template_sources(sources, ext)
325:         sources = self.swig_sources(sources, ext)
326:         sources = self.f2py_sources(sources, ext)
327:         sources = self.pyrex_sources(sources, ext)
328: 
329:         sources, py_files = self.filter_py_files(sources)
330: 
331:         if package not in self.py_modules_dict:
332:             self.py_modules_dict[package] = []
333:         modules = []
334:         for f in py_files:
335:             module = os.path.splitext(os.path.basename(f))[0]
336:             modules.append((package, module, f))
337:         self.py_modules_dict[package] += modules
338: 
339:         sources, h_files = self.filter_h_files(sources)
340: 
341:         if h_files:
342:             log.info('%s - nothing done with h_files = %s',
343:                      package, h_files)
344:         #for f in h_files:
345:         #    self.distribution.headers.append((package,f))
346: 
347:         ext.sources = sources
348: 
349:     def generate_sources(self, sources, extension):
350:         new_sources = []
351:         func_sources = []
352:         for source in sources:
353:             if is_string(source):
354:                 new_sources.append(source)
355:             else:
356:                 func_sources.append(source)
357:         if not func_sources:
358:             return new_sources
359:         if self.inplace and not is_sequence(extension):
360:             build_dir = self.ext_target_dir
361:         else:
362:             if is_sequence(extension):
363:                 name = extension[0]
364:             #    if 'include_dirs' not in extension[1]:
365:             #        extension[1]['include_dirs'] = []
366:             #    incl_dirs = extension[1]['include_dirs']
367:             else:
368:                 name = extension.name
369:             #    incl_dirs = extension.include_dirs
370:             #if self.build_src not in incl_dirs:
371:             #    incl_dirs.append(self.build_src)
372:             build_dir = os.path.join(*([self.build_src]\
373:                                        +name.split('.')[:-1]))
374:         self.mkpath(build_dir)
375:         for func in func_sources:
376:             source = func(extension, build_dir)
377:             if not source:
378:                 continue
379:             if is_sequence(source):
380:                 [log.info("  adding '%s' to sources." % (s,)) for s in source]
381:                 new_sources.extend(source)
382:             else:
383:                 log.info("  adding '%s' to sources." % (source,))
384:                 new_sources.append(source)
385: 
386:         return new_sources
387: 
388:     def filter_py_files(self, sources):
389:         return self.filter_files(sources, ['.py'])
390: 
391:     def filter_h_files(self, sources):
392:         return self.filter_files(sources, ['.h', '.hpp', '.inc'])
393: 
394:     def filter_files(self, sources, exts = []):
395:         new_sources = []
396:         files = []
397:         for source in sources:
398:             (base, ext) = os.path.splitext(source)
399:             if ext in exts:
400:                 files.append(source)
401:             else:
402:                 new_sources.append(source)
403:         return new_sources, files
404: 
405:     def template_sources(self, sources, extension):
406:         new_sources = []
407:         if is_sequence(extension):
408:             depends = extension[1].get('depends')
409:             include_dirs = extension[1].get('include_dirs')
410:         else:
411:             depends = extension.depends
412:             include_dirs = extension.include_dirs
413:         for source in sources:
414:             (base, ext) = os.path.splitext(source)
415:             if ext == '.src':  # Template file
416:                 if self.inplace:
417:                     target_dir = os.path.dirname(base)
418:                 else:
419:                     target_dir = appendpath(self.build_src, os.path.dirname(base))
420:                 self.mkpath(target_dir)
421:                 target_file = os.path.join(target_dir, os.path.basename(base))
422:                 if (self.force or newer_group([source] + depends, target_file)):
423:                     if _f_pyf_ext_match(base):
424:                         log.info("from_template:> %s" % (target_file))
425:                         outstr = process_f_file(source)
426:                     else:
427:                         log.info("conv_template:> %s" % (target_file))
428:                         outstr = process_c_file(source)
429:                     fid = open(target_file, 'w')
430:                     fid.write(outstr)
431:                     fid.close()
432:                 if _header_ext_match(target_file):
433:                     d = os.path.dirname(target_file)
434:                     if d not in include_dirs:
435:                         log.info("  adding '%s' to include_dirs." % (d))
436:                         include_dirs.append(d)
437:                 new_sources.append(target_file)
438:             else:
439:                 new_sources.append(source)
440:         return new_sources
441: 
442:     def pyrex_sources(self, sources, extension):
443:         '''Pyrex not supported; this remains for Cython support (see below)'''
444:         new_sources = []
445:         ext_name = extension.name.split('.')[-1]
446:         for source in sources:
447:             (base, ext) = os.path.splitext(source)
448:             if ext == '.pyx':
449:                 target_file = self.generate_a_pyrex_source(base, ext_name,
450:                                                            source,
451:                                                            extension)
452:                 new_sources.append(target_file)
453:             else:
454:                 new_sources.append(source)
455:         return new_sources
456: 
457:     def generate_a_pyrex_source(self, base, ext_name, source, extension):
458:         '''Pyrex is not supported, but some projects monkeypatch this method.
459: 
460:         That allows compiling Cython code, see gh-6955.
461:         This method will remain here for compatibility reasons.
462:         '''
463:         return []
464: 
465:     def f2py_sources(self, sources, extension):
466:         new_sources = []
467:         f2py_sources = []
468:         f_sources = []
469:         f2py_targets = {}
470:         target_dirs = []
471:         ext_name = extension.name.split('.')[-1]
472:         skip_f2py = 0
473: 
474:         for source in sources:
475:             (base, ext) = os.path.splitext(source)
476:             if ext == '.pyf': # F2PY interface file
477:                 if self.inplace:
478:                     target_dir = os.path.dirname(base)
479:                 else:
480:                     target_dir = appendpath(self.build_src, os.path.dirname(base))
481:                 if os.path.isfile(source):
482:                     name = get_f2py_modulename(source)
483:                     if name != ext_name:
484:                         raise DistutilsSetupError('mismatch of extension names: %s '
485:                                                   'provides %r but expected %r' % (
486:                             source, name, ext_name))
487:                     target_file = os.path.join(target_dir, name+'module.c')
488:                 else:
489:                     log.debug('  source %s does not exist: skipping f2py\'ing.' \
490:                               % (source))
491:                     name = ext_name
492:                     skip_f2py = 1
493:                     target_file = os.path.join(target_dir, name+'module.c')
494:                     if not os.path.isfile(target_file):
495:                         log.warn('  target %s does not exist:\n   '\
496:                                  'Assuming %smodule.c was generated with '\
497:                                  '"build_src --inplace" command.' \
498:                                  % (target_file, name))
499:                         target_dir = os.path.dirname(base)
500:                         target_file = os.path.join(target_dir, name+'module.c')
501:                         if not os.path.isfile(target_file):
502:                             raise DistutilsSetupError("%r missing" % (target_file,))
503:                         log.info('   Yes! Using %r as up-to-date target.' \
504:                                  % (target_file))
505:                 target_dirs.append(target_dir)
506:                 f2py_sources.append(source)
507:                 f2py_targets[source] = target_file
508:                 new_sources.append(target_file)
509:             elif fortran_ext_match(ext):
510:                 f_sources.append(source)
511:             else:
512:                 new_sources.append(source)
513: 
514:         if not (f2py_sources or f_sources):
515:             return new_sources
516: 
517:         for d in target_dirs:
518:             self.mkpath(d)
519: 
520:         f2py_options = extension.f2py_options + self.f2py_opts
521: 
522:         if self.distribution.libraries:
523:             for name, build_info in self.distribution.libraries:
524:                 if name in extension.libraries:
525:                     f2py_options.extend(build_info.get('f2py_options', []))
526: 
527:         log.info("f2py options: %s" % (f2py_options))
528: 
529:         if f2py_sources:
530:             if len(f2py_sources) != 1:
531:                 raise DistutilsSetupError(
532:                     'only one .pyf file is allowed per extension module but got'\
533:                     ' more: %r' % (f2py_sources,))
534:             source = f2py_sources[0]
535:             target_file = f2py_targets[source]
536:             target_dir = os.path.dirname(target_file) or '.'
537:             depends = [source] + extension.depends
538:             if (self.force or newer_group(depends, target_file, 'newer')) \
539:                    and not skip_f2py:
540:                 log.info("f2py: %s" % (source))
541:                 import numpy.f2py
542:                 numpy.f2py.run_main(f2py_options
543:                                     + ['--build-dir', target_dir, source])
544:             else:
545:                 log.debug("  skipping '%s' f2py interface (up-to-date)" % (source))
546:         else:
547:             #XXX TODO: --inplace support for sdist command
548:             if is_sequence(extension):
549:                 name = extension[0]
550:             else: name = extension.name
551:             target_dir = os.path.join(*([self.build_src]\
552:                                         +name.split('.')[:-1]))
553:             target_file = os.path.join(target_dir, ext_name + 'module.c')
554:             new_sources.append(target_file)
555:             depends = f_sources + extension.depends
556:             if (self.force or newer_group(depends, target_file, 'newer')) \
557:                    and not skip_f2py:
558:                 log.info("f2py:> %s" % (target_file))
559:                 self.mkpath(target_dir)
560:                 import numpy.f2py
561:                 numpy.f2py.run_main(f2py_options + ['--lower',
562:                                                 '--build-dir', target_dir]+\
563:                                 ['-m', ext_name]+f_sources)
564:             else:
565:                 log.debug("  skipping f2py fortran files for '%s' (up-to-date)"\
566:                           % (target_file))
567: 
568:         if not os.path.isfile(target_file):
569:             raise DistutilsError("f2py target file %r not generated" % (target_file,))
570: 
571:         target_c = os.path.join(self.build_src, 'fortranobject.c')
572:         target_h = os.path.join(self.build_src, 'fortranobject.h')
573:         log.info("  adding '%s' to sources." % (target_c))
574:         new_sources.append(target_c)
575:         if self.build_src not in extension.include_dirs:
576:             log.info("  adding '%s' to include_dirs." \
577:                      % (self.build_src))
578:             extension.include_dirs.append(self.build_src)
579: 
580:         if not skip_f2py:
581:             import numpy.f2py
582:             d = os.path.dirname(numpy.f2py.__file__)
583:             source_c = os.path.join(d, 'src', 'fortranobject.c')
584:             source_h = os.path.join(d, 'src', 'fortranobject.h')
585:             if newer(source_c, target_c) or newer(source_h, target_h):
586:                 self.mkpath(os.path.dirname(target_c))
587:                 self.copy_file(source_c, target_c)
588:                 self.copy_file(source_h, target_h)
589:         else:
590:             if not os.path.isfile(target_c):
591:                 raise DistutilsSetupError("f2py target_c file %r not found" % (target_c,))
592:             if not os.path.isfile(target_h):
593:                 raise DistutilsSetupError("f2py target_h file %r not found" % (target_h,))
594: 
595:         for name_ext in ['-f2pywrappers.f', '-f2pywrappers2.f90']:
596:             filename = os.path.join(target_dir, ext_name + name_ext)
597:             if os.path.isfile(filename):
598:                 log.info("  adding '%s' to sources." % (filename))
599:                 f_sources.append(filename)
600: 
601:         return new_sources + f_sources
602: 
603:     def swig_sources(self, sources, extension):
604:         # Assuming SWIG 1.3.14 or later. See compatibility note in
605:         #   http://www.swig.org/Doc1.3/Python.html#Python_nn6
606: 
607:         new_sources = []
608:         swig_sources = []
609:         swig_targets = {}
610:         target_dirs = []
611:         py_files = []     # swig generated .py files
612:         target_ext = '.c'
613:         if '-c++' in extension.swig_opts:
614:             typ = 'c++'
615:             is_cpp = True
616:             extension.swig_opts.remove('-c++')
617:         elif self.swig_cpp:
618:             typ = 'c++'
619:             is_cpp = True
620:         else:
621:             typ = None
622:             is_cpp = False
623:         skip_swig = 0
624:         ext_name = extension.name.split('.')[-1]
625: 
626:         for source in sources:
627:             (base, ext) = os.path.splitext(source)
628:             if ext == '.i': # SWIG interface file
629:                 # the code below assumes that the sources list
630:                 # contains not more than one .i SWIG interface file
631:                 if self.inplace:
632:                     target_dir = os.path.dirname(base)
633:                     py_target_dir = self.ext_target_dir
634:                 else:
635:                     target_dir = appendpath(self.build_src, os.path.dirname(base))
636:                     py_target_dir = target_dir
637:                 if os.path.isfile(source):
638:                     name = get_swig_modulename(source)
639:                     if name != ext_name[1:]:
640:                         raise DistutilsSetupError(
641:                             'mismatch of extension names: %s provides %r'
642:                             ' but expected %r' % (source, name, ext_name[1:]))
643:                     if typ is None:
644:                         typ = get_swig_target(source)
645:                         is_cpp = typ=='c++'
646:                     else:
647:                         typ2 = get_swig_target(source)
648:                         if typ2 is None:
649:                             log.warn('source %r does not define swig target, assuming %s swig target' \
650:                                      % (source, typ))
651:                         elif typ!=typ2:
652:                             log.warn('expected %r but source %r defines %r swig target' \
653:                                      % (typ, source, typ2))
654:                             if typ2=='c++':
655:                                 log.warn('resetting swig target to c++ (some targets may have .c extension)')
656:                                 is_cpp = True
657:                             else:
658:                                 log.warn('assuming that %r has c++ swig target' % (source))
659:                     if is_cpp:
660:                         target_ext = '.cpp'
661:                     target_file = os.path.join(target_dir, '%s_wrap%s' \
662:                                                % (name, target_ext))
663:                 else:
664:                     log.warn('  source %s does not exist: skipping swig\'ing.' \
665:                              % (source))
666:                     name = ext_name[1:]
667:                     skip_swig = 1
668:                     target_file = _find_swig_target(target_dir, name)
669:                     if not os.path.isfile(target_file):
670:                         log.warn('  target %s does not exist:\n   '\
671:                                  'Assuming %s_wrap.{c,cpp} was generated with '\
672:                                  '"build_src --inplace" command.' \
673:                                  % (target_file, name))
674:                         target_dir = os.path.dirname(base)
675:                         target_file = _find_swig_target(target_dir, name)
676:                         if not os.path.isfile(target_file):
677:                             raise DistutilsSetupError("%r missing" % (target_file,))
678:                         log.warn('   Yes! Using %r as up-to-date target.' \
679:                                  % (target_file))
680:                 target_dirs.append(target_dir)
681:                 new_sources.append(target_file)
682:                 py_files.append(os.path.join(py_target_dir, name+'.py'))
683:                 swig_sources.append(source)
684:                 swig_targets[source] = new_sources[-1]
685:             else:
686:                 new_sources.append(source)
687: 
688:         if not swig_sources:
689:             return new_sources
690: 
691:         if skip_swig:
692:             return new_sources + py_files
693: 
694:         for d in target_dirs:
695:             self.mkpath(d)
696: 
697:         swig = self.swig or self.find_swig()
698:         swig_cmd = [swig, "-python"] + extension.swig_opts
699:         if is_cpp:
700:             swig_cmd.append('-c++')
701:         for d in extension.include_dirs:
702:             swig_cmd.append('-I'+d)
703:         for source in swig_sources:
704:             target = swig_targets[source]
705:             depends = [source] + extension.depends
706:             if self.force or newer_group(depends, target, 'newer'):
707:                 log.info("%s: %s" % (os.path.basename(swig) \
708:                                      + (is_cpp and '++' or ''), source))
709:                 self.spawn(swig_cmd + self.swig_opts \
710:                            + ["-o", target, '-outdir', py_target_dir, source])
711:             else:
712:                 log.debug("  skipping '%s' swig interface (up-to-date)" \
713:                          % (source))
714: 
715:         return new_sources + py_files
716: 
717: _f_pyf_ext_match = re.compile(r'.*[.](f90|f95|f77|for|ftn|f|pyf)\Z', re.I).match
718: _header_ext_match = re.compile(r'.*[.](inc|h|hpp)\Z', re.I).match
719: 
720: #### SWIG related auxiliary functions ####
721: _swig_module_name_match = re.compile(r'\s*%module\s*(.*\(\s*package\s*=\s*"(?P<package>[\w_]+)".*\)|)\s*(?P<name>[\w_]+)',
722:                                      re.I).match
723: _has_c_header = re.compile(r'-[*]-\s*c\s*-[*]-', re.I).search
724: _has_cpp_header = re.compile(r'-[*]-\s*c[+][+]\s*-[*]-', re.I).search
725: 
726: def get_swig_target(source):
727:     f = open(source, 'r')
728:     result = None
729:     line = f.readline()
730:     if _has_cpp_header(line):
731:         result = 'c++'
732:     if _has_c_header(line):
733:         result = 'c'
734:     f.close()
735:     return result
736: 
737: def get_swig_modulename(source):
738:     f = open(source, 'r')
739:     name = None
740:     for line in f:
741:         m = _swig_module_name_match(line)
742:         if m:
743:             name = m.group('name')
744:             break
745:     f.close()
746:     return name
747: 
748: def _find_swig_target(target_dir, name):
749:     for ext in ['.cpp', '.c']:
750:         target = os.path.join(target_dir, '%s_wrap%s' % (name, ext))
751:         if os.path.isfile(target):
752:             break
753:     return target
754: 
755: #### F2PY related auxiliary functions ####
756: 
757: _f2py_module_name_match = re.compile(r'\s*python\s*module\s*(?P<name>[\w_]+)',
758:                                 re.I).match
759: _f2py_user_module_name_match = re.compile(r'\s*python\s*module\s*(?P<name>[\w_]*?'\
760:                                      '__user__[\w_]*)', re.I).match
761: 
762: def get_f2py_modulename(source):
763:     name = None
764:     f = open(source)
765:     for line in f:
766:         m = _f2py_module_name_match(line)
767:         if m:
768:             if _f2py_user_module_name_match(line): # skip *__user__* names
769:                 continue
770:             name = m.group('name')
771:             break
772:     f.close()
773:     return name
774: 
775: ##########################################
776: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_55320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', ' Build swig and f2py sources.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import os' statement (line 5)
import os

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import re' statement (line 6)
import re

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import sys' statement (line 7)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import shlex' statement (line 8)
import shlex

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'shlex', shlex, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import copy' statement (line 9)
import copy

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'copy', copy, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from distutils.command import build_ext' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_55321 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.command')

if (type(import_55321) is not StypyTypeError):

    if (import_55321 != 'pyd_module'):
        __import__(import_55321)
        sys_modules_55322 = sys.modules[import_55321]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.command', sys_modules_55322.module_type_store, module_type_store, ['build_ext'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_55322, sys_modules_55322.module_type_store, module_type_store)
    else:
        from distutils.command import build_ext

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.command', None, module_type_store, ['build_ext'], [build_ext])

else:
    # Assigning a type to the variable 'distutils.command' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.command', import_55321)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils.dep_util import newer_group, newer' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_55323 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.dep_util')

if (type(import_55323) is not StypyTypeError):

    if (import_55323 != 'pyd_module'):
        __import__(import_55323)
        sys_modules_55324 = sys.modules[import_55323]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.dep_util', sys_modules_55324.module_type_store, module_type_store, ['newer_group', 'newer'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_55324, sys_modules_55324.module_type_store, module_type_store)
    else:
        from distutils.dep_util import newer_group, newer

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.dep_util', None, module_type_store, ['newer_group', 'newer'], [newer_group, newer])

else:
    # Assigning a type to the variable 'distutils.dep_util' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.dep_util', import_55323)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from distutils.util import get_platform' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_55325 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.util')

if (type(import_55325) is not StypyTypeError):

    if (import_55325 != 'pyd_module'):
        __import__(import_55325)
        sys_modules_55326 = sys.modules[import_55325]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.util', sys_modules_55326.module_type_store, module_type_store, ['get_platform'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_55326, sys_modules_55326.module_type_store, module_type_store)
    else:
        from distutils.util import get_platform

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.util', None, module_type_store, ['get_platform'], [get_platform])

else:
    # Assigning a type to the variable 'distutils.util' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.util', import_55325)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from distutils.errors import DistutilsError, DistutilsSetupError' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_55327 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.errors')

if (type(import_55327) is not StypyTypeError):

    if (import_55327 != 'pyd_module'):
        __import__(import_55327)
        sys_modules_55328 = sys.modules[import_55327]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.errors', sys_modules_55328.module_type_store, module_type_store, ['DistutilsError', 'DistutilsSetupError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_55328, sys_modules_55328.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsError, DistutilsSetupError

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.errors', None, module_type_store, ['DistutilsError', 'DistutilsSetupError'], [DistutilsError, DistutilsSetupError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.errors', import_55327)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from numpy.distutils import log' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_55329 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.distutils')

if (type(import_55329) is not StypyTypeError):

    if (import_55329 != 'pyd_module'):
        __import__(import_55329)
        sys_modules_55330 = sys.modules[import_55329]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.distutils', sys_modules_55330.module_type_store, module_type_store, ['log'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_55330, sys_modules_55330.module_type_store, module_type_store)
    else:
        from numpy.distutils import log

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.distutils', None, module_type_store, ['log'], [log])

else:
    # Assigning a type to the variable 'numpy.distutils' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.distutils', import_55329)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from numpy.distutils.misc_util import fortran_ext_match, appendpath, is_string, is_sequence, get_cmd' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_55331 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.distutils.misc_util')

if (type(import_55331) is not StypyTypeError):

    if (import_55331 != 'pyd_module'):
        __import__(import_55331)
        sys_modules_55332 = sys.modules[import_55331]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.distutils.misc_util', sys_modules_55332.module_type_store, module_type_store, ['fortran_ext_match', 'appendpath', 'is_string', 'is_sequence', 'get_cmd'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_55332, sys_modules_55332.module_type_store, module_type_store)
    else:
        from numpy.distutils.misc_util import fortran_ext_match, appendpath, is_string, is_sequence, get_cmd

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.distutils.misc_util', None, module_type_store, ['fortran_ext_match', 'appendpath', 'is_string', 'is_sequence', 'get_cmd'], [fortran_ext_match, appendpath, is_string, is_sequence, get_cmd])

else:
    # Assigning a type to the variable 'numpy.distutils.misc_util' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.distutils.misc_util', import_55331)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from numpy.distutils.from_template import process_f_file' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_55333 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.distutils.from_template')

if (type(import_55333) is not StypyTypeError):

    if (import_55333 != 'pyd_module'):
        __import__(import_55333)
        sys_modules_55334 = sys.modules[import_55333]
        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.distutils.from_template', sys_modules_55334.module_type_store, module_type_store, ['process_file'])
        nest_module(stypy.reporting.localization.Localization(__file__, 23, 0), __file__, sys_modules_55334, sys_modules_55334.module_type_store, module_type_store)
    else:
        from numpy.distutils.from_template import process_file as process_f_file

        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.distutils.from_template', None, module_type_store, ['process_file'], [process_f_file])

else:
    # Assigning a type to the variable 'numpy.distutils.from_template' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.distutils.from_template', import_55333)

# Adding an alias
module_type_store.add_alias('process_f_file', 'process_file')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from numpy.distutils.conv_template import process_c_file' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_55335 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.distutils.conv_template')

if (type(import_55335) is not StypyTypeError):

    if (import_55335 != 'pyd_module'):
        __import__(import_55335)
        sys_modules_55336 = sys.modules[import_55335]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.distutils.conv_template', sys_modules_55336.module_type_store, module_type_store, ['process_file'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_55336, sys_modules_55336.module_type_store, module_type_store)
    else:
        from numpy.distutils.conv_template import process_file as process_c_file

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.distutils.conv_template', None, module_type_store, ['process_file'], [process_c_file])

else:
    # Assigning a type to the variable 'numpy.distutils.conv_template' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.distutils.conv_template', import_55335)

# Adding an alias
module_type_store.add_alias('process_c_file', 'process_file')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')


@norecursion
def subst_vars(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'subst_vars'
    module_type_store = module_type_store.open_function_context('subst_vars', 26, 0, False)
    
    # Passed parameters checking function
    subst_vars.stypy_localization = localization
    subst_vars.stypy_type_of_self = None
    subst_vars.stypy_type_store = module_type_store
    subst_vars.stypy_function_name = 'subst_vars'
    subst_vars.stypy_param_names_list = ['target', 'source', 'd']
    subst_vars.stypy_varargs_param_name = None
    subst_vars.stypy_kwargs_param_name = None
    subst_vars.stypy_call_defaults = defaults
    subst_vars.stypy_call_varargs = varargs
    subst_vars.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'subst_vars', ['target', 'source', 'd'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'subst_vars', localization, ['target', 'source', 'd'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'subst_vars(...)' code ##################

    str_55337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, (-1)), 'str', "Substitute any occurence of @foo@ by d['foo'] from source file into\n    target.")
    
    # Assigning a Call to a Name (line 29):
    
    # Assigning a Call to a Name (line 29):
    
    # Call to compile(...): (line 29)
    # Processing the call arguments (line 29)
    str_55340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 21), 'str', '@([a-zA-Z_]+)@')
    # Processing the call keyword arguments (line 29)
    kwargs_55341 = {}
    # Getting the type of 're' (line 29)
    re_55338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 10), 're', False)
    # Obtaining the member 'compile' of a type (line 29)
    compile_55339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 10), re_55338, 'compile')
    # Calling compile(args, kwargs) (line 29)
    compile_call_result_55342 = invoke(stypy.reporting.localization.Localization(__file__, 29, 10), compile_55339, *[str_55340], **kwargs_55341)
    
    # Assigning a type to the variable 'var' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'var', compile_call_result_55342)
    
    # Assigning a Call to a Name (line 30):
    
    # Assigning a Call to a Name (line 30):
    
    # Call to open(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'source' (line 30)
    source_55344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 14), 'source', False)
    str_55345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 22), 'str', 'r')
    # Processing the call keyword arguments (line 30)
    kwargs_55346 = {}
    # Getting the type of 'open' (line 30)
    open_55343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 9), 'open', False)
    # Calling open(args, kwargs) (line 30)
    open_call_result_55347 = invoke(stypy.reporting.localization.Localization(__file__, 30, 9), open_55343, *[source_55344, str_55345], **kwargs_55346)
    
    # Assigning a type to the variable 'fs' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'fs', open_call_result_55347)
    
    # Try-finally block (line 31)
    
    # Assigning a Call to a Name (line 32):
    
    # Assigning a Call to a Name (line 32):
    
    # Call to open(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'target' (line 32)
    target_55349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 18), 'target', False)
    str_55350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 26), 'str', 'w')
    # Processing the call keyword arguments (line 32)
    kwargs_55351 = {}
    # Getting the type of 'open' (line 32)
    open_55348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 13), 'open', False)
    # Calling open(args, kwargs) (line 32)
    open_call_result_55352 = invoke(stypy.reporting.localization.Localization(__file__, 32, 13), open_55348, *[target_55349, str_55350], **kwargs_55351)
    
    # Assigning a type to the variable 'ft' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'ft', open_call_result_55352)
    
    # Try-finally block (line 33)
    
    # Getting the type of 'fs' (line 34)
    fs_55353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 21), 'fs')
    # Testing the type of a for loop iterable (line 34)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 34, 12), fs_55353)
    # Getting the type of the for loop variable (line 34)
    for_loop_var_55354 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 34, 12), fs_55353)
    # Assigning a type to the variable 'l' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'l', for_loop_var_55354)
    # SSA begins for a for statement (line 34)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 35):
    
    # Assigning a Call to a Name (line 35):
    
    # Call to search(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'l' (line 35)
    l_55357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 31), 'l', False)
    # Processing the call keyword arguments (line 35)
    kwargs_55358 = {}
    # Getting the type of 'var' (line 35)
    var_55355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 20), 'var', False)
    # Obtaining the member 'search' of a type (line 35)
    search_55356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 20), var_55355, 'search')
    # Calling search(args, kwargs) (line 35)
    search_call_result_55359 = invoke(stypy.reporting.localization.Localization(__file__, 35, 20), search_55356, *[l_55357], **kwargs_55358)
    
    # Assigning a type to the variable 'm' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 16), 'm', search_call_result_55359)
    
    # Getting the type of 'm' (line 36)
    m_55360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 19), 'm')
    # Testing the type of an if condition (line 36)
    if_condition_55361 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 16), m_55360)
    # Assigning a type to the variable 'if_condition_55361' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'if_condition_55361', if_condition_55361)
    # SSA begins for if statement (line 36)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to write(...): (line 37)
    # Processing the call arguments (line 37)
    
    # Call to replace(...): (line 37)
    # Processing the call arguments (line 37)
    str_55366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 39), 'str', '@%s@')
    
    # Call to group(...): (line 37)
    # Processing the call arguments (line 37)
    int_55369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 56), 'int')
    # Processing the call keyword arguments (line 37)
    kwargs_55370 = {}
    # Getting the type of 'm' (line 37)
    m_55367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 48), 'm', False)
    # Obtaining the member 'group' of a type (line 37)
    group_55368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 48), m_55367, 'group')
    # Calling group(args, kwargs) (line 37)
    group_call_result_55371 = invoke(stypy.reporting.localization.Localization(__file__, 37, 48), group_55368, *[int_55369], **kwargs_55370)
    
    # Applying the binary operator '%' (line 37)
    result_mod_55372 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 39), '%', str_55366, group_call_result_55371)
    
    
    # Obtaining the type of the subscript
    
    # Call to group(...): (line 37)
    # Processing the call arguments (line 37)
    int_55375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 70), 'int')
    # Processing the call keyword arguments (line 37)
    kwargs_55376 = {}
    # Getting the type of 'm' (line 37)
    m_55373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 62), 'm', False)
    # Obtaining the member 'group' of a type (line 37)
    group_55374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 62), m_55373, 'group')
    # Calling group(args, kwargs) (line 37)
    group_call_result_55377 = invoke(stypy.reporting.localization.Localization(__file__, 37, 62), group_55374, *[int_55375], **kwargs_55376)
    
    # Getting the type of 'd' (line 37)
    d_55378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 60), 'd', False)
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___55379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 60), d_55378, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 37)
    subscript_call_result_55380 = invoke(stypy.reporting.localization.Localization(__file__, 37, 60), getitem___55379, group_call_result_55377)
    
    # Processing the call keyword arguments (line 37)
    kwargs_55381 = {}
    # Getting the type of 'l' (line 37)
    l_55364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 29), 'l', False)
    # Obtaining the member 'replace' of a type (line 37)
    replace_55365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 29), l_55364, 'replace')
    # Calling replace(args, kwargs) (line 37)
    replace_call_result_55382 = invoke(stypy.reporting.localization.Localization(__file__, 37, 29), replace_55365, *[result_mod_55372, subscript_call_result_55380], **kwargs_55381)
    
    # Processing the call keyword arguments (line 37)
    kwargs_55383 = {}
    # Getting the type of 'ft' (line 37)
    ft_55362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 20), 'ft', False)
    # Obtaining the member 'write' of a type (line 37)
    write_55363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 20), ft_55362, 'write')
    # Calling write(args, kwargs) (line 37)
    write_call_result_55384 = invoke(stypy.reporting.localization.Localization(__file__, 37, 20), write_55363, *[replace_call_result_55382], **kwargs_55383)
    
    # SSA branch for the else part of an if statement (line 36)
    module_type_store.open_ssa_branch('else')
    
    # Call to write(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'l' (line 39)
    l_55387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 29), 'l', False)
    # Processing the call keyword arguments (line 39)
    kwargs_55388 = {}
    # Getting the type of 'ft' (line 39)
    ft_55385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 'ft', False)
    # Obtaining the member 'write' of a type (line 39)
    write_55386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 20), ft_55385, 'write')
    # Calling write(args, kwargs) (line 39)
    write_call_result_55389 = invoke(stypy.reporting.localization.Localization(__file__, 39, 20), write_55386, *[l_55387], **kwargs_55388)
    
    # SSA join for if statement (line 36)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # finally branch of the try-finally block (line 33)
    
    # Call to close(...): (line 41)
    # Processing the call keyword arguments (line 41)
    kwargs_55392 = {}
    # Getting the type of 'ft' (line 41)
    ft_55390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'ft', False)
    # Obtaining the member 'close' of a type (line 41)
    close_55391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 12), ft_55390, 'close')
    # Calling close(args, kwargs) (line 41)
    close_call_result_55393 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), close_55391, *[], **kwargs_55392)
    
    
    
    # finally branch of the try-finally block (line 31)
    
    # Call to close(...): (line 43)
    # Processing the call keyword arguments (line 43)
    kwargs_55396 = {}
    # Getting the type of 'fs' (line 43)
    fs_55394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'fs', False)
    # Obtaining the member 'close' of a type (line 43)
    close_55395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), fs_55394, 'close')
    # Calling close(args, kwargs) (line 43)
    close_call_result_55397 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), close_55395, *[], **kwargs_55396)
    
    
    
    # ################# End of 'subst_vars(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'subst_vars' in the type store
    # Getting the type of 'stypy_return_type' (line 26)
    stypy_return_type_55398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_55398)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'subst_vars'
    return stypy_return_type_55398

# Assigning a type to the variable 'subst_vars' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'subst_vars', subst_vars)
# Declaration of the 'build_src' class
# Getting the type of 'build_ext' (line 45)
build_ext_55399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 16), 'build_ext')
# Obtaining the member 'build_ext' of a type (line 45)
build_ext_55400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 16), build_ext_55399, 'build_ext')

class build_src(build_ext_55400, ):
    
    # Assigning a Str to a Name (line 47):
    
    # Assigning a List to a Name (line 49):
    
    # Assigning a List to a Name (line 63):
    
    # Assigning a List to a Name (line 65):

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 67, 4, False)
        # Assigning a type to the variable 'self' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_src.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        build_src.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_src.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_src.initialize_options.__dict__.__setitem__('stypy_function_name', 'build_src.initialize_options')
        build_src.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        build_src.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_src.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_src.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_src.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_src.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_src.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_src.initialize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 68):
        
        # Assigning a Name to a Attribute (line 68):
        # Getting the type of 'None' (line 68)
        None_55401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 26), 'None')
        # Getting the type of 'self' (line 68)
        self_55402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'self')
        # Setting the type of the member 'extensions' of a type (line 68)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), self_55402, 'extensions', None_55401)
        
        # Assigning a Name to a Attribute (line 69):
        
        # Assigning a Name to a Attribute (line 69):
        # Getting the type of 'None' (line 69)
        None_55403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 23), 'None')
        # Getting the type of 'self' (line 69)
        self_55404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'self')
        # Setting the type of the member 'package' of a type (line 69)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), self_55404, 'package', None_55403)
        
        # Assigning a Name to a Attribute (line 70):
        
        # Assigning a Name to a Attribute (line 70):
        # Getting the type of 'None' (line 70)
        None_55405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 26), 'None')
        # Getting the type of 'self' (line 70)
        self_55406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'self')
        # Setting the type of the member 'py_modules' of a type (line 70)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), self_55406, 'py_modules', None_55405)
        
        # Assigning a Name to a Attribute (line 71):
        
        # Assigning a Name to a Attribute (line 71):
        # Getting the type of 'None' (line 71)
        None_55407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 31), 'None')
        # Getting the type of 'self' (line 71)
        self_55408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'self')
        # Setting the type of the member 'py_modules_dict' of a type (line 71)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), self_55408, 'py_modules_dict', None_55407)
        
        # Assigning a Name to a Attribute (line 72):
        
        # Assigning a Name to a Attribute (line 72):
        # Getting the type of 'None' (line 72)
        None_55409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 25), 'None')
        # Getting the type of 'self' (line 72)
        self_55410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'self')
        # Setting the type of the member 'build_src' of a type (line 72)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), self_55410, 'build_src', None_55409)
        
        # Assigning a Name to a Attribute (line 73):
        
        # Assigning a Name to a Attribute (line 73):
        # Getting the type of 'None' (line 73)
        None_55411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 25), 'None')
        # Getting the type of 'self' (line 73)
        self_55412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'self')
        # Setting the type of the member 'build_lib' of a type (line 73)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), self_55412, 'build_lib', None_55411)
        
        # Assigning a Name to a Attribute (line 74):
        
        # Assigning a Name to a Attribute (line 74):
        # Getting the type of 'None' (line 74)
        None_55413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 26), 'None')
        # Getting the type of 'self' (line 74)
        self_55414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'self')
        # Setting the type of the member 'build_base' of a type (line 74)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), self_55414, 'build_base', None_55413)
        
        # Assigning a Name to a Attribute (line 75):
        
        # Assigning a Name to a Attribute (line 75):
        # Getting the type of 'None' (line 75)
        None_55415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 21), 'None')
        # Getting the type of 'self' (line 75)
        self_55416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'self')
        # Setting the type of the member 'force' of a type (line 75)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), self_55416, 'force', None_55415)
        
        # Assigning a Name to a Attribute (line 76):
        
        # Assigning a Name to a Attribute (line 76):
        # Getting the type of 'None' (line 76)
        None_55417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'None')
        # Getting the type of 'self' (line 76)
        self_55418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'self')
        # Setting the type of the member 'inplace' of a type (line 76)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), self_55418, 'inplace', None_55417)
        
        # Assigning a Name to a Attribute (line 77):
        
        # Assigning a Name to a Attribute (line 77):
        # Getting the type of 'None' (line 77)
        None_55419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 27), 'None')
        # Getting the type of 'self' (line 77)
        self_55420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'self')
        # Setting the type of the member 'package_dir' of a type (line 77)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), self_55420, 'package_dir', None_55419)
        
        # Assigning a Name to a Attribute (line 78):
        
        # Assigning a Name to a Attribute (line 78):
        # Getting the type of 'None' (line 78)
        None_55421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 25), 'None')
        # Getting the type of 'self' (line 78)
        self_55422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'self')
        # Setting the type of the member 'f2pyflags' of a type (line 78)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), self_55422, 'f2pyflags', None_55421)
        
        # Assigning a Name to a Attribute (line 79):
        
        # Assigning a Name to a Attribute (line 79):
        # Getting the type of 'None' (line 79)
        None_55423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 25), 'None')
        # Getting the type of 'self' (line 79)
        self_55424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'self')
        # Setting the type of the member 'f2py_opts' of a type (line 79)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), self_55424, 'f2py_opts', None_55423)
        
        # Assigning a Name to a Attribute (line 80):
        
        # Assigning a Name to a Attribute (line 80):
        # Getting the type of 'None' (line 80)
        None_55425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 25), 'None')
        # Getting the type of 'self' (line 80)
        self_55426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'self')
        # Setting the type of the member 'swigflags' of a type (line 80)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), self_55426, 'swigflags', None_55425)
        
        # Assigning a Name to a Attribute (line 81):
        
        # Assigning a Name to a Attribute (line 81):
        # Getting the type of 'None' (line 81)
        None_55427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 25), 'None')
        # Getting the type of 'self' (line 81)
        self_55428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'self')
        # Setting the type of the member 'swig_opts' of a type (line 81)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), self_55428, 'swig_opts', None_55427)
        
        # Assigning a Name to a Attribute (line 82):
        
        # Assigning a Name to a Attribute (line 82):
        # Getting the type of 'None' (line 82)
        None_55429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 24), 'None')
        # Getting the type of 'self' (line 82)
        self_55430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'self')
        # Setting the type of the member 'swig_cpp' of a type (line 82)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), self_55430, 'swig_cpp', None_55429)
        
        # Assigning a Name to a Attribute (line 83):
        
        # Assigning a Name to a Attribute (line 83):
        # Getting the type of 'None' (line 83)
        None_55431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'None')
        # Getting the type of 'self' (line 83)
        self_55432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'self')
        # Setting the type of the member 'swig' of a type (line 83)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), self_55432, 'swig', None_55431)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 67)
        stypy_return_type_55433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_55433)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_55433


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 85, 4, False)
        # Assigning a type to the variable 'self' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_src.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        build_src.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_src.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_src.finalize_options.__dict__.__setitem__('stypy_function_name', 'build_src.finalize_options')
        build_src.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        build_src.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_src.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_src.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_src.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_src.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_src.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_src.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to set_undefined_options(...): (line 86)
        # Processing the call arguments (line 86)
        str_55436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 35), 'str', 'build')
        
        # Obtaining an instance of the builtin type 'tuple' (line 87)
        tuple_55437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 87)
        # Adding element type (line 87)
        str_55438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 36), 'str', 'build_base')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 36), tuple_55437, str_55438)
        # Adding element type (line 87)
        str_55439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 50), 'str', 'build_base')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 36), tuple_55437, str_55439)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 88)
        tuple_55440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 88)
        # Adding element type (line 88)
        str_55441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 36), 'str', 'build_lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 36), tuple_55440, str_55441)
        # Adding element type (line 88)
        str_55442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 49), 'str', 'build_lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 36), tuple_55440, str_55442)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 89)
        tuple_55443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 89)
        # Adding element type (line 89)
        str_55444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 36), 'str', 'force')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 36), tuple_55443, str_55444)
        # Adding element type (line 89)
        str_55445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 45), 'str', 'force')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 36), tuple_55443, str_55445)
        
        # Processing the call keyword arguments (line 86)
        kwargs_55446 = {}
        # Getting the type of 'self' (line 86)
        self_55434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'self', False)
        # Obtaining the member 'set_undefined_options' of a type (line 86)
        set_undefined_options_55435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), self_55434, 'set_undefined_options')
        # Calling set_undefined_options(args, kwargs) (line 86)
        set_undefined_options_call_result_55447 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), set_undefined_options_55435, *[str_55436, tuple_55437, tuple_55440, tuple_55443], **kwargs_55446)
        
        
        # Type idiom detected: calculating its left and rigth part (line 90)
        # Getting the type of 'self' (line 90)
        self_55448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 11), 'self')
        # Obtaining the member 'package' of a type (line 90)
        package_55449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 11), self_55448, 'package')
        # Getting the type of 'None' (line 90)
        None_55450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 27), 'None')
        
        (may_be_55451, more_types_in_union_55452) = may_be_none(package_55449, None_55450)

        if may_be_55451:

            if more_types_in_union_55452:
                # Runtime conditional SSA (line 90)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Attribute (line 91):
            
            # Assigning a Attribute to a Attribute (line 91):
            # Getting the type of 'self' (line 91)
            self_55453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 27), 'self')
            # Obtaining the member 'distribution' of a type (line 91)
            distribution_55454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 27), self_55453, 'distribution')
            # Obtaining the member 'ext_package' of a type (line 91)
            ext_package_55455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 27), distribution_55454, 'ext_package')
            # Getting the type of 'self' (line 91)
            self_55456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'self')
            # Setting the type of the member 'package' of a type (line 91)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 12), self_55456, 'package', ext_package_55455)

            if more_types_in_union_55452:
                # SSA join for if statement (line 90)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Attribute (line 92):
        
        # Assigning a Attribute to a Attribute (line 92):
        # Getting the type of 'self' (line 92)
        self_55457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 26), 'self')
        # Obtaining the member 'distribution' of a type (line 92)
        distribution_55458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 26), self_55457, 'distribution')
        # Obtaining the member 'ext_modules' of a type (line 92)
        ext_modules_55459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 26), distribution_55458, 'ext_modules')
        # Getting the type of 'self' (line 92)
        self_55460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'self')
        # Setting the type of the member 'extensions' of a type (line 92)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), self_55460, 'extensions', ext_modules_55459)
        
        # Assigning a BoolOp to a Attribute (line 93):
        
        # Assigning a BoolOp to a Attribute (line 93):
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 93)
        self_55461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 25), 'self')
        # Obtaining the member 'distribution' of a type (line 93)
        distribution_55462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 25), self_55461, 'distribution')
        # Obtaining the member 'libraries' of a type (line 93)
        libraries_55463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 25), distribution_55462, 'libraries')
        
        # Obtaining an instance of the builtin type 'list' (line 93)
        list_55464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 93)
        
        # Applying the binary operator 'or' (line 93)
        result_or_keyword_55465 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 25), 'or', libraries_55463, list_55464)
        
        # Getting the type of 'self' (line 93)
        self_55466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'self')
        # Setting the type of the member 'libraries' of a type (line 93)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), self_55466, 'libraries', result_or_keyword_55465)
        
        # Assigning a BoolOp to a Attribute (line 94):
        
        # Assigning a BoolOp to a Attribute (line 94):
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 94)
        self_55467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 26), 'self')
        # Obtaining the member 'distribution' of a type (line 94)
        distribution_55468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 26), self_55467, 'distribution')
        # Obtaining the member 'py_modules' of a type (line 94)
        py_modules_55469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 26), distribution_55468, 'py_modules')
        
        # Obtaining an instance of the builtin type 'list' (line 94)
        list_55470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 94)
        
        # Applying the binary operator 'or' (line 94)
        result_or_keyword_55471 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 26), 'or', py_modules_55469, list_55470)
        
        # Getting the type of 'self' (line 94)
        self_55472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'self')
        # Setting the type of the member 'py_modules' of a type (line 94)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), self_55472, 'py_modules', result_or_keyword_55471)
        
        # Assigning a BoolOp to a Attribute (line 95):
        
        # Assigning a BoolOp to a Attribute (line 95):
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 95)
        self_55473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 26), 'self')
        # Obtaining the member 'distribution' of a type (line 95)
        distribution_55474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 26), self_55473, 'distribution')
        # Obtaining the member 'data_files' of a type (line 95)
        data_files_55475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 26), distribution_55474, 'data_files')
        
        # Obtaining an instance of the builtin type 'list' (line 95)
        list_55476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 95)
        
        # Applying the binary operator 'or' (line 95)
        result_or_keyword_55477 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 26), 'or', data_files_55475, list_55476)
        
        # Getting the type of 'self' (line 95)
        self_55478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'self')
        # Setting the type of the member 'data_files' of a type (line 95)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), self_55478, 'data_files', result_or_keyword_55477)
        
        # Type idiom detected: calculating its left and rigth part (line 97)
        # Getting the type of 'self' (line 97)
        self_55479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 'self')
        # Obtaining the member 'build_src' of a type (line 97)
        build_src_55480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 11), self_55479, 'build_src')
        # Getting the type of 'None' (line 97)
        None_55481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 29), 'None')
        
        (may_be_55482, more_types_in_union_55483) = may_be_none(build_src_55480, None_55481)

        if may_be_55482:

            if more_types_in_union_55483:
                # Runtime conditional SSA (line 97)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 98):
            
            # Assigning a BinOp to a Name (line 98):
            str_55484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 29), 'str', '.%s-%s')
            
            # Obtaining an instance of the builtin type 'tuple' (line 98)
            tuple_55485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 41), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 98)
            # Adding element type (line 98)
            
            # Call to get_platform(...): (line 98)
            # Processing the call keyword arguments (line 98)
            kwargs_55487 = {}
            # Getting the type of 'get_platform' (line 98)
            get_platform_55486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 41), 'get_platform', False)
            # Calling get_platform(args, kwargs) (line 98)
            get_platform_call_result_55488 = invoke(stypy.reporting.localization.Localization(__file__, 98, 41), get_platform_55486, *[], **kwargs_55487)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 41), tuple_55485, get_platform_call_result_55488)
            # Adding element type (line 98)
            
            # Obtaining the type of the subscript
            int_55489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 69), 'int')
            int_55490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 71), 'int')
            slice_55491 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 98, 57), int_55489, int_55490, None)
            # Getting the type of 'sys' (line 98)
            sys_55492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 57), 'sys')
            # Obtaining the member 'version' of a type (line 98)
            version_55493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 57), sys_55492, 'version')
            # Obtaining the member '__getitem__' of a type (line 98)
            getitem___55494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 57), version_55493, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 98)
            subscript_call_result_55495 = invoke(stypy.reporting.localization.Localization(__file__, 98, 57), getitem___55494, slice_55491)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 41), tuple_55485, subscript_call_result_55495)
            
            # Applying the binary operator '%' (line 98)
            result_mod_55496 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 29), '%', str_55484, tuple_55485)
            
            # Assigning a type to the variable 'plat_specifier' (line 98)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'plat_specifier', result_mod_55496)
            
            # Assigning a Call to a Attribute (line 99):
            
            # Assigning a Call to a Attribute (line 99):
            
            # Call to join(...): (line 99)
            # Processing the call arguments (line 99)
            # Getting the type of 'self' (line 99)
            self_55500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 42), 'self', False)
            # Obtaining the member 'build_base' of a type (line 99)
            build_base_55501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 42), self_55500, 'build_base')
            str_55502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 59), 'str', 'src')
            # Getting the type of 'plat_specifier' (line 99)
            plat_specifier_55503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 65), 'plat_specifier', False)
            # Applying the binary operator '+' (line 99)
            result_add_55504 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 59), '+', str_55502, plat_specifier_55503)
            
            # Processing the call keyword arguments (line 99)
            kwargs_55505 = {}
            # Getting the type of 'os' (line 99)
            os_55497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 29), 'os', False)
            # Obtaining the member 'path' of a type (line 99)
            path_55498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 29), os_55497, 'path')
            # Obtaining the member 'join' of a type (line 99)
            join_55499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 29), path_55498, 'join')
            # Calling join(args, kwargs) (line 99)
            join_call_result_55506 = invoke(stypy.reporting.localization.Localization(__file__, 99, 29), join_55499, *[build_base_55501, result_add_55504], **kwargs_55505)
            
            # Getting the type of 'self' (line 99)
            self_55507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'self')
            # Setting the type of the member 'build_src' of a type (line 99)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 12), self_55507, 'build_src', join_call_result_55506)

            if more_types_in_union_55483:
                # SSA join for if statement (line 97)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Dict to a Attribute (line 102):
        
        # Assigning a Dict to a Attribute (line 102):
        
        # Obtaining an instance of the builtin type 'dict' (line 102)
        dict_55508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 31), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 102)
        
        # Getting the type of 'self' (line 102)
        self_55509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'self')
        # Setting the type of the member 'py_modules_dict' of a type (line 102)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), self_55509, 'py_modules_dict', dict_55508)
        
        # Getting the type of 'self' (line 104)
        self_55510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 11), 'self')
        # Obtaining the member 'f2pyflags' of a type (line 104)
        f2pyflags_55511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 11), self_55510, 'f2pyflags')
        # Testing the type of an if condition (line 104)
        if_condition_55512 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 8), f2pyflags_55511)
        # Assigning a type to the variable 'if_condition_55512' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'if_condition_55512', if_condition_55512)
        # SSA begins for if statement (line 104)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 105)
        self_55513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 15), 'self')
        # Obtaining the member 'f2py_opts' of a type (line 105)
        f2py_opts_55514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 15), self_55513, 'f2py_opts')
        # Testing the type of an if condition (line 105)
        if_condition_55515 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 12), f2py_opts_55514)
        # Assigning a type to the variable 'if_condition_55515' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'if_condition_55515', if_condition_55515)
        # SSA begins for if statement (line 105)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 106)
        # Processing the call arguments (line 106)
        str_55518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 25), 'str', 'ignoring --f2pyflags as --f2py-opts already used')
        # Processing the call keyword arguments (line 106)
        kwargs_55519 = {}
        # Getting the type of 'log' (line 106)
        log_55516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'log', False)
        # Obtaining the member 'warn' of a type (line 106)
        warn_55517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 16), log_55516, 'warn')
        # Calling warn(args, kwargs) (line 106)
        warn_call_result_55520 = invoke(stypy.reporting.localization.Localization(__file__, 106, 16), warn_55517, *[str_55518], **kwargs_55519)
        
        # SSA branch for the else part of an if statement (line 105)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Attribute (line 108):
        
        # Assigning a Attribute to a Attribute (line 108):
        # Getting the type of 'self' (line 108)
        self_55521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 33), 'self')
        # Obtaining the member 'f2pyflags' of a type (line 108)
        f2pyflags_55522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 33), self_55521, 'f2pyflags')
        # Getting the type of 'self' (line 108)
        self_55523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'self')
        # Setting the type of the member 'f2py_opts' of a type (line 108)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 16), self_55523, 'f2py_opts', f2pyflags_55522)
        # SSA join for if statement (line 105)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 109):
        
        # Assigning a Name to a Attribute (line 109):
        # Getting the type of 'None' (line 109)
        None_55524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 29), 'None')
        # Getting the type of 'self' (line 109)
        self_55525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'self')
        # Setting the type of the member 'f2pyflags' of a type (line 109)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 12), self_55525, 'f2pyflags', None_55524)
        # SSA join for if statement (line 104)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 110)
        # Getting the type of 'self' (line 110)
        self_55526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'self')
        # Obtaining the member 'f2py_opts' of a type (line 110)
        f2py_opts_55527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 11), self_55526, 'f2py_opts')
        # Getting the type of 'None' (line 110)
        None_55528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 29), 'None')
        
        (may_be_55529, more_types_in_union_55530) = may_be_none(f2py_opts_55527, None_55528)

        if may_be_55529:

            if more_types_in_union_55530:
                # Runtime conditional SSA (line 110)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a List to a Attribute (line 111):
            
            # Assigning a List to a Attribute (line 111):
            
            # Obtaining an instance of the builtin type 'list' (line 111)
            list_55531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 29), 'list')
            # Adding type elements to the builtin type 'list' instance (line 111)
            
            # Getting the type of 'self' (line 111)
            self_55532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'self')
            # Setting the type of the member 'f2py_opts' of a type (line 111)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 12), self_55532, 'f2py_opts', list_55531)

            if more_types_in_union_55530:
                # Runtime conditional SSA for else branch (line 110)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_55529) or more_types_in_union_55530):
            
            # Assigning a Call to a Attribute (line 113):
            
            # Assigning a Call to a Attribute (line 113):
            
            # Call to split(...): (line 113)
            # Processing the call arguments (line 113)
            # Getting the type of 'self' (line 113)
            self_55535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 41), 'self', False)
            # Obtaining the member 'f2py_opts' of a type (line 113)
            f2py_opts_55536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 41), self_55535, 'f2py_opts')
            # Processing the call keyword arguments (line 113)
            kwargs_55537 = {}
            # Getting the type of 'shlex' (line 113)
            shlex_55533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 29), 'shlex', False)
            # Obtaining the member 'split' of a type (line 113)
            split_55534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 29), shlex_55533, 'split')
            # Calling split(args, kwargs) (line 113)
            split_call_result_55538 = invoke(stypy.reporting.localization.Localization(__file__, 113, 29), split_55534, *[f2py_opts_55536], **kwargs_55537)
            
            # Getting the type of 'self' (line 113)
            self_55539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'self')
            # Setting the type of the member 'f2py_opts' of a type (line 113)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 12), self_55539, 'f2py_opts', split_call_result_55538)

            if (may_be_55529 and more_types_in_union_55530):
                # SSA join for if statement (line 110)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'self' (line 115)
        self_55540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 11), 'self')
        # Obtaining the member 'swigflags' of a type (line 115)
        swigflags_55541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 11), self_55540, 'swigflags')
        # Testing the type of an if condition (line 115)
        if_condition_55542 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 115, 8), swigflags_55541)
        # Assigning a type to the variable 'if_condition_55542' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'if_condition_55542', if_condition_55542)
        # SSA begins for if statement (line 115)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 116)
        self_55543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 15), 'self')
        # Obtaining the member 'swig_opts' of a type (line 116)
        swig_opts_55544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 15), self_55543, 'swig_opts')
        # Testing the type of an if condition (line 116)
        if_condition_55545 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 12), swig_opts_55544)
        # Assigning a type to the variable 'if_condition_55545' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'if_condition_55545', if_condition_55545)
        # SSA begins for if statement (line 116)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 117)
        # Processing the call arguments (line 117)
        str_55548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 25), 'str', 'ignoring --swigflags as --swig-opts already used')
        # Processing the call keyword arguments (line 117)
        kwargs_55549 = {}
        # Getting the type of 'log' (line 117)
        log_55546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'log', False)
        # Obtaining the member 'warn' of a type (line 117)
        warn_55547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 16), log_55546, 'warn')
        # Calling warn(args, kwargs) (line 117)
        warn_call_result_55550 = invoke(stypy.reporting.localization.Localization(__file__, 117, 16), warn_55547, *[str_55548], **kwargs_55549)
        
        # SSA branch for the else part of an if statement (line 116)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Attribute (line 119):
        
        # Assigning a Attribute to a Attribute (line 119):
        # Getting the type of 'self' (line 119)
        self_55551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 33), 'self')
        # Obtaining the member 'swigflags' of a type (line 119)
        swigflags_55552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 33), self_55551, 'swigflags')
        # Getting the type of 'self' (line 119)
        self_55553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'self')
        # Setting the type of the member 'swig_opts' of a type (line 119)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 16), self_55553, 'swig_opts', swigflags_55552)
        # SSA join for if statement (line 116)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 120):
        
        # Assigning a Name to a Attribute (line 120):
        # Getting the type of 'None' (line 120)
        None_55554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 29), 'None')
        # Getting the type of 'self' (line 120)
        self_55555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'self')
        # Setting the type of the member 'swigflags' of a type (line 120)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 12), self_55555, 'swigflags', None_55554)
        # SSA join for if statement (line 115)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 122)
        # Getting the type of 'self' (line 122)
        self_55556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 11), 'self')
        # Obtaining the member 'swig_opts' of a type (line 122)
        swig_opts_55557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 11), self_55556, 'swig_opts')
        # Getting the type of 'None' (line 122)
        None_55558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 29), 'None')
        
        (may_be_55559, more_types_in_union_55560) = may_be_none(swig_opts_55557, None_55558)

        if may_be_55559:

            if more_types_in_union_55560:
                # Runtime conditional SSA (line 122)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a List to a Attribute (line 123):
            
            # Assigning a List to a Attribute (line 123):
            
            # Obtaining an instance of the builtin type 'list' (line 123)
            list_55561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 29), 'list')
            # Adding type elements to the builtin type 'list' instance (line 123)
            
            # Getting the type of 'self' (line 123)
            self_55562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'self')
            # Setting the type of the member 'swig_opts' of a type (line 123)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 12), self_55562, 'swig_opts', list_55561)

            if more_types_in_union_55560:
                # Runtime conditional SSA for else branch (line 122)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_55559) or more_types_in_union_55560):
            
            # Assigning a Call to a Attribute (line 125):
            
            # Assigning a Call to a Attribute (line 125):
            
            # Call to split(...): (line 125)
            # Processing the call arguments (line 125)
            # Getting the type of 'self' (line 125)
            self_55565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 41), 'self', False)
            # Obtaining the member 'swig_opts' of a type (line 125)
            swig_opts_55566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 41), self_55565, 'swig_opts')
            # Processing the call keyword arguments (line 125)
            kwargs_55567 = {}
            # Getting the type of 'shlex' (line 125)
            shlex_55563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 29), 'shlex', False)
            # Obtaining the member 'split' of a type (line 125)
            split_55564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 29), shlex_55563, 'split')
            # Calling split(args, kwargs) (line 125)
            split_call_result_55568 = invoke(stypy.reporting.localization.Localization(__file__, 125, 29), split_55564, *[swig_opts_55566], **kwargs_55567)
            
            # Getting the type of 'self' (line 125)
            self_55569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'self')
            # Setting the type of the member 'swig_opts' of a type (line 125)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 12), self_55569, 'swig_opts', split_call_result_55568)

            if (may_be_55559 and more_types_in_union_55560):
                # SSA join for if statement (line 122)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 128):
        
        # Assigning a Call to a Name (line 128):
        
        # Call to get_finalized_command(...): (line 128)
        # Processing the call arguments (line 128)
        str_55572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 47), 'str', 'build_ext')
        # Processing the call keyword arguments (line 128)
        kwargs_55573 = {}
        # Getting the type of 'self' (line 128)
        self_55570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 20), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 128)
        get_finalized_command_55571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 20), self_55570, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 128)
        get_finalized_command_call_result_55574 = invoke(stypy.reporting.localization.Localization(__file__, 128, 20), get_finalized_command_55571, *[str_55572], **kwargs_55573)
        
        # Assigning a type to the variable 'build_ext' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'build_ext', get_finalized_command_call_result_55574)
        
        # Type idiom detected: calculating its left and rigth part (line 129)
        # Getting the type of 'self' (line 129)
        self_55575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 11), 'self')
        # Obtaining the member 'inplace' of a type (line 129)
        inplace_55576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 11), self_55575, 'inplace')
        # Getting the type of 'None' (line 129)
        None_55577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 27), 'None')
        
        (may_be_55578, more_types_in_union_55579) = may_be_none(inplace_55576, None_55577)

        if may_be_55578:

            if more_types_in_union_55579:
                # Runtime conditional SSA (line 129)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Attribute (line 130):
            
            # Assigning a Attribute to a Attribute (line 130):
            # Getting the type of 'build_ext' (line 130)
            build_ext_55580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 27), 'build_ext')
            # Obtaining the member 'inplace' of a type (line 130)
            inplace_55581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 27), build_ext_55580, 'inplace')
            # Getting the type of 'self' (line 130)
            self_55582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'self')
            # Setting the type of the member 'inplace' of a type (line 130)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 12), self_55582, 'inplace', inplace_55581)

            if more_types_in_union_55579:
                # SSA join for if statement (line 129)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 131)
        # Getting the type of 'self' (line 131)
        self_55583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 11), 'self')
        # Obtaining the member 'swig_cpp' of a type (line 131)
        swig_cpp_55584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 11), self_55583, 'swig_cpp')
        # Getting the type of 'None' (line 131)
        None_55585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 28), 'None')
        
        (may_be_55586, more_types_in_union_55587) = may_be_none(swig_cpp_55584, None_55585)

        if may_be_55586:

            if more_types_in_union_55587:
                # Runtime conditional SSA (line 131)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Attribute (line 132):
            
            # Assigning a Attribute to a Attribute (line 132):
            # Getting the type of 'build_ext' (line 132)
            build_ext_55588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 28), 'build_ext')
            # Obtaining the member 'swig_cpp' of a type (line 132)
            swig_cpp_55589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 28), build_ext_55588, 'swig_cpp')
            # Getting the type of 'self' (line 132)
            self_55590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'self')
            # Setting the type of the member 'swig_cpp' of a type (line 132)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 12), self_55590, 'swig_cpp', swig_cpp_55589)

            if more_types_in_union_55587:
                # SSA join for if statement (line 131)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Obtaining an instance of the builtin type 'list' (line 133)
        list_55591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 133)
        # Adding element type (line 133)
        str_55592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 18), 'str', 'swig')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 17), list_55591, str_55592)
        # Adding element type (line 133)
        str_55593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 26), 'str', 'swig_opt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 17), list_55591, str_55593)
        
        # Testing the type of a for loop iterable (line 133)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 133, 8), list_55591)
        # Getting the type of the for loop variable (line 133)
        for_loop_var_55594 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 133, 8), list_55591)
        # Assigning a type to the variable 'c' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'c', for_loop_var_55594)
        # SSA begins for a for statement (line 133)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 134):
        
        # Assigning a BinOp to a Name (line 134):
        str_55595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 16), 'str', '--')
        
        # Call to replace(...): (line 134)
        # Processing the call arguments (line 134)
        str_55598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 31), 'str', '_')
        str_55599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 36), 'str', '-')
        # Processing the call keyword arguments (line 134)
        kwargs_55600 = {}
        # Getting the type of 'c' (line 134)
        c_55596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 21), 'c', False)
        # Obtaining the member 'replace' of a type (line 134)
        replace_55597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 21), c_55596, 'replace')
        # Calling replace(args, kwargs) (line 134)
        replace_call_result_55601 = invoke(stypy.reporting.localization.Localization(__file__, 134, 21), replace_55597, *[str_55598, str_55599], **kwargs_55600)
        
        # Applying the binary operator '+' (line 134)
        result_add_55602 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 16), '+', str_55595, replace_call_result_55601)
        
        # Assigning a type to the variable 'o' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'o', result_add_55602)
        
        # Assigning a Call to a Name (line 135):
        
        # Assigning a Call to a Name (line 135):
        
        # Call to getattr(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'build_ext' (line 135)
        build_ext_55604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 24), 'build_ext', False)
        # Getting the type of 'c' (line 135)
        c_55605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 35), 'c', False)
        # Getting the type of 'None' (line 135)
        None_55606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 38), 'None', False)
        # Processing the call keyword arguments (line 135)
        kwargs_55607 = {}
        # Getting the type of 'getattr' (line 135)
        getattr_55603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'getattr', False)
        # Calling getattr(args, kwargs) (line 135)
        getattr_call_result_55608 = invoke(stypy.reporting.localization.Localization(__file__, 135, 16), getattr_55603, *[build_ext_55604, c_55605, None_55606], **kwargs_55607)
        
        # Assigning a type to the variable 'v' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'v', getattr_call_result_55608)
        
        # Getting the type of 'v' (line 136)
        v_55609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 15), 'v')
        # Testing the type of an if condition (line 136)
        if_condition_55610 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 136, 12), v_55609)
        # Assigning a type to the variable 'if_condition_55610' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'if_condition_55610', if_condition_55610)
        # SSA begins for if statement (line 136)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to getattr(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'self' (line 137)
        self_55612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 27), 'self', False)
        # Getting the type of 'c' (line 137)
        c_55613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 33), 'c', False)
        # Processing the call keyword arguments (line 137)
        kwargs_55614 = {}
        # Getting the type of 'getattr' (line 137)
        getattr_55611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 19), 'getattr', False)
        # Calling getattr(args, kwargs) (line 137)
        getattr_call_result_55615 = invoke(stypy.reporting.localization.Localization(__file__, 137, 19), getattr_55611, *[self_55612, c_55613], **kwargs_55614)
        
        # Testing the type of an if condition (line 137)
        if_condition_55616 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 16), getattr_call_result_55615)
        # Assigning a type to the variable 'if_condition_55616' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'if_condition_55616', if_condition_55616)
        # SSA begins for if statement (line 137)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 138)
        # Processing the call arguments (line 138)
        str_55619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 29), 'str', 'both build_src and build_ext define %s option')
        # Getting the type of 'o' (line 138)
        o_55620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 80), 'o', False)
        # Applying the binary operator '%' (line 138)
        result_mod_55621 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 29), '%', str_55619, o_55620)
        
        # Processing the call keyword arguments (line 138)
        kwargs_55622 = {}
        # Getting the type of 'log' (line 138)
        log_55617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 20), 'log', False)
        # Obtaining the member 'warn' of a type (line 138)
        warn_55618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 20), log_55617, 'warn')
        # Calling warn(args, kwargs) (line 138)
        warn_call_result_55623 = invoke(stypy.reporting.localization.Localization(__file__, 138, 20), warn_55618, *[result_mod_55621], **kwargs_55622)
        
        # SSA branch for the else part of an if statement (line 137)
        module_type_store.open_ssa_branch('else')
        
        # Call to info(...): (line 140)
        # Processing the call arguments (line 140)
        str_55626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 29), 'str', 'using "%s=%s" option from build_ext command')
        
        # Obtaining an instance of the builtin type 'tuple' (line 140)
        tuple_55627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 78), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 140)
        # Adding element type (line 140)
        # Getting the type of 'o' (line 140)
        o_55628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 78), 'o', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 78), tuple_55627, o_55628)
        # Adding element type (line 140)
        # Getting the type of 'v' (line 140)
        v_55629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 81), 'v', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 78), tuple_55627, v_55629)
        
        # Applying the binary operator '%' (line 140)
        result_mod_55630 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 29), '%', str_55626, tuple_55627)
        
        # Processing the call keyword arguments (line 140)
        kwargs_55631 = {}
        # Getting the type of 'log' (line 140)
        log_55624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 20), 'log', False)
        # Obtaining the member 'info' of a type (line 140)
        info_55625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 20), log_55624, 'info')
        # Calling info(args, kwargs) (line 140)
        info_call_result_55632 = invoke(stypy.reporting.localization.Localization(__file__, 140, 20), info_55625, *[result_mod_55630], **kwargs_55631)
        
        
        # Call to setattr(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'self' (line 141)
        self_55634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 28), 'self', False)
        # Getting the type of 'c' (line 141)
        c_55635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 34), 'c', False)
        # Getting the type of 'v' (line 141)
        v_55636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 37), 'v', False)
        # Processing the call keyword arguments (line 141)
        kwargs_55637 = {}
        # Getting the type of 'setattr' (line 141)
        setattr_55633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 20), 'setattr', False)
        # Calling setattr(args, kwargs) (line 141)
        setattr_call_result_55638 = invoke(stypy.reporting.localization.Localization(__file__, 141, 20), setattr_55633, *[self_55634, c_55635, v_55636], **kwargs_55637)
        
        # SSA join for if statement (line 137)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 136)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 85)
        stypy_return_type_55639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_55639)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_55639


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 143, 4, False)
        # Assigning a type to the variable 'self' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_src.run.__dict__.__setitem__('stypy_localization', localization)
        build_src.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_src.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_src.run.__dict__.__setitem__('stypy_function_name', 'build_src.run')
        build_src.run.__dict__.__setitem__('stypy_param_names_list', [])
        build_src.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_src.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_src.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_src.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_src.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_src.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_src.run', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to info(...): (line 144)
        # Processing the call arguments (line 144)
        str_55642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 17), 'str', 'build_src')
        # Processing the call keyword arguments (line 144)
        kwargs_55643 = {}
        # Getting the type of 'log' (line 144)
        log_55640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 144)
        info_55641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), log_55640, 'info')
        # Calling info(args, kwargs) (line 144)
        info_call_result_55644 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), info_55641, *[str_55642], **kwargs_55643)
        
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 145)
        self_55645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 16), 'self')
        # Obtaining the member 'extensions' of a type (line 145)
        extensions_55646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 16), self_55645, 'extensions')
        # Getting the type of 'self' (line 145)
        self_55647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 35), 'self')
        # Obtaining the member 'libraries' of a type (line 145)
        libraries_55648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 35), self_55647, 'libraries')
        # Applying the binary operator 'or' (line 145)
        result_or_keyword_55649 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 16), 'or', extensions_55646, libraries_55648)
        
        # Applying the 'not' unary operator (line 145)
        result_not__55650 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 11), 'not', result_or_keyword_55649)
        
        # Testing the type of an if condition (line 145)
        if_condition_55651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 145, 8), result_not__55650)
        # Assigning a type to the variable 'if_condition_55651' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'if_condition_55651', if_condition_55651)
        # SSA begins for if statement (line 145)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 145)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to build_sources(...): (line 147)
        # Processing the call keyword arguments (line 147)
        kwargs_55654 = {}
        # Getting the type of 'self' (line 147)
        self_55652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'self', False)
        # Obtaining the member 'build_sources' of a type (line 147)
        build_sources_55653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), self_55652, 'build_sources')
        # Calling build_sources(args, kwargs) (line 147)
        build_sources_call_result_55655 = invoke(stypy.reporting.localization.Localization(__file__, 147, 8), build_sources_55653, *[], **kwargs_55654)
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 143)
        stypy_return_type_55656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_55656)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_55656


    @norecursion
    def build_sources(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'build_sources'
        module_type_store = module_type_store.open_function_context('build_sources', 149, 4, False)
        # Assigning a type to the variable 'self' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_src.build_sources.__dict__.__setitem__('stypy_localization', localization)
        build_src.build_sources.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_src.build_sources.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_src.build_sources.__dict__.__setitem__('stypy_function_name', 'build_src.build_sources')
        build_src.build_sources.__dict__.__setitem__('stypy_param_names_list', [])
        build_src.build_sources.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_src.build_sources.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_src.build_sources.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_src.build_sources.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_src.build_sources.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_src.build_sources.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_src.build_sources', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'build_sources', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'build_sources(...)' code ##################

        
        # Getting the type of 'self' (line 151)
        self_55657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 11), 'self')
        # Obtaining the member 'inplace' of a type (line 151)
        inplace_55658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 11), self_55657, 'inplace')
        # Testing the type of an if condition (line 151)
        if_condition_55659 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 8), inplace_55658)
        # Assigning a type to the variable 'if_condition_55659' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'if_condition_55659', if_condition_55659)
        # SSA begins for if statement (line 151)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 152):
        
        # Assigning a Attribute to a Attribute (line 152):
        
        # Call to get_finalized_command(...): (line 153)
        # Processing the call arguments (line 153)
        str_55662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 48), 'str', 'build_py')
        # Processing the call keyword arguments (line 153)
        kwargs_55663 = {}
        # Getting the type of 'self' (line 153)
        self_55660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 21), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 153)
        get_finalized_command_55661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 21), self_55660, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 153)
        get_finalized_command_call_result_55664 = invoke(stypy.reporting.localization.Localization(__file__, 153, 21), get_finalized_command_55661, *[str_55662], **kwargs_55663)
        
        # Obtaining the member 'get_package_dir' of a type (line 153)
        get_package_dir_55665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 21), get_finalized_command_call_result_55664, 'get_package_dir')
        # Getting the type of 'self' (line 152)
        self_55666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'self')
        # Setting the type of the member 'get_package_dir' of a type (line 152)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 12), self_55666, 'get_package_dir', get_package_dir_55665)
        # SSA join for if statement (line 151)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to build_py_modules_sources(...): (line 155)
        # Processing the call keyword arguments (line 155)
        kwargs_55669 = {}
        # Getting the type of 'self' (line 155)
        self_55667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'self', False)
        # Obtaining the member 'build_py_modules_sources' of a type (line 155)
        build_py_modules_sources_55668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), self_55667, 'build_py_modules_sources')
        # Calling build_py_modules_sources(args, kwargs) (line 155)
        build_py_modules_sources_call_result_55670 = invoke(stypy.reporting.localization.Localization(__file__, 155, 8), build_py_modules_sources_55668, *[], **kwargs_55669)
        
        
        # Getting the type of 'self' (line 157)
        self_55671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 28), 'self')
        # Obtaining the member 'libraries' of a type (line 157)
        libraries_55672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 28), self_55671, 'libraries')
        # Testing the type of a for loop iterable (line 157)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 157, 8), libraries_55672)
        # Getting the type of the for loop variable (line 157)
        for_loop_var_55673 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 157, 8), libraries_55672)
        # Assigning a type to the variable 'libname_info' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'libname_info', for_loop_var_55673)
        # SSA begins for a for statement (line 157)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to build_library_sources(...): (line 158)
        # Getting the type of 'libname_info' (line 158)
        libname_info_55676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 40), 'libname_info', False)
        # Processing the call keyword arguments (line 158)
        kwargs_55677 = {}
        # Getting the type of 'self' (line 158)
        self_55674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'self', False)
        # Obtaining the member 'build_library_sources' of a type (line 158)
        build_library_sources_55675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 12), self_55674, 'build_library_sources')
        # Calling build_library_sources(args, kwargs) (line 158)
        build_library_sources_call_result_55678 = invoke(stypy.reporting.localization.Localization(__file__, 158, 12), build_library_sources_55675, *[libname_info_55676], **kwargs_55677)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 160)
        self_55679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 11), 'self')
        # Obtaining the member 'extensions' of a type (line 160)
        extensions_55680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 11), self_55679, 'extensions')
        # Testing the type of an if condition (line 160)
        if_condition_55681 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 8), extensions_55680)
        # Assigning a type to the variable 'if_condition_55681' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'if_condition_55681', if_condition_55681)
        # SSA begins for if statement (line 160)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to check_extensions_list(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'self' (line 161)
        self_55684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 39), 'self', False)
        # Obtaining the member 'extensions' of a type (line 161)
        extensions_55685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 39), self_55684, 'extensions')
        # Processing the call keyword arguments (line 161)
        kwargs_55686 = {}
        # Getting the type of 'self' (line 161)
        self_55682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'self', False)
        # Obtaining the member 'check_extensions_list' of a type (line 161)
        check_extensions_list_55683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 12), self_55682, 'check_extensions_list')
        # Calling check_extensions_list(args, kwargs) (line 161)
        check_extensions_list_call_result_55687 = invoke(stypy.reporting.localization.Localization(__file__, 161, 12), check_extensions_list_55683, *[extensions_55685], **kwargs_55686)
        
        
        # Getting the type of 'self' (line 163)
        self_55688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 23), 'self')
        # Obtaining the member 'extensions' of a type (line 163)
        extensions_55689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 23), self_55688, 'extensions')
        # Testing the type of a for loop iterable (line 163)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 163, 12), extensions_55689)
        # Getting the type of the for loop variable (line 163)
        for_loop_var_55690 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 163, 12), extensions_55689)
        # Assigning a type to the variable 'ext' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'ext', for_loop_var_55690)
        # SSA begins for a for statement (line 163)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to build_extension_sources(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'ext' (line 164)
        ext_55693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 45), 'ext', False)
        # Processing the call keyword arguments (line 164)
        kwargs_55694 = {}
        # Getting the type of 'self' (line 164)
        self_55691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'self', False)
        # Obtaining the member 'build_extension_sources' of a type (line 164)
        build_extension_sources_55692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 16), self_55691, 'build_extension_sources')
        # Calling build_extension_sources(args, kwargs) (line 164)
        build_extension_sources_call_result_55695 = invoke(stypy.reporting.localization.Localization(__file__, 164, 16), build_extension_sources_55692, *[ext_55693], **kwargs_55694)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 160)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to build_data_files_sources(...): (line 166)
        # Processing the call keyword arguments (line 166)
        kwargs_55698 = {}
        # Getting the type of 'self' (line 166)
        self_55696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'self', False)
        # Obtaining the member 'build_data_files_sources' of a type (line 166)
        build_data_files_sources_55697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), self_55696, 'build_data_files_sources')
        # Calling build_data_files_sources(args, kwargs) (line 166)
        build_data_files_sources_call_result_55699 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), build_data_files_sources_55697, *[], **kwargs_55698)
        
        
        # Call to build_npy_pkg_config(...): (line 167)
        # Processing the call keyword arguments (line 167)
        kwargs_55702 = {}
        # Getting the type of 'self' (line 167)
        self_55700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'self', False)
        # Obtaining the member 'build_npy_pkg_config' of a type (line 167)
        build_npy_pkg_config_55701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), self_55700, 'build_npy_pkg_config')
        # Calling build_npy_pkg_config(args, kwargs) (line 167)
        build_npy_pkg_config_call_result_55703 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), build_npy_pkg_config_55701, *[], **kwargs_55702)
        
        
        # ################# End of 'build_sources(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'build_sources' in the type store
        # Getting the type of 'stypy_return_type' (line 149)
        stypy_return_type_55704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_55704)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'build_sources'
        return stypy_return_type_55704


    @norecursion
    def build_data_files_sources(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'build_data_files_sources'
        module_type_store = module_type_store.open_function_context('build_data_files_sources', 169, 4, False)
        # Assigning a type to the variable 'self' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_src.build_data_files_sources.__dict__.__setitem__('stypy_localization', localization)
        build_src.build_data_files_sources.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_src.build_data_files_sources.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_src.build_data_files_sources.__dict__.__setitem__('stypy_function_name', 'build_src.build_data_files_sources')
        build_src.build_data_files_sources.__dict__.__setitem__('stypy_param_names_list', [])
        build_src.build_data_files_sources.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_src.build_data_files_sources.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_src.build_data_files_sources.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_src.build_data_files_sources.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_src.build_data_files_sources.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_src.build_data_files_sources.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_src.build_data_files_sources', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'build_data_files_sources', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'build_data_files_sources(...)' code ##################

        
        
        # Getting the type of 'self' (line 170)
        self_55705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 15), 'self')
        # Obtaining the member 'data_files' of a type (line 170)
        data_files_55706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 15), self_55705, 'data_files')
        # Applying the 'not' unary operator (line 170)
        result_not__55707 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 11), 'not', data_files_55706)
        
        # Testing the type of an if condition (line 170)
        if_condition_55708 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 8), result_not__55707)
        # Assigning a type to the variable 'if_condition_55708' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'if_condition_55708', if_condition_55708)
        # SSA begins for if statement (line 170)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 170)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to info(...): (line 172)
        # Processing the call arguments (line 172)
        str_55711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 17), 'str', 'building data_files sources')
        # Processing the call keyword arguments (line 172)
        kwargs_55712 = {}
        # Getting the type of 'log' (line 172)
        log_55709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 172)
        info_55710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), log_55709, 'info')
        # Calling info(args, kwargs) (line 172)
        info_call_result_55713 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), info_55710, *[str_55711], **kwargs_55712)
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 173, 8))
        
        # 'from numpy.distutils.misc_util import get_data_files' statement (line 173)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
        import_55714 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 173, 8), 'numpy.distutils.misc_util')

        if (type(import_55714) is not StypyTypeError):

            if (import_55714 != 'pyd_module'):
                __import__(import_55714)
                sys_modules_55715 = sys.modules[import_55714]
                import_from_module(stypy.reporting.localization.Localization(__file__, 173, 8), 'numpy.distutils.misc_util', sys_modules_55715.module_type_store, module_type_store, ['get_data_files'])
                nest_module(stypy.reporting.localization.Localization(__file__, 173, 8), __file__, sys_modules_55715, sys_modules_55715.module_type_store, module_type_store)
            else:
                from numpy.distutils.misc_util import get_data_files

                import_from_module(stypy.reporting.localization.Localization(__file__, 173, 8), 'numpy.distutils.misc_util', None, module_type_store, ['get_data_files'], [get_data_files])

        else:
            # Assigning a type to the variable 'numpy.distutils.misc_util' (line 173)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'numpy.distutils.misc_util', import_55714)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')
        
        
        # Assigning a List to a Name (line 174):
        
        # Assigning a List to a Name (line 174):
        
        # Obtaining an instance of the builtin type 'list' (line 174)
        list_55716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 174)
        
        # Assigning a type to the variable 'new_data_files' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'new_data_files', list_55716)
        
        # Getting the type of 'self' (line 175)
        self_55717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 20), 'self')
        # Obtaining the member 'data_files' of a type (line 175)
        data_files_55718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 20), self_55717, 'data_files')
        # Testing the type of a for loop iterable (line 175)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 175, 8), data_files_55718)
        # Getting the type of the for loop variable (line 175)
        for_loop_var_55719 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 175, 8), data_files_55718)
        # Assigning a type to the variable 'data' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'data', for_loop_var_55719)
        # SSA begins for a for statement (line 175)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 176)
        # Getting the type of 'str' (line 176)
        str_55720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 32), 'str')
        # Getting the type of 'data' (line 176)
        data_55721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 26), 'data')
        
        (may_be_55722, more_types_in_union_55723) = may_be_subtype(str_55720, data_55721)

        if may_be_55722:

            if more_types_in_union_55723:
                # Runtime conditional SSA (line 176)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'data' (line 176)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'data', remove_not_subtype_from_union(data_55721, str))
            
            # Call to append(...): (line 177)
            # Processing the call arguments (line 177)
            # Getting the type of 'data' (line 177)
            data_55726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 38), 'data', False)
            # Processing the call keyword arguments (line 177)
            kwargs_55727 = {}
            # Getting the type of 'new_data_files' (line 177)
            new_data_files_55724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 16), 'new_data_files', False)
            # Obtaining the member 'append' of a type (line 177)
            append_55725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 16), new_data_files_55724, 'append')
            # Calling append(args, kwargs) (line 177)
            append_call_result_55728 = invoke(stypy.reporting.localization.Localization(__file__, 177, 16), append_55725, *[data_55726], **kwargs_55727)
            

            if more_types_in_union_55723:
                # Runtime conditional SSA for else branch (line 176)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_55722) or more_types_in_union_55723):
            # Assigning a type to the variable 'data' (line 176)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'data', remove_subtype_from_union(data_55721, str))
            
            # Type idiom detected: calculating its left and rigth part (line 178)
            # Getting the type of 'tuple' (line 178)
            tuple_55729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 34), 'tuple')
            # Getting the type of 'data' (line 178)
            data_55730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 28), 'data')
            
            (may_be_55731, more_types_in_union_55732) = may_be_subtype(tuple_55729, data_55730)

            if may_be_55731:

                if more_types_in_union_55732:
                    # Runtime conditional SSA (line 178)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'data' (line 178)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 17), 'data', remove_not_subtype_from_union(data_55730, tuple))
                
                # Assigning a Name to a Tuple (line 179):
                
                # Assigning a Subscript to a Name (line 179):
                
                # Obtaining the type of the subscript
                int_55733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 16), 'int')
                # Getting the type of 'data' (line 179)
                data_55734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 27), 'data')
                # Obtaining the member '__getitem__' of a type (line 179)
                getitem___55735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 16), data_55734, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 179)
                subscript_call_result_55736 = invoke(stypy.reporting.localization.Localization(__file__, 179, 16), getitem___55735, int_55733)
                
                # Assigning a type to the variable 'tuple_var_assignment_55285' (line 179)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'tuple_var_assignment_55285', subscript_call_result_55736)
                
                # Assigning a Subscript to a Name (line 179):
                
                # Obtaining the type of the subscript
                int_55737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 16), 'int')
                # Getting the type of 'data' (line 179)
                data_55738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 27), 'data')
                # Obtaining the member '__getitem__' of a type (line 179)
                getitem___55739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 16), data_55738, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 179)
                subscript_call_result_55740 = invoke(stypy.reporting.localization.Localization(__file__, 179, 16), getitem___55739, int_55737)
                
                # Assigning a type to the variable 'tuple_var_assignment_55286' (line 179)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'tuple_var_assignment_55286', subscript_call_result_55740)
                
                # Assigning a Name to a Name (line 179):
                # Getting the type of 'tuple_var_assignment_55285' (line 179)
                tuple_var_assignment_55285_55741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'tuple_var_assignment_55285')
                # Assigning a type to the variable 'd' (line 179)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'd', tuple_var_assignment_55285_55741)
                
                # Assigning a Name to a Name (line 179):
                # Getting the type of 'tuple_var_assignment_55286' (line 179)
                tuple_var_assignment_55286_55742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'tuple_var_assignment_55286')
                # Assigning a type to the variable 'files' (line 179)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 19), 'files', tuple_var_assignment_55286_55742)
                
                # Getting the type of 'self' (line 180)
                self_55743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 19), 'self')
                # Obtaining the member 'inplace' of a type (line 180)
                inplace_55744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 19), self_55743, 'inplace')
                # Testing the type of an if condition (line 180)
                if_condition_55745 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 180, 16), inplace_55744)
                # Assigning a type to the variable 'if_condition_55745' (line 180)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 16), 'if_condition_55745', if_condition_55745)
                # SSA begins for if statement (line 180)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 181):
                
                # Assigning a Call to a Name (line 181):
                
                # Call to get_package_dir(...): (line 181)
                # Processing the call arguments (line 181)
                
                # Call to join(...): (line 181)
                # Processing the call arguments (line 181)
                
                # Call to split(...): (line 181)
                # Processing the call arguments (line 181)
                # Getting the type of 'os' (line 181)
                os_55752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 70), 'os', False)
                # Obtaining the member 'sep' of a type (line 181)
                sep_55753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 70), os_55752, 'sep')
                # Processing the call keyword arguments (line 181)
                kwargs_55754 = {}
                # Getting the type of 'd' (line 181)
                d_55750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 62), 'd', False)
                # Obtaining the member 'split' of a type (line 181)
                split_55751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 62), d_55750, 'split')
                # Calling split(args, kwargs) (line 181)
                split_call_result_55755 = invoke(stypy.reporting.localization.Localization(__file__, 181, 62), split_55751, *[sep_55753], **kwargs_55754)
                
                # Processing the call keyword arguments (line 181)
                kwargs_55756 = {}
                str_55748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 53), 'str', '.')
                # Obtaining the member 'join' of a type (line 181)
                join_55749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 53), str_55748, 'join')
                # Calling join(args, kwargs) (line 181)
                join_call_result_55757 = invoke(stypy.reporting.localization.Localization(__file__, 181, 53), join_55749, *[split_call_result_55755], **kwargs_55756)
                
                # Processing the call keyword arguments (line 181)
                kwargs_55758 = {}
                # Getting the type of 'self' (line 181)
                self_55746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 32), 'self', False)
                # Obtaining the member 'get_package_dir' of a type (line 181)
                get_package_dir_55747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 32), self_55746, 'get_package_dir')
                # Calling get_package_dir(args, kwargs) (line 181)
                get_package_dir_call_result_55759 = invoke(stypy.reporting.localization.Localization(__file__, 181, 32), get_package_dir_55747, *[join_call_result_55757], **kwargs_55758)
                
                # Assigning a type to the variable 'build_dir' (line 181)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 20), 'build_dir', get_package_dir_call_result_55759)
                # SSA branch for the else part of an if statement (line 180)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Name (line 183):
                
                # Assigning a Call to a Name (line 183):
                
                # Call to join(...): (line 183)
                # Processing the call arguments (line 183)
                # Getting the type of 'self' (line 183)
                self_55763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 45), 'self', False)
                # Obtaining the member 'build_src' of a type (line 183)
                build_src_55764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 45), self_55763, 'build_src')
                # Getting the type of 'd' (line 183)
                d_55765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 61), 'd', False)
                # Processing the call keyword arguments (line 183)
                kwargs_55766 = {}
                # Getting the type of 'os' (line 183)
                os_55760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 32), 'os', False)
                # Obtaining the member 'path' of a type (line 183)
                path_55761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 32), os_55760, 'path')
                # Obtaining the member 'join' of a type (line 183)
                join_55762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 32), path_55761, 'join')
                # Calling join(args, kwargs) (line 183)
                join_call_result_55767 = invoke(stypy.reporting.localization.Localization(__file__, 183, 32), join_55762, *[build_src_55764, d_55765], **kwargs_55766)
                
                # Assigning a type to the variable 'build_dir' (line 183)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 20), 'build_dir', join_call_result_55767)
                # SSA join for if statement (line 180)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # Assigning a ListComp to a Name (line 184):
                
                # Assigning a ListComp to a Name (line 184):
                # Calculating list comprehension
                # Calculating comprehension expression
                # Getting the type of 'files' (line 184)
                files_55774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 36), 'files')
                comprehension_55775 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 25), files_55774)
                # Assigning a type to the variable 'f' (line 184)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 25), 'f', comprehension_55775)
                
                # Call to hasattr(...): (line 184)
                # Processing the call arguments (line 184)
                # Getting the type of 'f' (line 184)
                f_55770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 53), 'f', False)
                str_55771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 56), 'str', '__call__')
                # Processing the call keyword arguments (line 184)
                kwargs_55772 = {}
                # Getting the type of 'hasattr' (line 184)
                hasattr_55769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 45), 'hasattr', False)
                # Calling hasattr(args, kwargs) (line 184)
                hasattr_call_result_55773 = invoke(stypy.reporting.localization.Localization(__file__, 184, 45), hasattr_55769, *[f_55770, str_55771], **kwargs_55772)
                
                # Getting the type of 'f' (line 184)
                f_55768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 25), 'f')
                list_55776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 25), 'list')
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 25), list_55776, f_55768)
                # Assigning a type to the variable 'funcs' (line 184)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'funcs', list_55776)
                
                # Assigning a ListComp to a Name (line 185):
                
                # Assigning a ListComp to a Name (line 185):
                # Calculating list comprehension
                # Calculating comprehension expression
                # Getting the type of 'files' (line 185)
                files_55784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 36), 'files')
                comprehension_55785 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 25), files_55784)
                # Assigning a type to the variable 'f' (line 185)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 25), 'f', comprehension_55785)
                
                
                # Call to hasattr(...): (line 185)
                # Processing the call arguments (line 185)
                # Getting the type of 'f' (line 185)
                f_55779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 57), 'f', False)
                str_55780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 60), 'str', '__call__')
                # Processing the call keyword arguments (line 185)
                kwargs_55781 = {}
                # Getting the type of 'hasattr' (line 185)
                hasattr_55778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 49), 'hasattr', False)
                # Calling hasattr(args, kwargs) (line 185)
                hasattr_call_result_55782 = invoke(stypy.reporting.localization.Localization(__file__, 185, 49), hasattr_55778, *[f_55779, str_55780], **kwargs_55781)
                
                # Applying the 'not' unary operator (line 185)
                result_not__55783 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 45), 'not', hasattr_call_result_55782)
                
                # Getting the type of 'f' (line 185)
                f_55777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 25), 'f')
                list_55786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 25), 'list')
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 25), list_55786, f_55777)
                # Assigning a type to the variable 'files' (line 185)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 16), 'files', list_55786)
                
                # Getting the type of 'funcs' (line 186)
                funcs_55787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 25), 'funcs')
                # Testing the type of a for loop iterable (line 186)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 186, 16), funcs_55787)
                # Getting the type of the for loop variable (line 186)
                for_loop_var_55788 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 186, 16), funcs_55787)
                # Assigning a type to the variable 'f' (line 186)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 16), 'f', for_loop_var_55788)
                # SSA begins for a for statement (line 186)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Getting the type of 'f' (line 187)
                f_55789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 23), 'f')
                # Obtaining the member '__code__' of a type (line 187)
                code___55790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 23), f_55789, '__code__')
                # Obtaining the member 'co_argcount' of a type (line 187)
                co_argcount_55791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 23), code___55790, 'co_argcount')
                int_55792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 47), 'int')
                # Applying the binary operator '==' (line 187)
                result_eq_55793 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 23), '==', co_argcount_55791, int_55792)
                
                # Testing the type of an if condition (line 187)
                if_condition_55794 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 187, 20), result_eq_55793)
                # Assigning a type to the variable 'if_condition_55794' (line 187)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 20), 'if_condition_55794', if_condition_55794)
                # SSA begins for if statement (line 187)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 188):
                
                # Assigning a Call to a Name (line 188):
                
                # Call to f(...): (line 188)
                # Processing the call arguments (line 188)
                # Getting the type of 'build_dir' (line 188)
                build_dir_55796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 30), 'build_dir', False)
                # Processing the call keyword arguments (line 188)
                kwargs_55797 = {}
                # Getting the type of 'f' (line 188)
                f_55795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 28), 'f', False)
                # Calling f(args, kwargs) (line 188)
                f_call_result_55798 = invoke(stypy.reporting.localization.Localization(__file__, 188, 28), f_55795, *[build_dir_55796], **kwargs_55797)
                
                # Assigning a type to the variable 's' (line 188)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 24), 's', f_call_result_55798)
                # SSA branch for the else part of an if statement (line 187)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Name (line 190):
                
                # Assigning a Call to a Name (line 190):
                
                # Call to f(...): (line 190)
                # Processing the call keyword arguments (line 190)
                kwargs_55800 = {}
                # Getting the type of 'f' (line 190)
                f_55799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 28), 'f', False)
                # Calling f(args, kwargs) (line 190)
                f_call_result_55801 = invoke(stypy.reporting.localization.Localization(__file__, 190, 28), f_55799, *[], **kwargs_55800)
                
                # Assigning a type to the variable 's' (line 190)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 24), 's', f_call_result_55801)
                # SSA join for if statement (line 187)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # Type idiom detected: calculating its left and rigth part (line 191)
                # Getting the type of 's' (line 191)
                s_55802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 20), 's')
                # Getting the type of 'None' (line 191)
                None_55803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 32), 'None')
                
                (may_be_55804, more_types_in_union_55805) = may_not_be_none(s_55802, None_55803)

                if may_be_55804:

                    if more_types_in_union_55805:
                        # Runtime conditional SSA (line 191)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    
                    # Type idiom detected: calculating its left and rigth part (line 192)
                    # Getting the type of 'list' (line 192)
                    list_55806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 41), 'list')
                    # Getting the type of 's' (line 192)
                    s_55807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 38), 's')
                    
                    (may_be_55808, more_types_in_union_55809) = may_be_subtype(list_55806, s_55807)

                    if may_be_55808:

                        if more_types_in_union_55809:
                            # Runtime conditional SSA (line 192)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                        else:
                            module_type_store = module_type_store

                        # Assigning a type to the variable 's' (line 192)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 24), 's', remove_not_subtype_from_union(s_55807, list))
                        
                        # Call to extend(...): (line 193)
                        # Processing the call arguments (line 193)
                        # Getting the type of 's' (line 193)
                        s_55812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 41), 's', False)
                        # Processing the call keyword arguments (line 193)
                        kwargs_55813 = {}
                        # Getting the type of 'files' (line 193)
                        files_55810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 28), 'files', False)
                        # Obtaining the member 'extend' of a type (line 193)
                        extend_55811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 28), files_55810, 'extend')
                        # Calling extend(args, kwargs) (line 193)
                        extend_call_result_55814 = invoke(stypy.reporting.localization.Localization(__file__, 193, 28), extend_55811, *[s_55812], **kwargs_55813)
                        

                        if more_types_in_union_55809:
                            # Runtime conditional SSA for else branch (line 192)
                            module_type_store.open_ssa_branch('idiom else')



                    if ((not may_be_55808) or more_types_in_union_55809):
                        # Assigning a type to the variable 's' (line 192)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 24), 's', remove_subtype_from_union(s_55807, list))
                        
                        # Type idiom detected: calculating its left and rigth part (line 194)
                        # Getting the type of 'str' (line 194)
                        str_55815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 43), 'str')
                        # Getting the type of 's' (line 194)
                        s_55816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 40), 's')
                        
                        (may_be_55817, more_types_in_union_55818) = may_be_subtype(str_55815, s_55816)

                        if may_be_55817:

                            if more_types_in_union_55818:
                                # Runtime conditional SSA (line 194)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                            else:
                                module_type_store = module_type_store

                            # Assigning a type to the variable 's' (line 194)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 29), 's', remove_not_subtype_from_union(s_55816, str))
                            
                            # Call to append(...): (line 195)
                            # Processing the call arguments (line 195)
                            # Getting the type of 's' (line 195)
                            s_55821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 41), 's', False)
                            # Processing the call keyword arguments (line 195)
                            kwargs_55822 = {}
                            # Getting the type of 'files' (line 195)
                            files_55819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 28), 'files', False)
                            # Obtaining the member 'append' of a type (line 195)
                            append_55820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 28), files_55819, 'append')
                            # Calling append(args, kwargs) (line 195)
                            append_call_result_55823 = invoke(stypy.reporting.localization.Localization(__file__, 195, 28), append_55820, *[s_55821], **kwargs_55822)
                            

                            if more_types_in_union_55818:
                                # Runtime conditional SSA for else branch (line 194)
                                module_type_store.open_ssa_branch('idiom else')



                        if ((not may_be_55817) or more_types_in_union_55818):
                            # Assigning a type to the variable 's' (line 194)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 29), 's', remove_subtype_from_union(s_55816, str))
                            
                            # Call to TypeError(...): (line 197)
                            # Processing the call arguments (line 197)
                            
                            # Call to repr(...): (line 197)
                            # Processing the call arguments (line 197)
                            # Getting the type of 's' (line 197)
                            s_55826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 49), 's', False)
                            # Processing the call keyword arguments (line 197)
                            kwargs_55827 = {}
                            # Getting the type of 'repr' (line 197)
                            repr_55825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 44), 'repr', False)
                            # Calling repr(args, kwargs) (line 197)
                            repr_call_result_55828 = invoke(stypy.reporting.localization.Localization(__file__, 197, 44), repr_55825, *[s_55826], **kwargs_55827)
                            
                            # Processing the call keyword arguments (line 197)
                            kwargs_55829 = {}
                            # Getting the type of 'TypeError' (line 197)
                            TypeError_55824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 34), 'TypeError', False)
                            # Calling TypeError(args, kwargs) (line 197)
                            TypeError_call_result_55830 = invoke(stypy.reporting.localization.Localization(__file__, 197, 34), TypeError_55824, *[repr_call_result_55828], **kwargs_55829)
                            
                            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 197, 28), TypeError_call_result_55830, 'raise parameter', BaseException)

                            if (may_be_55817 and more_types_in_union_55818):
                                # SSA join for if statement (line 194)
                                module_type_store = module_type_store.join_ssa_context()


                        

                        if (may_be_55808 and more_types_in_union_55809):
                            # SSA join for if statement (line 192)
                            module_type_store = module_type_store.join_ssa_context()


                    

                    if more_types_in_union_55805:
                        # SSA join for if statement (line 191)
                        module_type_store = module_type_store.join_ssa_context()


                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()
                
                
                # Assigning a Call to a Name (line 198):
                
                # Assigning a Call to a Name (line 198):
                
                # Call to get_data_files(...): (line 198)
                # Processing the call arguments (line 198)
                
                # Obtaining an instance of the builtin type 'tuple' (line 198)
                tuple_55832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 44), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 198)
                # Adding element type (line 198)
                # Getting the type of 'd' (line 198)
                d_55833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 44), 'd', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 44), tuple_55832, d_55833)
                # Adding element type (line 198)
                # Getting the type of 'files' (line 198)
                files_55834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 47), 'files', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 44), tuple_55832, files_55834)
                
                # Processing the call keyword arguments (line 198)
                kwargs_55835 = {}
                # Getting the type of 'get_data_files' (line 198)
                get_data_files_55831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 28), 'get_data_files', False)
                # Calling get_data_files(args, kwargs) (line 198)
                get_data_files_call_result_55836 = invoke(stypy.reporting.localization.Localization(__file__, 198, 28), get_data_files_55831, *[tuple_55832], **kwargs_55835)
                
                # Assigning a type to the variable 'filenames' (line 198)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'filenames', get_data_files_call_result_55836)
                
                # Call to append(...): (line 199)
                # Processing the call arguments (line 199)
                
                # Obtaining an instance of the builtin type 'tuple' (line 199)
                tuple_55839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 39), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 199)
                # Adding element type (line 199)
                # Getting the type of 'd' (line 199)
                d_55840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 39), 'd', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 39), tuple_55839, d_55840)
                # Adding element type (line 199)
                # Getting the type of 'filenames' (line 199)
                filenames_55841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 42), 'filenames', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 39), tuple_55839, filenames_55841)
                
                # Processing the call keyword arguments (line 199)
                kwargs_55842 = {}
                # Getting the type of 'new_data_files' (line 199)
                new_data_files_55837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'new_data_files', False)
                # Obtaining the member 'append' of a type (line 199)
                append_55838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 16), new_data_files_55837, 'append')
                # Calling append(args, kwargs) (line 199)
                append_call_result_55843 = invoke(stypy.reporting.localization.Localization(__file__, 199, 16), append_55838, *[tuple_55839], **kwargs_55842)
                

                if more_types_in_union_55732:
                    # Runtime conditional SSA for else branch (line 178)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_55731) or more_types_in_union_55732):
                # Assigning a type to the variable 'data' (line 178)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 17), 'data', remove_subtype_from_union(data_55730, tuple))
                
                # Call to TypeError(...): (line 201)
                # Processing the call arguments (line 201)
                
                # Call to repr(...): (line 201)
                # Processing the call arguments (line 201)
                # Getting the type of 'data' (line 201)
                data_55846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 37), 'data', False)
                # Processing the call keyword arguments (line 201)
                kwargs_55847 = {}
                # Getting the type of 'repr' (line 201)
                repr_55845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 32), 'repr', False)
                # Calling repr(args, kwargs) (line 201)
                repr_call_result_55848 = invoke(stypy.reporting.localization.Localization(__file__, 201, 32), repr_55845, *[data_55846], **kwargs_55847)
                
                # Processing the call keyword arguments (line 201)
                kwargs_55849 = {}
                # Getting the type of 'TypeError' (line 201)
                TypeError_55844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 22), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 201)
                TypeError_call_result_55850 = invoke(stypy.reporting.localization.Localization(__file__, 201, 22), TypeError_55844, *[repr_call_result_55848], **kwargs_55849)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 201, 16), TypeError_call_result_55850, 'raise parameter', BaseException)

                if (may_be_55731 and more_types_in_union_55732):
                    # SSA join for if statement (line 178)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_55722 and more_types_in_union_55723):
                # SSA join for if statement (line 176)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 202):
        
        # Assigning a Name to a Subscript (line 202):
        # Getting the type of 'new_data_files' (line 202)
        new_data_files_55851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 29), 'new_data_files')
        # Getting the type of 'self' (line 202)
        self_55852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'self')
        # Obtaining the member 'data_files' of a type (line 202)
        data_files_55853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), self_55852, 'data_files')
        slice_55854 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 202, 8), None, None, None)
        # Storing an element on a container (line 202)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 8), data_files_55853, (slice_55854, new_data_files_55851))
        
        # ################# End of 'build_data_files_sources(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'build_data_files_sources' in the type store
        # Getting the type of 'stypy_return_type' (line 169)
        stypy_return_type_55855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_55855)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'build_data_files_sources'
        return stypy_return_type_55855


    @norecursion
    def _build_npy_pkg_config(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_build_npy_pkg_config'
        module_type_store = module_type_store.open_function_context('_build_npy_pkg_config', 205, 4, False)
        # Assigning a type to the variable 'self' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_src._build_npy_pkg_config.__dict__.__setitem__('stypy_localization', localization)
        build_src._build_npy_pkg_config.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_src._build_npy_pkg_config.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_src._build_npy_pkg_config.__dict__.__setitem__('stypy_function_name', 'build_src._build_npy_pkg_config')
        build_src._build_npy_pkg_config.__dict__.__setitem__('stypy_param_names_list', ['info', 'gd'])
        build_src._build_npy_pkg_config.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_src._build_npy_pkg_config.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_src._build_npy_pkg_config.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_src._build_npy_pkg_config.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_src._build_npy_pkg_config.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_src._build_npy_pkg_config.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_src._build_npy_pkg_config', ['info', 'gd'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_build_npy_pkg_config', localization, ['info', 'gd'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_build_npy_pkg_config(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 206, 8))
        
        # 'import shutil' statement (line 206)
        import shutil

        import_module(stypy.reporting.localization.Localization(__file__, 206, 8), 'shutil', shutil, module_type_store)
        
        
        # Assigning a Name to a Tuple (line 207):
        
        # Assigning a Subscript to a Name (line 207):
        
        # Obtaining the type of the subscript
        int_55856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 8), 'int')
        # Getting the type of 'info' (line 207)
        info_55857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 44), 'info')
        # Obtaining the member '__getitem__' of a type (line 207)
        getitem___55858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), info_55857, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 207)
        subscript_call_result_55859 = invoke(stypy.reporting.localization.Localization(__file__, 207, 8), getitem___55858, int_55856)
        
        # Assigning a type to the variable 'tuple_var_assignment_55287' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'tuple_var_assignment_55287', subscript_call_result_55859)
        
        # Assigning a Subscript to a Name (line 207):
        
        # Obtaining the type of the subscript
        int_55860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 8), 'int')
        # Getting the type of 'info' (line 207)
        info_55861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 44), 'info')
        # Obtaining the member '__getitem__' of a type (line 207)
        getitem___55862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), info_55861, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 207)
        subscript_call_result_55863 = invoke(stypy.reporting.localization.Localization(__file__, 207, 8), getitem___55862, int_55860)
        
        # Assigning a type to the variable 'tuple_var_assignment_55288' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'tuple_var_assignment_55288', subscript_call_result_55863)
        
        # Assigning a Subscript to a Name (line 207):
        
        # Obtaining the type of the subscript
        int_55864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 8), 'int')
        # Getting the type of 'info' (line 207)
        info_55865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 44), 'info')
        # Obtaining the member '__getitem__' of a type (line 207)
        getitem___55866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), info_55865, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 207)
        subscript_call_result_55867 = invoke(stypy.reporting.localization.Localization(__file__, 207, 8), getitem___55866, int_55864)
        
        # Assigning a type to the variable 'tuple_var_assignment_55289' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'tuple_var_assignment_55289', subscript_call_result_55867)
        
        # Assigning a Name to a Name (line 207):
        # Getting the type of 'tuple_var_assignment_55287' (line 207)
        tuple_var_assignment_55287_55868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'tuple_var_assignment_55287')
        # Assigning a type to the variable 'template' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'template', tuple_var_assignment_55287_55868)
        
        # Assigning a Name to a Name (line 207):
        # Getting the type of 'tuple_var_assignment_55288' (line 207)
        tuple_var_assignment_55288_55869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'tuple_var_assignment_55288')
        # Assigning a type to the variable 'install_dir' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 18), 'install_dir', tuple_var_assignment_55288_55869)
        
        # Assigning a Name to a Name (line 207):
        # Getting the type of 'tuple_var_assignment_55289' (line 207)
        tuple_var_assignment_55289_55870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'tuple_var_assignment_55289')
        # Assigning a type to the variable 'subst_dict' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 31), 'subst_dict', tuple_var_assignment_55289_55870)
        
        # Assigning a Call to a Name (line 208):
        
        # Assigning a Call to a Name (line 208):
        
        # Call to dirname(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'template' (line 208)
        template_55874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 39), 'template', False)
        # Processing the call keyword arguments (line 208)
        kwargs_55875 = {}
        # Getting the type of 'os' (line 208)
        os_55871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 208)
        path_55872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 23), os_55871, 'path')
        # Obtaining the member 'dirname' of a type (line 208)
        dirname_55873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 23), path_55872, 'dirname')
        # Calling dirname(args, kwargs) (line 208)
        dirname_call_result_55876 = invoke(stypy.reporting.localization.Localization(__file__, 208, 23), dirname_55873, *[template_55874], **kwargs_55875)
        
        # Assigning a type to the variable 'template_dir' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'template_dir', dirname_call_result_55876)
        
        
        # Call to items(...): (line 209)
        # Processing the call keyword arguments (line 209)
        kwargs_55879 = {}
        # Getting the type of 'gd' (line 209)
        gd_55877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 20), 'gd', False)
        # Obtaining the member 'items' of a type (line 209)
        items_55878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 20), gd_55877, 'items')
        # Calling items(args, kwargs) (line 209)
        items_call_result_55880 = invoke(stypy.reporting.localization.Localization(__file__, 209, 20), items_55878, *[], **kwargs_55879)
        
        # Testing the type of a for loop iterable (line 209)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 209, 8), items_call_result_55880)
        # Getting the type of the for loop variable (line 209)
        for_loop_var_55881 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 209, 8), items_call_result_55880)
        # Assigning a type to the variable 'k' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 8), for_loop_var_55881))
        # Assigning a type to the variable 'v' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 8), for_loop_var_55881))
        # SSA begins for a for statement (line 209)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Subscript (line 210):
        
        # Assigning a Name to a Subscript (line 210):
        # Getting the type of 'v' (line 210)
        v_55882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 28), 'v')
        # Getting the type of 'subst_dict' (line 210)
        subst_dict_55883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'subst_dict')
        # Getting the type of 'k' (line 210)
        k_55884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 23), 'k')
        # Storing an element on a container (line 210)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 12), subst_dict_55883, (k_55884, v_55882))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 212)
        self_55885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 11), 'self')
        # Obtaining the member 'inplace' of a type (line 212)
        inplace_55886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 11), self_55885, 'inplace')
        int_55887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 27), 'int')
        # Applying the binary operator '==' (line 212)
        result_eq_55888 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 11), '==', inplace_55886, int_55887)
        
        # Testing the type of an if condition (line 212)
        if_condition_55889 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 212, 8), result_eq_55888)
        # Assigning a type to the variable 'if_condition_55889' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'if_condition_55889', if_condition_55889)
        # SSA begins for if statement (line 212)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 213):
        
        # Assigning a Call to a Name (line 213):
        
        # Call to join(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'template_dir' (line 213)
        template_dir_55893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 41), 'template_dir', False)
        # Getting the type of 'install_dir' (line 213)
        install_dir_55894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 55), 'install_dir', False)
        # Processing the call keyword arguments (line 213)
        kwargs_55895 = {}
        # Getting the type of 'os' (line 213)
        os_55890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 28), 'os', False)
        # Obtaining the member 'path' of a type (line 213)
        path_55891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 28), os_55890, 'path')
        # Obtaining the member 'join' of a type (line 213)
        join_55892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 28), path_55891, 'join')
        # Calling join(args, kwargs) (line 213)
        join_call_result_55896 = invoke(stypy.reporting.localization.Localization(__file__, 213, 28), join_55892, *[template_dir_55893, install_dir_55894], **kwargs_55895)
        
        # Assigning a type to the variable 'generated_dir' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'generated_dir', join_call_result_55896)
        # SSA branch for the else part of an if statement (line 212)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 215):
        
        # Assigning a Call to a Name (line 215):
        
        # Call to join(...): (line 215)
        # Processing the call arguments (line 215)
        # Getting the type of 'self' (line 215)
        self_55900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 41), 'self', False)
        # Obtaining the member 'build_src' of a type (line 215)
        build_src_55901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 41), self_55900, 'build_src')
        # Getting the type of 'template_dir' (line 215)
        template_dir_55902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 57), 'template_dir', False)
        # Getting the type of 'install_dir' (line 216)
        install_dir_55903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 20), 'install_dir', False)
        # Processing the call keyword arguments (line 215)
        kwargs_55904 = {}
        # Getting the type of 'os' (line 215)
        os_55897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 28), 'os', False)
        # Obtaining the member 'path' of a type (line 215)
        path_55898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 28), os_55897, 'path')
        # Obtaining the member 'join' of a type (line 215)
        join_55899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 28), path_55898, 'join')
        # Calling join(args, kwargs) (line 215)
        join_call_result_55905 = invoke(stypy.reporting.localization.Localization(__file__, 215, 28), join_55899, *[build_src_55901, template_dir_55902, install_dir_55903], **kwargs_55904)
        
        # Assigning a type to the variable 'generated_dir' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'generated_dir', join_call_result_55905)
        # SSA join for if statement (line 212)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 217):
        
        # Assigning a Call to a Name (line 217):
        
        # Call to basename(...): (line 217)
        # Processing the call arguments (line 217)
        
        # Obtaining the type of the subscript
        int_55909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 64), 'int')
        
        # Call to splitext(...): (line 217)
        # Processing the call arguments (line 217)
        # Getting the type of 'template' (line 217)
        template_55913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 54), 'template', False)
        # Processing the call keyword arguments (line 217)
        kwargs_55914 = {}
        # Getting the type of 'os' (line 217)
        os_55910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 37), 'os', False)
        # Obtaining the member 'path' of a type (line 217)
        path_55911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 37), os_55910, 'path')
        # Obtaining the member 'splitext' of a type (line 217)
        splitext_55912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 37), path_55911, 'splitext')
        # Calling splitext(args, kwargs) (line 217)
        splitext_call_result_55915 = invoke(stypy.reporting.localization.Localization(__file__, 217, 37), splitext_55912, *[template_55913], **kwargs_55914)
        
        # Obtaining the member '__getitem__' of a type (line 217)
        getitem___55916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 37), splitext_call_result_55915, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 217)
        subscript_call_result_55917 = invoke(stypy.reporting.localization.Localization(__file__, 217, 37), getitem___55916, int_55909)
        
        # Processing the call keyword arguments (line 217)
        kwargs_55918 = {}
        # Getting the type of 'os' (line 217)
        os_55906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 217)
        path_55907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 20), os_55906, 'path')
        # Obtaining the member 'basename' of a type (line 217)
        basename_55908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 20), path_55907, 'basename')
        # Calling basename(args, kwargs) (line 217)
        basename_call_result_55919 = invoke(stypy.reporting.localization.Localization(__file__, 217, 20), basename_55908, *[subscript_call_result_55917], **kwargs_55918)
        
        # Assigning a type to the variable 'generated' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'generated', basename_call_result_55919)
        
        # Assigning a Call to a Name (line 218):
        
        # Assigning a Call to a Name (line 218):
        
        # Call to join(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'generated_dir' (line 218)
        generated_dir_55923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 38), 'generated_dir', False)
        # Getting the type of 'generated' (line 218)
        generated_55924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 53), 'generated', False)
        # Processing the call keyword arguments (line 218)
        kwargs_55925 = {}
        # Getting the type of 'os' (line 218)
        os_55920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 218)
        path_55921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 25), os_55920, 'path')
        # Obtaining the member 'join' of a type (line 218)
        join_55922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 25), path_55921, 'join')
        # Calling join(args, kwargs) (line 218)
        join_call_result_55926 = invoke(stypy.reporting.localization.Localization(__file__, 218, 25), join_55922, *[generated_dir_55923, generated_55924], **kwargs_55925)
        
        # Assigning a type to the variable 'generated_path' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'generated_path', join_call_result_55926)
        
        
        
        # Call to exists(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'generated_dir' (line 219)
        generated_dir_55930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 30), 'generated_dir', False)
        # Processing the call keyword arguments (line 219)
        kwargs_55931 = {}
        # Getting the type of 'os' (line 219)
        os_55927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 219)
        path_55928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 15), os_55927, 'path')
        # Obtaining the member 'exists' of a type (line 219)
        exists_55929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 15), path_55928, 'exists')
        # Calling exists(args, kwargs) (line 219)
        exists_call_result_55932 = invoke(stypy.reporting.localization.Localization(__file__, 219, 15), exists_55929, *[generated_dir_55930], **kwargs_55931)
        
        # Applying the 'not' unary operator (line 219)
        result_not__55933 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 11), 'not', exists_call_result_55932)
        
        # Testing the type of an if condition (line 219)
        if_condition_55934 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 219, 8), result_not__55933)
        # Assigning a type to the variable 'if_condition_55934' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'if_condition_55934', if_condition_55934)
        # SSA begins for if statement (line 219)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to makedirs(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'generated_dir' (line 220)
        generated_dir_55937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 24), 'generated_dir', False)
        # Processing the call keyword arguments (line 220)
        kwargs_55938 = {}
        # Getting the type of 'os' (line 220)
        os_55935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'os', False)
        # Obtaining the member 'makedirs' of a type (line 220)
        makedirs_55936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 12), os_55935, 'makedirs')
        # Calling makedirs(args, kwargs) (line 220)
        makedirs_call_result_55939 = invoke(stypy.reporting.localization.Localization(__file__, 220, 12), makedirs_55936, *[generated_dir_55937], **kwargs_55938)
        
        # SSA join for if statement (line 219)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to subst_vars(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'generated_path' (line 222)
        generated_path_55941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 19), 'generated_path', False)
        # Getting the type of 'template' (line 222)
        template_55942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 35), 'template', False)
        # Getting the type of 'subst_dict' (line 222)
        subst_dict_55943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 45), 'subst_dict', False)
        # Processing the call keyword arguments (line 222)
        kwargs_55944 = {}
        # Getting the type of 'subst_vars' (line 222)
        subst_vars_55940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'subst_vars', False)
        # Calling subst_vars(args, kwargs) (line 222)
        subst_vars_call_result_55945 = invoke(stypy.reporting.localization.Localization(__file__, 222, 8), subst_vars_55940, *[generated_path_55941, template_55942, subst_dict_55943], **kwargs_55944)
        
        
        # Assigning a Call to a Name (line 225):
        
        # Assigning a Call to a Name (line 225):
        
        # Call to join(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of 'template_dir' (line 225)
        template_dir_55949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 40), 'template_dir', False)
        # Getting the type of 'install_dir' (line 225)
        install_dir_55950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 54), 'install_dir', False)
        # Processing the call keyword arguments (line 225)
        kwargs_55951 = {}
        # Getting the type of 'os' (line 225)
        os_55946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 225)
        path_55947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 27), os_55946, 'path')
        # Obtaining the member 'join' of a type (line 225)
        join_55948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 27), path_55947, 'join')
        # Calling join(args, kwargs) (line 225)
        join_call_result_55952 = invoke(stypy.reporting.localization.Localization(__file__, 225, 27), join_55948, *[template_dir_55949, install_dir_55950], **kwargs_55951)
        
        # Assigning a type to the variable 'full_install_dir' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'full_install_dir', join_call_result_55952)
        
        # Obtaining an instance of the builtin type 'tuple' (line 226)
        tuple_55953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 226)
        # Adding element type (line 226)
        # Getting the type of 'full_install_dir' (line 226)
        full_install_dir_55954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 15), 'full_install_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 15), tuple_55953, full_install_dir_55954)
        # Adding element type (line 226)
        # Getting the type of 'generated_path' (line 226)
        generated_path_55955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 33), 'generated_path')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 15), tuple_55953, generated_path_55955)
        
        # Assigning a type to the variable 'stypy_return_type' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'stypy_return_type', tuple_55953)
        
        # ################# End of '_build_npy_pkg_config(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_build_npy_pkg_config' in the type store
        # Getting the type of 'stypy_return_type' (line 205)
        stypy_return_type_55956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_55956)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_build_npy_pkg_config'
        return stypy_return_type_55956


    @norecursion
    def build_npy_pkg_config(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'build_npy_pkg_config'
        module_type_store = module_type_store.open_function_context('build_npy_pkg_config', 228, 4, False)
        # Assigning a type to the variable 'self' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_src.build_npy_pkg_config.__dict__.__setitem__('stypy_localization', localization)
        build_src.build_npy_pkg_config.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_src.build_npy_pkg_config.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_src.build_npy_pkg_config.__dict__.__setitem__('stypy_function_name', 'build_src.build_npy_pkg_config')
        build_src.build_npy_pkg_config.__dict__.__setitem__('stypy_param_names_list', [])
        build_src.build_npy_pkg_config.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_src.build_npy_pkg_config.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_src.build_npy_pkg_config.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_src.build_npy_pkg_config.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_src.build_npy_pkg_config.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_src.build_npy_pkg_config.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_src.build_npy_pkg_config', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'build_npy_pkg_config', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'build_npy_pkg_config(...)' code ##################

        
        # Call to info(...): (line 229)
        # Processing the call arguments (line 229)
        str_55959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 17), 'str', 'build_src: building npy-pkg config files')
        # Processing the call keyword arguments (line 229)
        kwargs_55960 = {}
        # Getting the type of 'log' (line 229)
        log_55957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 229)
        info_55958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 8), log_55957, 'info')
        # Calling info(args, kwargs) (line 229)
        info_call_result_55961 = invoke(stypy.reporting.localization.Localization(__file__, 229, 8), info_55958, *[str_55959], **kwargs_55960)
        
        
        # Assigning a Call to a Name (line 237):
        
        # Assigning a Call to a Name (line 237):
        
        # Call to copy(...): (line 237)
        # Processing the call arguments (line 237)
        
        # Call to get_cmd(...): (line 237)
        # Processing the call arguments (line 237)
        str_55965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 40), 'str', 'install')
        # Processing the call keyword arguments (line 237)
        kwargs_55966 = {}
        # Getting the type of 'get_cmd' (line 237)
        get_cmd_55964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 32), 'get_cmd', False)
        # Calling get_cmd(args, kwargs) (line 237)
        get_cmd_call_result_55967 = invoke(stypy.reporting.localization.Localization(__file__, 237, 32), get_cmd_55964, *[str_55965], **kwargs_55966)
        
        # Processing the call keyword arguments (line 237)
        kwargs_55968 = {}
        # Getting the type of 'copy' (line 237)
        copy_55962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 22), 'copy', False)
        # Obtaining the member 'copy' of a type (line 237)
        copy_55963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 22), copy_55962, 'copy')
        # Calling copy(args, kwargs) (line 237)
        copy_call_result_55969 = invoke(stypy.reporting.localization.Localization(__file__, 237, 22), copy_55963, *[get_cmd_call_result_55967], **kwargs_55968)
        
        # Assigning a type to the variable 'install_cmd' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'install_cmd', copy_call_result_55969)
        
        
        
        # Getting the type of 'install_cmd' (line 238)
        install_cmd_55970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 15), 'install_cmd')
        # Obtaining the member 'finalized' of a type (line 238)
        finalized_55971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 15), install_cmd_55970, 'finalized')
        int_55972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 40), 'int')
        # Applying the binary operator '==' (line 238)
        result_eq_55973 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 15), '==', finalized_55971, int_55972)
        
        # Applying the 'not' unary operator (line 238)
        result_not__55974 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 11), 'not', result_eq_55973)
        
        # Testing the type of an if condition (line 238)
        if_condition_55975 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 238, 8), result_not__55974)
        # Assigning a type to the variable 'if_condition_55975' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'if_condition_55975', if_condition_55975)
        # SSA begins for if statement (line 238)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to finalize_options(...): (line 239)
        # Processing the call keyword arguments (line 239)
        kwargs_55978 = {}
        # Getting the type of 'install_cmd' (line 239)
        install_cmd_55976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'install_cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 239)
        finalize_options_55977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 12), install_cmd_55976, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 239)
        finalize_options_call_result_55979 = invoke(stypy.reporting.localization.Localization(__file__, 239, 12), finalize_options_55977, *[], **kwargs_55978)
        
        # SSA join for if statement (line 238)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 240):
        
        # Assigning a Name to a Name (line 240):
        # Getting the type of 'False' (line 240)
        False_55980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 21), 'False')
        # Assigning a type to the variable 'build_npkg' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'build_npkg', False_55980)
        
        # Assigning a Dict to a Name (line 241):
        
        # Assigning a Dict to a Name (line 241):
        
        # Obtaining an instance of the builtin type 'dict' (line 241)
        dict_55981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 13), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 241)
        
        # Assigning a type to the variable 'gd' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'gd', dict_55981)
        
        
        # Getting the type of 'self' (line 242)
        self_55982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 11), 'self')
        # Obtaining the member 'inplace' of a type (line 242)
        inplace_55983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 11), self_55982, 'inplace')
        int_55984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 27), 'int')
        # Applying the binary operator '==' (line 242)
        result_eq_55985 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 11), '==', inplace_55983, int_55984)
        
        # Testing the type of an if condition (line 242)
        if_condition_55986 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 242, 8), result_eq_55985)
        # Assigning a type to the variable 'if_condition_55986' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'if_condition_55986', if_condition_55986)
        # SSA begins for if statement (line 242)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 243):
        
        # Assigning a Str to a Name (line 243):
        str_55987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 25), 'str', '.')
        # Assigning a type to the variable 'top_prefix' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'top_prefix', str_55987)
        
        # Assigning a Name to a Name (line 244):
        
        # Assigning a Name to a Name (line 244):
        # Getting the type of 'True' (line 244)
        True_55988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 25), 'True')
        # Assigning a type to the variable 'build_npkg' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'build_npkg', True_55988)
        # SSA branch for the else part of an if statement (line 242)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 245)
        str_55989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 34), 'str', 'install_libbase')
        # Getting the type of 'install_cmd' (line 245)
        install_cmd_55990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 21), 'install_cmd')
        
        (may_be_55991, more_types_in_union_55992) = may_provide_member(str_55989, install_cmd_55990)

        if may_be_55991:

            if more_types_in_union_55992:
                # Runtime conditional SSA (line 245)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'install_cmd' (line 245)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 13), 'install_cmd', remove_not_member_provider_from_union(install_cmd_55990, 'install_libbase'))
            
            # Assigning a Attribute to a Name (line 246):
            
            # Assigning a Attribute to a Name (line 246):
            # Getting the type of 'install_cmd' (line 246)
            install_cmd_55993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 25), 'install_cmd')
            # Obtaining the member 'install_libbase' of a type (line 246)
            install_libbase_55994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 25), install_cmd_55993, 'install_libbase')
            # Assigning a type to the variable 'top_prefix' (line 246)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'top_prefix', install_libbase_55994)
            
            # Assigning a Name to a Name (line 247):
            
            # Assigning a Name to a Name (line 247):
            # Getting the type of 'True' (line 247)
            True_55995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 25), 'True')
            # Assigning a type to the variable 'build_npkg' (line 247)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'build_npkg', True_55995)

            if more_types_in_union_55992:
                # SSA join for if statement (line 245)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 242)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'build_npkg' (line 249)
        build_npkg_55996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 11), 'build_npkg')
        # Testing the type of an if condition (line 249)
        if_condition_55997 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 249, 8), build_npkg_55996)
        # Assigning a type to the variable 'if_condition_55997' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'if_condition_55997', if_condition_55997)
        # SSA begins for if statement (line 249)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to items(...): (line 250)
        # Processing the call keyword arguments (line 250)
        kwargs_56002 = {}
        # Getting the type of 'self' (line 250)
        self_55998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 30), 'self', False)
        # Obtaining the member 'distribution' of a type (line 250)
        distribution_55999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 30), self_55998, 'distribution')
        # Obtaining the member 'installed_pkg_config' of a type (line 250)
        installed_pkg_config_56000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 30), distribution_55999, 'installed_pkg_config')
        # Obtaining the member 'items' of a type (line 250)
        items_56001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 30), installed_pkg_config_56000, 'items')
        # Calling items(args, kwargs) (line 250)
        items_call_result_56003 = invoke(stypy.reporting.localization.Localization(__file__, 250, 30), items_56001, *[], **kwargs_56002)
        
        # Testing the type of a for loop iterable (line 250)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 250, 12), items_call_result_56003)
        # Getting the type of the for loop variable (line 250)
        for_loop_var_56004 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 250, 12), items_call_result_56003)
        # Assigning a type to the variable 'pkg' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'pkg', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 12), for_loop_var_56004))
        # Assigning a type to the variable 'infos' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'infos', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 12), for_loop_var_56004))
        # SSA begins for a for statement (line 250)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 251):
        
        # Assigning a Subscript to a Name (line 251):
        
        # Obtaining the type of the subscript
        # Getting the type of 'pkg' (line 251)
        pkg_56005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 57), 'pkg')
        # Getting the type of 'self' (line 251)
        self_56006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 27), 'self')
        # Obtaining the member 'distribution' of a type (line 251)
        distribution_56007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 27), self_56006, 'distribution')
        # Obtaining the member 'package_dir' of a type (line 251)
        package_dir_56008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 27), distribution_56007, 'package_dir')
        # Obtaining the member '__getitem__' of a type (line 251)
        getitem___56009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 27), package_dir_56008, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 251)
        subscript_call_result_56010 = invoke(stypy.reporting.localization.Localization(__file__, 251, 27), getitem___56009, pkg_56005)
        
        # Assigning a type to the variable 'pkg_path' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 16), 'pkg_path', subscript_call_result_56010)
        
        # Assigning a Call to a Name (line 252):
        
        # Assigning a Call to a Name (line 252):
        
        # Call to join(...): (line 252)
        # Processing the call arguments (line 252)
        
        # Call to abspath(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'top_prefix' (line 252)
        top_prefix_56017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 54), 'top_prefix', False)
        # Processing the call keyword arguments (line 252)
        kwargs_56018 = {}
        # Getting the type of 'os' (line 252)
        os_56014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 38), 'os', False)
        # Obtaining the member 'path' of a type (line 252)
        path_56015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 38), os_56014, 'path')
        # Obtaining the member 'abspath' of a type (line 252)
        abspath_56016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 38), path_56015, 'abspath')
        # Calling abspath(args, kwargs) (line 252)
        abspath_call_result_56019 = invoke(stypy.reporting.localization.Localization(__file__, 252, 38), abspath_56016, *[top_prefix_56017], **kwargs_56018)
        
        # Getting the type of 'pkg_path' (line 252)
        pkg_path_56020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 67), 'pkg_path', False)
        # Processing the call keyword arguments (line 252)
        kwargs_56021 = {}
        # Getting the type of 'os' (line 252)
        os_56011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 252)
        path_56012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 25), os_56011, 'path')
        # Obtaining the member 'join' of a type (line 252)
        join_56013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 25), path_56012, 'join')
        # Calling join(args, kwargs) (line 252)
        join_call_result_56022 = invoke(stypy.reporting.localization.Localization(__file__, 252, 25), join_56013, *[abspath_call_result_56019, pkg_path_56020], **kwargs_56021)
        
        # Assigning a type to the variable 'prefix' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 16), 'prefix', join_call_result_56022)
        
        # Assigning a Dict to a Name (line 253):
        
        # Assigning a Dict to a Name (line 253):
        
        # Obtaining an instance of the builtin type 'dict' (line 253)
        dict_56023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 20), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 253)
        # Adding element type (key, value) (line 253)
        str_56024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 21), 'str', 'prefix')
        # Getting the type of 'prefix' (line 253)
        prefix_56025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 31), 'prefix')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 20), dict_56023, (str_56024, prefix_56025))
        
        # Assigning a type to the variable 'd' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 16), 'd', dict_56023)
        
        # Getting the type of 'infos' (line 254)
        infos_56026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 28), 'infos')
        # Testing the type of a for loop iterable (line 254)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 254, 16), infos_56026)
        # Getting the type of the for loop variable (line 254)
        for_loop_var_56027 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 254, 16), infos_56026)
        # Assigning a type to the variable 'info' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 16), 'info', for_loop_var_56027)
        # SSA begins for a for statement (line 254)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 255):
        
        # Assigning a Call to a Name:
        
        # Call to _build_npy_pkg_config(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'info' (line 255)
        info_56030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 72), 'info', False)
        # Getting the type of 'd' (line 255)
        d_56031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 78), 'd', False)
        # Processing the call keyword arguments (line 255)
        kwargs_56032 = {}
        # Getting the type of 'self' (line 255)
        self_56028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 45), 'self', False)
        # Obtaining the member '_build_npy_pkg_config' of a type (line 255)
        _build_npy_pkg_config_56029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 45), self_56028, '_build_npy_pkg_config')
        # Calling _build_npy_pkg_config(args, kwargs) (line 255)
        _build_npy_pkg_config_call_result_56033 = invoke(stypy.reporting.localization.Localization(__file__, 255, 45), _build_npy_pkg_config_56029, *[info_56030, d_56031], **kwargs_56032)
        
        # Assigning a type to the variable 'call_assignment_55290' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 20), 'call_assignment_55290', _build_npy_pkg_config_call_result_56033)
        
        # Assigning a Call to a Name (line 255):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_56036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 20), 'int')
        # Processing the call keyword arguments
        kwargs_56037 = {}
        # Getting the type of 'call_assignment_55290' (line 255)
        call_assignment_55290_56034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 20), 'call_assignment_55290', False)
        # Obtaining the member '__getitem__' of a type (line 255)
        getitem___56035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 20), call_assignment_55290_56034, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_56038 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___56035, *[int_56036], **kwargs_56037)
        
        # Assigning a type to the variable 'call_assignment_55291' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 20), 'call_assignment_55291', getitem___call_result_56038)
        
        # Assigning a Name to a Name (line 255):
        # Getting the type of 'call_assignment_55291' (line 255)
        call_assignment_55291_56039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 20), 'call_assignment_55291')
        # Assigning a type to the variable 'install_dir' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 20), 'install_dir', call_assignment_55291_56039)
        
        # Assigning a Call to a Name (line 255):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_56042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 20), 'int')
        # Processing the call keyword arguments
        kwargs_56043 = {}
        # Getting the type of 'call_assignment_55290' (line 255)
        call_assignment_55290_56040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 20), 'call_assignment_55290', False)
        # Obtaining the member '__getitem__' of a type (line 255)
        getitem___56041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 20), call_assignment_55290_56040, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_56044 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___56041, *[int_56042], **kwargs_56043)
        
        # Assigning a type to the variable 'call_assignment_55292' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 20), 'call_assignment_55292', getitem___call_result_56044)
        
        # Assigning a Name to a Name (line 255):
        # Getting the type of 'call_assignment_55292' (line 255)
        call_assignment_55292_56045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 20), 'call_assignment_55292')
        # Assigning a type to the variable 'generated' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 33), 'generated', call_assignment_55292_56045)
        
        # Call to append(...): (line 256)
        # Processing the call arguments (line 256)
        
        # Obtaining an instance of the builtin type 'tuple' (line 256)
        tuple_56050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 57), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 256)
        # Adding element type (line 256)
        # Getting the type of 'install_dir' (line 256)
        install_dir_56051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 57), 'install_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 57), tuple_56050, install_dir_56051)
        # Adding element type (line 256)
        
        # Obtaining an instance of the builtin type 'list' (line 257)
        list_56052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 257)
        # Adding element type (line 257)
        # Getting the type of 'generated' (line 257)
        generated_56053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 25), 'generated', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 24), list_56052, generated_56053)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 57), tuple_56050, list_56052)
        
        # Processing the call keyword arguments (line 256)
        kwargs_56054 = {}
        # Getting the type of 'self' (line 256)
        self_56046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 20), 'self', False)
        # Obtaining the member 'distribution' of a type (line 256)
        distribution_56047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 20), self_56046, 'distribution')
        # Obtaining the member 'data_files' of a type (line 256)
        data_files_56048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 20), distribution_56047, 'data_files')
        # Obtaining the member 'append' of a type (line 256)
        append_56049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 20), data_files_56048, 'append')
        # Calling append(args, kwargs) (line 256)
        append_call_result_56055 = invoke(stypy.reporting.localization.Localization(__file__, 256, 20), append_56049, *[tuple_56050], **kwargs_56054)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 249)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'build_npy_pkg_config(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'build_npy_pkg_config' in the type store
        # Getting the type of 'stypy_return_type' (line 228)
        stypy_return_type_56056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_56056)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'build_npy_pkg_config'
        return stypy_return_type_56056


    @norecursion
    def build_py_modules_sources(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'build_py_modules_sources'
        module_type_store = module_type_store.open_function_context('build_py_modules_sources', 259, 4, False)
        # Assigning a type to the variable 'self' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_src.build_py_modules_sources.__dict__.__setitem__('stypy_localization', localization)
        build_src.build_py_modules_sources.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_src.build_py_modules_sources.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_src.build_py_modules_sources.__dict__.__setitem__('stypy_function_name', 'build_src.build_py_modules_sources')
        build_src.build_py_modules_sources.__dict__.__setitem__('stypy_param_names_list', [])
        build_src.build_py_modules_sources.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_src.build_py_modules_sources.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_src.build_py_modules_sources.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_src.build_py_modules_sources.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_src.build_py_modules_sources.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_src.build_py_modules_sources.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_src.build_py_modules_sources', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'build_py_modules_sources', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'build_py_modules_sources(...)' code ##################

        
        
        # Getting the type of 'self' (line 260)
        self_56057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 15), 'self')
        # Obtaining the member 'py_modules' of a type (line 260)
        py_modules_56058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 15), self_56057, 'py_modules')
        # Applying the 'not' unary operator (line 260)
        result_not__56059 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 11), 'not', py_modules_56058)
        
        # Testing the type of an if condition (line 260)
        if_condition_56060 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 260, 8), result_not__56059)
        # Assigning a type to the variable 'if_condition_56060' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'if_condition_56060', if_condition_56060)
        # SSA begins for if statement (line 260)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 260)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to info(...): (line 262)
        # Processing the call arguments (line 262)
        str_56063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 17), 'str', 'building py_modules sources')
        # Processing the call keyword arguments (line 262)
        kwargs_56064 = {}
        # Getting the type of 'log' (line 262)
        log_56061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 262)
        info_56062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 8), log_56061, 'info')
        # Calling info(args, kwargs) (line 262)
        info_call_result_56065 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), info_56062, *[str_56063], **kwargs_56064)
        
        
        # Assigning a List to a Name (line 263):
        
        # Assigning a List to a Name (line 263):
        
        # Obtaining an instance of the builtin type 'list' (line 263)
        list_56066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 263)
        
        # Assigning a type to the variable 'new_py_modules' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'new_py_modules', list_56066)
        
        # Getting the type of 'self' (line 264)
        self_56067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 22), 'self')
        # Obtaining the member 'py_modules' of a type (line 264)
        py_modules_56068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 22), self_56067, 'py_modules')
        # Testing the type of a for loop iterable (line 264)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 264, 8), py_modules_56068)
        # Getting the type of the for loop variable (line 264)
        for_loop_var_56069 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 264, 8), py_modules_56068)
        # Assigning a type to the variable 'source' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'source', for_loop_var_56069)
        # SSA begins for a for statement (line 264)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Call to is_sequence(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'source' (line 265)
        source_56071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 27), 'source', False)
        # Processing the call keyword arguments (line 265)
        kwargs_56072 = {}
        # Getting the type of 'is_sequence' (line 265)
        is_sequence_56070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 15), 'is_sequence', False)
        # Calling is_sequence(args, kwargs) (line 265)
        is_sequence_call_result_56073 = invoke(stypy.reporting.localization.Localization(__file__, 265, 15), is_sequence_56070, *[source_56071], **kwargs_56072)
        
        
        
        # Call to len(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'source' (line 265)
        source_56075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 43), 'source', False)
        # Processing the call keyword arguments (line 265)
        kwargs_56076 = {}
        # Getting the type of 'len' (line 265)
        len_56074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 39), 'len', False)
        # Calling len(args, kwargs) (line 265)
        len_call_result_56077 = invoke(stypy.reporting.localization.Localization(__file__, 265, 39), len_56074, *[source_56075], **kwargs_56076)
        
        int_56078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 52), 'int')
        # Applying the binary operator '==' (line 265)
        result_eq_56079 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 39), '==', len_call_result_56077, int_56078)
        
        # Applying the binary operator 'and' (line 265)
        result_and_keyword_56080 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 15), 'and', is_sequence_call_result_56073, result_eq_56079)
        
        # Testing the type of an if condition (line 265)
        if_condition_56081 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 265, 12), result_and_keyword_56080)
        # Assigning a type to the variable 'if_condition_56081' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'if_condition_56081', if_condition_56081)
        # SSA begins for if statement (line 265)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Tuple (line 266):
        
        # Assigning a Subscript to a Name (line 266):
        
        # Obtaining the type of the subscript
        int_56082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 16), 'int')
        # Getting the type of 'source' (line 266)
        source_56083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 47), 'source')
        # Obtaining the member '__getitem__' of a type (line 266)
        getitem___56084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 16), source_56083, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 266)
        subscript_call_result_56085 = invoke(stypy.reporting.localization.Localization(__file__, 266, 16), getitem___56084, int_56082)
        
        # Assigning a type to the variable 'tuple_var_assignment_55293' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'tuple_var_assignment_55293', subscript_call_result_56085)
        
        # Assigning a Subscript to a Name (line 266):
        
        # Obtaining the type of the subscript
        int_56086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 16), 'int')
        # Getting the type of 'source' (line 266)
        source_56087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 47), 'source')
        # Obtaining the member '__getitem__' of a type (line 266)
        getitem___56088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 16), source_56087, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 266)
        subscript_call_result_56089 = invoke(stypy.reporting.localization.Localization(__file__, 266, 16), getitem___56088, int_56086)
        
        # Assigning a type to the variable 'tuple_var_assignment_55294' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'tuple_var_assignment_55294', subscript_call_result_56089)
        
        # Assigning a Subscript to a Name (line 266):
        
        # Obtaining the type of the subscript
        int_56090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 16), 'int')
        # Getting the type of 'source' (line 266)
        source_56091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 47), 'source')
        # Obtaining the member '__getitem__' of a type (line 266)
        getitem___56092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 16), source_56091, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 266)
        subscript_call_result_56093 = invoke(stypy.reporting.localization.Localization(__file__, 266, 16), getitem___56092, int_56090)
        
        # Assigning a type to the variable 'tuple_var_assignment_55295' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'tuple_var_assignment_55295', subscript_call_result_56093)
        
        # Assigning a Name to a Name (line 266):
        # Getting the type of 'tuple_var_assignment_55293' (line 266)
        tuple_var_assignment_55293_56094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'tuple_var_assignment_55293')
        # Assigning a type to the variable 'package' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'package', tuple_var_assignment_55293_56094)
        
        # Assigning a Name to a Name (line 266):
        # Getting the type of 'tuple_var_assignment_55294' (line 266)
        tuple_var_assignment_55294_56095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'tuple_var_assignment_55294')
        # Assigning a type to the variable 'module_base' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 25), 'module_base', tuple_var_assignment_55294_56095)
        
        # Assigning a Name to a Name (line 266):
        # Getting the type of 'tuple_var_assignment_55295' (line 266)
        tuple_var_assignment_55295_56096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'tuple_var_assignment_55295')
        # Assigning a type to the variable 'source' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 38), 'source', tuple_var_assignment_55295_56096)
        
        # Getting the type of 'self' (line 267)
        self_56097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 19), 'self')
        # Obtaining the member 'inplace' of a type (line 267)
        inplace_56098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 19), self_56097, 'inplace')
        # Testing the type of an if condition (line 267)
        if_condition_56099 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 267, 16), inplace_56098)
        # Assigning a type to the variable 'if_condition_56099' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 16), 'if_condition_56099', if_condition_56099)
        # SSA begins for if statement (line 267)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 268):
        
        # Assigning a Call to a Name (line 268):
        
        # Call to get_package_dir(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'package' (line 268)
        package_56102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 53), 'package', False)
        # Processing the call keyword arguments (line 268)
        kwargs_56103 = {}
        # Getting the type of 'self' (line 268)
        self_56100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 32), 'self', False)
        # Obtaining the member 'get_package_dir' of a type (line 268)
        get_package_dir_56101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 32), self_56100, 'get_package_dir')
        # Calling get_package_dir(args, kwargs) (line 268)
        get_package_dir_call_result_56104 = invoke(stypy.reporting.localization.Localization(__file__, 268, 32), get_package_dir_56101, *[package_56102], **kwargs_56103)
        
        # Assigning a type to the variable 'build_dir' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 20), 'build_dir', get_package_dir_call_result_56104)
        # SSA branch for the else part of an if statement (line 267)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 270):
        
        # Assigning a Call to a Name (line 270):
        
        # Call to join(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'self' (line 270)
        self_56108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 45), 'self', False)
        # Obtaining the member 'build_src' of a type (line 270)
        build_src_56109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 45), self_56108, 'build_src')
        
        # Call to join(...): (line 271)
        
        # Call to split(...): (line 271)
        # Processing the call arguments (line 271)
        str_56115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 73), 'str', '.')
        # Processing the call keyword arguments (line 271)
        kwargs_56116 = {}
        # Getting the type of 'package' (line 271)
        package_56113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 59), 'package', False)
        # Obtaining the member 'split' of a type (line 271)
        split_56114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 59), package_56113, 'split')
        # Calling split(args, kwargs) (line 271)
        split_call_result_56117 = invoke(stypy.reporting.localization.Localization(__file__, 271, 59), split_56114, *[str_56115], **kwargs_56116)
        
        # Processing the call keyword arguments (line 271)
        kwargs_56118 = {}
        # Getting the type of 'os' (line 271)
        os_56110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 45), 'os', False)
        # Obtaining the member 'path' of a type (line 271)
        path_56111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 45), os_56110, 'path')
        # Obtaining the member 'join' of a type (line 271)
        join_56112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 45), path_56111, 'join')
        # Calling join(args, kwargs) (line 271)
        join_call_result_56119 = invoke(stypy.reporting.localization.Localization(__file__, 271, 45), join_56112, *[split_call_result_56117], **kwargs_56118)
        
        # Processing the call keyword arguments (line 270)
        kwargs_56120 = {}
        # Getting the type of 'os' (line 270)
        os_56105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 32), 'os', False)
        # Obtaining the member 'path' of a type (line 270)
        path_56106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 32), os_56105, 'path')
        # Obtaining the member 'join' of a type (line 270)
        join_56107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 32), path_56106, 'join')
        # Calling join(args, kwargs) (line 270)
        join_call_result_56121 = invoke(stypy.reporting.localization.Localization(__file__, 270, 32), join_56107, *[build_src_56109, join_call_result_56119], **kwargs_56120)
        
        # Assigning a type to the variable 'build_dir' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 20), 'build_dir', join_call_result_56121)
        # SSA join for if statement (line 267)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 272)
        str_56122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 35), 'str', '__call__')
        # Getting the type of 'source' (line 272)
        source_56123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 27), 'source')
        
        (may_be_56124, more_types_in_union_56125) = may_provide_member(str_56122, source_56123)

        if may_be_56124:

            if more_types_in_union_56125:
                # Runtime conditional SSA (line 272)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'source' (line 272)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 16), 'source', remove_not_member_provider_from_union(source_56123, '__call__'))
            
            # Assigning a Call to a Name (line 273):
            
            # Assigning a Call to a Name (line 273):
            
            # Call to join(...): (line 273)
            # Processing the call arguments (line 273)
            # Getting the type of 'build_dir' (line 273)
            build_dir_56129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 42), 'build_dir', False)
            # Getting the type of 'module_base' (line 273)
            module_base_56130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 53), 'module_base', False)
            str_56131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 67), 'str', '.py')
            # Applying the binary operator '+' (line 273)
            result_add_56132 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 53), '+', module_base_56130, str_56131)
            
            # Processing the call keyword arguments (line 273)
            kwargs_56133 = {}
            # Getting the type of 'os' (line 273)
            os_56126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 29), 'os', False)
            # Obtaining the member 'path' of a type (line 273)
            path_56127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 29), os_56126, 'path')
            # Obtaining the member 'join' of a type (line 273)
            join_56128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 29), path_56127, 'join')
            # Calling join(args, kwargs) (line 273)
            join_call_result_56134 = invoke(stypy.reporting.localization.Localization(__file__, 273, 29), join_56128, *[build_dir_56129, result_add_56132], **kwargs_56133)
            
            # Assigning a type to the variable 'target' (line 273)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 20), 'target', join_call_result_56134)
            
            # Assigning a Call to a Name (line 274):
            
            # Assigning a Call to a Name (line 274):
            
            # Call to source(...): (line 274)
            # Processing the call arguments (line 274)
            # Getting the type of 'target' (line 274)
            target_56136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 36), 'target', False)
            # Processing the call keyword arguments (line 274)
            kwargs_56137 = {}
            # Getting the type of 'source' (line 274)
            source_56135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 29), 'source', False)
            # Calling source(args, kwargs) (line 274)
            source_call_result_56138 = invoke(stypy.reporting.localization.Localization(__file__, 274, 29), source_56135, *[target_56136], **kwargs_56137)
            
            # Assigning a type to the variable 'source' (line 274)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 20), 'source', source_call_result_56138)

            if more_types_in_union_56125:
                # SSA join for if statement (line 272)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 275)
        # Getting the type of 'source' (line 275)
        source_56139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 19), 'source')
        # Getting the type of 'None' (line 275)
        None_56140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 29), 'None')
        
        (may_be_56141, more_types_in_union_56142) = may_be_none(source_56139, None_56140)

        if may_be_56141:

            if more_types_in_union_56142:
                # Runtime conditional SSA (line 275)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store


            if more_types_in_union_56142:
                # SSA join for if statement (line 275)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a List to a Name (line 277):
        
        # Assigning a List to a Name (line 277):
        
        # Obtaining an instance of the builtin type 'list' (line 277)
        list_56143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 277)
        # Adding element type (line 277)
        
        # Obtaining an instance of the builtin type 'tuple' (line 277)
        tuple_56144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 277)
        # Adding element type (line 277)
        # Getting the type of 'package' (line 277)
        package_56145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 28), 'package')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 28), tuple_56144, package_56145)
        # Adding element type (line 277)
        # Getting the type of 'module_base' (line 277)
        module_base_56146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 37), 'module_base')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 28), tuple_56144, module_base_56146)
        # Adding element type (line 277)
        # Getting the type of 'source' (line 277)
        source_56147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 50), 'source')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 28), tuple_56144, source_56147)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 26), list_56143, tuple_56144)
        
        # Assigning a type to the variable 'modules' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 16), 'modules', list_56143)
        
        
        # Getting the type of 'package' (line 278)
        package_56148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 19), 'package')
        # Getting the type of 'self' (line 278)
        self_56149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 34), 'self')
        # Obtaining the member 'py_modules_dict' of a type (line 278)
        py_modules_dict_56150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 34), self_56149, 'py_modules_dict')
        # Applying the binary operator 'notin' (line 278)
        result_contains_56151 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 19), 'notin', package_56148, py_modules_dict_56150)
        
        # Testing the type of an if condition (line 278)
        if_condition_56152 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 278, 16), result_contains_56151)
        # Assigning a type to the variable 'if_condition_56152' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 16), 'if_condition_56152', if_condition_56152)
        # SSA begins for if statement (line 278)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Subscript (line 279):
        
        # Assigning a List to a Subscript (line 279):
        
        # Obtaining an instance of the builtin type 'list' (line 279)
        list_56153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 279)
        
        # Getting the type of 'self' (line 279)
        self_56154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 20), 'self')
        # Obtaining the member 'py_modules_dict' of a type (line 279)
        py_modules_dict_56155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 20), self_56154, 'py_modules_dict')
        # Getting the type of 'package' (line 279)
        package_56156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 41), 'package')
        # Storing an element on a container (line 279)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 20), py_modules_dict_56155, (package_56156, list_56153))
        # SSA join for if statement (line 278)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 280)
        self_56157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 16), 'self')
        # Obtaining the member 'py_modules_dict' of a type (line 280)
        py_modules_dict_56158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 16), self_56157, 'py_modules_dict')
        
        # Obtaining the type of the subscript
        # Getting the type of 'package' (line 280)
        package_56159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 37), 'package')
        # Getting the type of 'self' (line 280)
        self_56160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 16), 'self')
        # Obtaining the member 'py_modules_dict' of a type (line 280)
        py_modules_dict_56161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 16), self_56160, 'py_modules_dict')
        # Obtaining the member '__getitem__' of a type (line 280)
        getitem___56162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 16), py_modules_dict_56161, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 280)
        subscript_call_result_56163 = invoke(stypy.reporting.localization.Localization(__file__, 280, 16), getitem___56162, package_56159)
        
        # Getting the type of 'modules' (line 280)
        modules_56164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 49), 'modules')
        # Applying the binary operator '+=' (line 280)
        result_iadd_56165 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 16), '+=', subscript_call_result_56163, modules_56164)
        # Getting the type of 'self' (line 280)
        self_56166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 16), 'self')
        # Obtaining the member 'py_modules_dict' of a type (line 280)
        py_modules_dict_56167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 16), self_56166, 'py_modules_dict')
        # Getting the type of 'package' (line 280)
        package_56168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 37), 'package')
        # Storing an element on a container (line 280)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 16), py_modules_dict_56167, (package_56168, result_iadd_56165))
        
        # SSA branch for the else part of an if statement (line 265)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'source' (line 282)
        source_56171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 38), 'source', False)
        # Processing the call keyword arguments (line 282)
        kwargs_56172 = {}
        # Getting the type of 'new_py_modules' (line 282)
        new_py_modules_56169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 16), 'new_py_modules', False)
        # Obtaining the member 'append' of a type (line 282)
        append_56170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 16), new_py_modules_56169, 'append')
        # Calling append(args, kwargs) (line 282)
        append_call_result_56173 = invoke(stypy.reporting.localization.Localization(__file__, 282, 16), append_56170, *[source_56171], **kwargs_56172)
        
        # SSA join for if statement (line 265)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 283):
        
        # Assigning a Name to a Subscript (line 283):
        # Getting the type of 'new_py_modules' (line 283)
        new_py_modules_56174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 29), 'new_py_modules')
        # Getting the type of 'self' (line 283)
        self_56175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'self')
        # Obtaining the member 'py_modules' of a type (line 283)
        py_modules_56176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 8), self_56175, 'py_modules')
        slice_56177 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 283, 8), None, None, None)
        # Storing an element on a container (line 283)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 8), py_modules_56176, (slice_56177, new_py_modules_56174))
        
        # ################# End of 'build_py_modules_sources(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'build_py_modules_sources' in the type store
        # Getting the type of 'stypy_return_type' (line 259)
        stypy_return_type_56178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_56178)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'build_py_modules_sources'
        return stypy_return_type_56178


    @norecursion
    def build_library_sources(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'build_library_sources'
        module_type_store = module_type_store.open_function_context('build_library_sources', 285, 4, False)
        # Assigning a type to the variable 'self' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_src.build_library_sources.__dict__.__setitem__('stypy_localization', localization)
        build_src.build_library_sources.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_src.build_library_sources.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_src.build_library_sources.__dict__.__setitem__('stypy_function_name', 'build_src.build_library_sources')
        build_src.build_library_sources.__dict__.__setitem__('stypy_param_names_list', ['lib_name', 'build_info'])
        build_src.build_library_sources.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_src.build_library_sources.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_src.build_library_sources.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_src.build_library_sources.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_src.build_library_sources.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_src.build_library_sources.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_src.build_library_sources', ['lib_name', 'build_info'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'build_library_sources', localization, ['lib_name', 'build_info'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'build_library_sources(...)' code ##################

        
        # Assigning a Call to a Name (line 286):
        
        # Assigning a Call to a Name (line 286):
        
        # Call to list(...): (line 286)
        # Processing the call arguments (line 286)
        
        # Call to get(...): (line 286)
        # Processing the call arguments (line 286)
        str_56182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 38), 'str', 'sources')
        
        # Obtaining an instance of the builtin type 'list' (line 286)
        list_56183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 286)
        
        # Processing the call keyword arguments (line 286)
        kwargs_56184 = {}
        # Getting the type of 'build_info' (line 286)
        build_info_56180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 23), 'build_info', False)
        # Obtaining the member 'get' of a type (line 286)
        get_56181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 23), build_info_56180, 'get')
        # Calling get(args, kwargs) (line 286)
        get_call_result_56185 = invoke(stypy.reporting.localization.Localization(__file__, 286, 23), get_56181, *[str_56182, list_56183], **kwargs_56184)
        
        # Processing the call keyword arguments (line 286)
        kwargs_56186 = {}
        # Getting the type of 'list' (line 286)
        list_56179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 18), 'list', False)
        # Calling list(args, kwargs) (line 286)
        list_call_result_56187 = invoke(stypy.reporting.localization.Localization(__file__, 286, 18), list_56179, *[get_call_result_56185], **kwargs_56186)
        
        # Assigning a type to the variable 'sources' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'sources', list_call_result_56187)
        
        
        # Getting the type of 'sources' (line 288)
        sources_56188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 15), 'sources')
        # Applying the 'not' unary operator (line 288)
        result_not__56189 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 11), 'not', sources_56188)
        
        # Testing the type of an if condition (line 288)
        if_condition_56190 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 288, 8), result_not__56189)
        # Assigning a type to the variable 'if_condition_56190' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'if_condition_56190', if_condition_56190)
        # SSA begins for if statement (line 288)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 288)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to info(...): (line 291)
        # Processing the call arguments (line 291)
        str_56193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 17), 'str', 'building library "%s" sources')
        # Getting the type of 'lib_name' (line 291)
        lib_name_56194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 52), 'lib_name', False)
        # Applying the binary operator '%' (line 291)
        result_mod_56195 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 17), '%', str_56193, lib_name_56194)
        
        # Processing the call keyword arguments (line 291)
        kwargs_56196 = {}
        # Getting the type of 'log' (line 291)
        log_56191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 291)
        info_56192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 8), log_56191, 'info')
        # Calling info(args, kwargs) (line 291)
        info_call_result_56197 = invoke(stypy.reporting.localization.Localization(__file__, 291, 8), info_56192, *[result_mod_56195], **kwargs_56196)
        
        
        # Assigning a Call to a Name (line 293):
        
        # Assigning a Call to a Name (line 293):
        
        # Call to generate_sources(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'sources' (line 293)
        sources_56200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 40), 'sources', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 293)
        tuple_56201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 293)
        # Adding element type (line 293)
        # Getting the type of 'lib_name' (line 293)
        lib_name_56202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 50), 'lib_name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 50), tuple_56201, lib_name_56202)
        # Adding element type (line 293)
        # Getting the type of 'build_info' (line 293)
        build_info_56203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 60), 'build_info', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 50), tuple_56201, build_info_56203)
        
        # Processing the call keyword arguments (line 293)
        kwargs_56204 = {}
        # Getting the type of 'self' (line 293)
        self_56198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 18), 'self', False)
        # Obtaining the member 'generate_sources' of a type (line 293)
        generate_sources_56199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 18), self_56198, 'generate_sources')
        # Calling generate_sources(args, kwargs) (line 293)
        generate_sources_call_result_56205 = invoke(stypy.reporting.localization.Localization(__file__, 293, 18), generate_sources_56199, *[sources_56200, tuple_56201], **kwargs_56204)
        
        # Assigning a type to the variable 'sources' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'sources', generate_sources_call_result_56205)
        
        # Assigning a Call to a Name (line 295):
        
        # Assigning a Call to a Name (line 295):
        
        # Call to template_sources(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'sources' (line 295)
        sources_56208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 40), 'sources', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 295)
        tuple_56209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 295)
        # Adding element type (line 295)
        # Getting the type of 'lib_name' (line 295)
        lib_name_56210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 50), 'lib_name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 50), tuple_56209, lib_name_56210)
        # Adding element type (line 295)
        # Getting the type of 'build_info' (line 295)
        build_info_56211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 60), 'build_info', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 50), tuple_56209, build_info_56211)
        
        # Processing the call keyword arguments (line 295)
        kwargs_56212 = {}
        # Getting the type of 'self' (line 295)
        self_56206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 18), 'self', False)
        # Obtaining the member 'template_sources' of a type (line 295)
        template_sources_56207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 18), self_56206, 'template_sources')
        # Calling template_sources(args, kwargs) (line 295)
        template_sources_call_result_56213 = invoke(stypy.reporting.localization.Localization(__file__, 295, 18), template_sources_56207, *[sources_56208, tuple_56209], **kwargs_56212)
        
        # Assigning a type to the variable 'sources' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'sources', template_sources_call_result_56213)
        
        # Assigning a Call to a Tuple (line 297):
        
        # Assigning a Call to a Name:
        
        # Call to filter_h_files(...): (line 297)
        # Processing the call arguments (line 297)
        # Getting the type of 'sources' (line 297)
        sources_56216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 47), 'sources', False)
        # Processing the call keyword arguments (line 297)
        kwargs_56217 = {}
        # Getting the type of 'self' (line 297)
        self_56214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 27), 'self', False)
        # Obtaining the member 'filter_h_files' of a type (line 297)
        filter_h_files_56215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 27), self_56214, 'filter_h_files')
        # Calling filter_h_files(args, kwargs) (line 297)
        filter_h_files_call_result_56218 = invoke(stypy.reporting.localization.Localization(__file__, 297, 27), filter_h_files_56215, *[sources_56216], **kwargs_56217)
        
        # Assigning a type to the variable 'call_assignment_55296' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'call_assignment_55296', filter_h_files_call_result_56218)
        
        # Assigning a Call to a Name (line 297):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_56221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 8), 'int')
        # Processing the call keyword arguments
        kwargs_56222 = {}
        # Getting the type of 'call_assignment_55296' (line 297)
        call_assignment_55296_56219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'call_assignment_55296', False)
        # Obtaining the member '__getitem__' of a type (line 297)
        getitem___56220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 8), call_assignment_55296_56219, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_56223 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___56220, *[int_56221], **kwargs_56222)
        
        # Assigning a type to the variable 'call_assignment_55297' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'call_assignment_55297', getitem___call_result_56223)
        
        # Assigning a Name to a Name (line 297):
        # Getting the type of 'call_assignment_55297' (line 297)
        call_assignment_55297_56224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'call_assignment_55297')
        # Assigning a type to the variable 'sources' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'sources', call_assignment_55297_56224)
        
        # Assigning a Call to a Name (line 297):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_56227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 8), 'int')
        # Processing the call keyword arguments
        kwargs_56228 = {}
        # Getting the type of 'call_assignment_55296' (line 297)
        call_assignment_55296_56225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'call_assignment_55296', False)
        # Obtaining the member '__getitem__' of a type (line 297)
        getitem___56226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 8), call_assignment_55296_56225, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_56229 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___56226, *[int_56227], **kwargs_56228)
        
        # Assigning a type to the variable 'call_assignment_55298' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'call_assignment_55298', getitem___call_result_56229)
        
        # Assigning a Name to a Name (line 297):
        # Getting the type of 'call_assignment_55298' (line 297)
        call_assignment_55298_56230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'call_assignment_55298')
        # Assigning a type to the variable 'h_files' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 17), 'h_files', call_assignment_55298_56230)
        
        # Getting the type of 'h_files' (line 299)
        h_files_56231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 11), 'h_files')
        # Testing the type of an if condition (line 299)
        if_condition_56232 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 299, 8), h_files_56231)
        # Assigning a type to the variable 'if_condition_56232' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'if_condition_56232', if_condition_56232)
        # SSA begins for if statement (line 299)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 300)
        # Processing the call arguments (line 300)
        str_56235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 21), 'str', '%s - nothing done with h_files = %s')
        # Getting the type of 'self' (line 301)
        self_56236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 21), 'self', False)
        # Obtaining the member 'package' of a type (line 301)
        package_56237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 21), self_56236, 'package')
        # Getting the type of 'h_files' (line 301)
        h_files_56238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 35), 'h_files', False)
        # Processing the call keyword arguments (line 300)
        kwargs_56239 = {}
        # Getting the type of 'log' (line 300)
        log_56233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'log', False)
        # Obtaining the member 'info' of a type (line 300)
        info_56234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 12), log_56233, 'info')
        # Calling info(args, kwargs) (line 300)
        info_call_result_56240 = invoke(stypy.reporting.localization.Localization(__file__, 300, 12), info_56234, *[str_56235, package_56237, h_files_56238], **kwargs_56239)
        
        # SSA join for if statement (line 299)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 306):
        
        # Assigning a Name to a Subscript (line 306):
        # Getting the type of 'sources' (line 306)
        sources_56241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 32), 'sources')
        # Getting the type of 'build_info' (line 306)
        build_info_56242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'build_info')
        str_56243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 19), 'str', 'sources')
        # Storing an element on a container (line 306)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 8), build_info_56242, (str_56243, sources_56241))
        # Assigning a type to the variable 'stypy_return_type' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of 'build_library_sources(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'build_library_sources' in the type store
        # Getting the type of 'stypy_return_type' (line 285)
        stypy_return_type_56244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_56244)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'build_library_sources'
        return stypy_return_type_56244


    @norecursion
    def build_extension_sources(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'build_extension_sources'
        module_type_store = module_type_store.open_function_context('build_extension_sources', 309, 4, False)
        # Assigning a type to the variable 'self' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_src.build_extension_sources.__dict__.__setitem__('stypy_localization', localization)
        build_src.build_extension_sources.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_src.build_extension_sources.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_src.build_extension_sources.__dict__.__setitem__('stypy_function_name', 'build_src.build_extension_sources')
        build_src.build_extension_sources.__dict__.__setitem__('stypy_param_names_list', ['ext'])
        build_src.build_extension_sources.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_src.build_extension_sources.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_src.build_extension_sources.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_src.build_extension_sources.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_src.build_extension_sources.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_src.build_extension_sources.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_src.build_extension_sources', ['ext'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'build_extension_sources', localization, ['ext'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'build_extension_sources(...)' code ##################

        
        # Assigning a Call to a Name (line 311):
        
        # Assigning a Call to a Name (line 311):
        
        # Call to list(...): (line 311)
        # Processing the call arguments (line 311)
        # Getting the type of 'ext' (line 311)
        ext_56246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 23), 'ext', False)
        # Obtaining the member 'sources' of a type (line 311)
        sources_56247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 23), ext_56246, 'sources')
        # Processing the call keyword arguments (line 311)
        kwargs_56248 = {}
        # Getting the type of 'list' (line 311)
        list_56245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 18), 'list', False)
        # Calling list(args, kwargs) (line 311)
        list_call_result_56249 = invoke(stypy.reporting.localization.Localization(__file__, 311, 18), list_56245, *[sources_56247], **kwargs_56248)
        
        # Assigning a type to the variable 'sources' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'sources', list_call_result_56249)
        
        # Call to info(...): (line 313)
        # Processing the call arguments (line 313)
        str_56252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 17), 'str', 'building extension "%s" sources')
        # Getting the type of 'ext' (line 313)
        ext_56253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 54), 'ext', False)
        # Obtaining the member 'name' of a type (line 313)
        name_56254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 54), ext_56253, 'name')
        # Applying the binary operator '%' (line 313)
        result_mod_56255 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 17), '%', str_56252, name_56254)
        
        # Processing the call keyword arguments (line 313)
        kwargs_56256 = {}
        # Getting the type of 'log' (line 313)
        log_56250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 313)
        info_56251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 8), log_56250, 'info')
        # Calling info(args, kwargs) (line 313)
        info_call_result_56257 = invoke(stypy.reporting.localization.Localization(__file__, 313, 8), info_56251, *[result_mod_56255], **kwargs_56256)
        
        
        # Assigning a Call to a Name (line 315):
        
        # Assigning a Call to a Name (line 315):
        
        # Call to get_ext_fullname(...): (line 315)
        # Processing the call arguments (line 315)
        # Getting the type of 'ext' (line 315)
        ext_56260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 41), 'ext', False)
        # Obtaining the member 'name' of a type (line 315)
        name_56261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 41), ext_56260, 'name')
        # Processing the call keyword arguments (line 315)
        kwargs_56262 = {}
        # Getting the type of 'self' (line 315)
        self_56258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 19), 'self', False)
        # Obtaining the member 'get_ext_fullname' of a type (line 315)
        get_ext_fullname_56259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 19), self_56258, 'get_ext_fullname')
        # Calling get_ext_fullname(args, kwargs) (line 315)
        get_ext_fullname_call_result_56263 = invoke(stypy.reporting.localization.Localization(__file__, 315, 19), get_ext_fullname_56259, *[name_56261], **kwargs_56262)
        
        # Assigning a type to the variable 'fullname' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'fullname', get_ext_fullname_call_result_56263)
        
        # Assigning a Call to a Name (line 317):
        
        # Assigning a Call to a Name (line 317):
        
        # Call to split(...): (line 317)
        # Processing the call arguments (line 317)
        str_56266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 33), 'str', '.')
        # Processing the call keyword arguments (line 317)
        kwargs_56267 = {}
        # Getting the type of 'fullname' (line 317)
        fullname_56264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 18), 'fullname', False)
        # Obtaining the member 'split' of a type (line 317)
        split_56265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 18), fullname_56264, 'split')
        # Calling split(args, kwargs) (line 317)
        split_call_result_56268 = invoke(stypy.reporting.localization.Localization(__file__, 317, 18), split_56265, *[str_56266], **kwargs_56267)
        
        # Assigning a type to the variable 'modpath' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'modpath', split_call_result_56268)
        
        # Assigning a Call to a Name (line 318):
        
        # Assigning a Call to a Name (line 318):
        
        # Call to join(...): (line 318)
        # Processing the call arguments (line 318)
        
        # Obtaining the type of the subscript
        int_56271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 35), 'int')
        int_56272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 37), 'int')
        slice_56273 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 318, 27), int_56271, int_56272, None)
        # Getting the type of 'modpath' (line 318)
        modpath_56274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 27), 'modpath', False)
        # Obtaining the member '__getitem__' of a type (line 318)
        getitem___56275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 27), modpath_56274, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 318)
        subscript_call_result_56276 = invoke(stypy.reporting.localization.Localization(__file__, 318, 27), getitem___56275, slice_56273)
        
        # Processing the call keyword arguments (line 318)
        kwargs_56277 = {}
        str_56269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 18), 'str', '.')
        # Obtaining the member 'join' of a type (line 318)
        join_56270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 18), str_56269, 'join')
        # Calling join(args, kwargs) (line 318)
        join_call_result_56278 = invoke(stypy.reporting.localization.Localization(__file__, 318, 18), join_56270, *[subscript_call_result_56276], **kwargs_56277)
        
        # Assigning a type to the variable 'package' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'package', join_call_result_56278)
        
        # Getting the type of 'self' (line 320)
        self_56279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 11), 'self')
        # Obtaining the member 'inplace' of a type (line 320)
        inplace_56280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 11), self_56279, 'inplace')
        # Testing the type of an if condition (line 320)
        if_condition_56281 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 320, 8), inplace_56280)
        # Assigning a type to the variable 'if_condition_56281' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'if_condition_56281', if_condition_56281)
        # SSA begins for if statement (line 320)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 321):
        
        # Assigning a Call to a Attribute (line 321):
        
        # Call to get_package_dir(...): (line 321)
        # Processing the call arguments (line 321)
        # Getting the type of 'package' (line 321)
        package_56284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 55), 'package', False)
        # Processing the call keyword arguments (line 321)
        kwargs_56285 = {}
        # Getting the type of 'self' (line 321)
        self_56282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 34), 'self', False)
        # Obtaining the member 'get_package_dir' of a type (line 321)
        get_package_dir_56283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 34), self_56282, 'get_package_dir')
        # Calling get_package_dir(args, kwargs) (line 321)
        get_package_dir_call_result_56286 = invoke(stypy.reporting.localization.Localization(__file__, 321, 34), get_package_dir_56283, *[package_56284], **kwargs_56285)
        
        # Getting the type of 'self' (line 321)
        self_56287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'self')
        # Setting the type of the member 'ext_target_dir' of a type (line 321)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 12), self_56287, 'ext_target_dir', get_package_dir_call_result_56286)
        # SSA join for if statement (line 320)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 323):
        
        # Assigning a Call to a Name (line 323):
        
        # Call to generate_sources(...): (line 323)
        # Processing the call arguments (line 323)
        # Getting the type of 'sources' (line 323)
        sources_56290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 40), 'sources', False)
        # Getting the type of 'ext' (line 323)
        ext_56291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 49), 'ext', False)
        # Processing the call keyword arguments (line 323)
        kwargs_56292 = {}
        # Getting the type of 'self' (line 323)
        self_56288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 18), 'self', False)
        # Obtaining the member 'generate_sources' of a type (line 323)
        generate_sources_56289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 18), self_56288, 'generate_sources')
        # Calling generate_sources(args, kwargs) (line 323)
        generate_sources_call_result_56293 = invoke(stypy.reporting.localization.Localization(__file__, 323, 18), generate_sources_56289, *[sources_56290, ext_56291], **kwargs_56292)
        
        # Assigning a type to the variable 'sources' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'sources', generate_sources_call_result_56293)
        
        # Assigning a Call to a Name (line 324):
        
        # Assigning a Call to a Name (line 324):
        
        # Call to template_sources(...): (line 324)
        # Processing the call arguments (line 324)
        # Getting the type of 'sources' (line 324)
        sources_56296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 40), 'sources', False)
        # Getting the type of 'ext' (line 324)
        ext_56297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 49), 'ext', False)
        # Processing the call keyword arguments (line 324)
        kwargs_56298 = {}
        # Getting the type of 'self' (line 324)
        self_56294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 18), 'self', False)
        # Obtaining the member 'template_sources' of a type (line 324)
        template_sources_56295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 18), self_56294, 'template_sources')
        # Calling template_sources(args, kwargs) (line 324)
        template_sources_call_result_56299 = invoke(stypy.reporting.localization.Localization(__file__, 324, 18), template_sources_56295, *[sources_56296, ext_56297], **kwargs_56298)
        
        # Assigning a type to the variable 'sources' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'sources', template_sources_call_result_56299)
        
        # Assigning a Call to a Name (line 325):
        
        # Assigning a Call to a Name (line 325):
        
        # Call to swig_sources(...): (line 325)
        # Processing the call arguments (line 325)
        # Getting the type of 'sources' (line 325)
        sources_56302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 36), 'sources', False)
        # Getting the type of 'ext' (line 325)
        ext_56303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 45), 'ext', False)
        # Processing the call keyword arguments (line 325)
        kwargs_56304 = {}
        # Getting the type of 'self' (line 325)
        self_56300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 18), 'self', False)
        # Obtaining the member 'swig_sources' of a type (line 325)
        swig_sources_56301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 18), self_56300, 'swig_sources')
        # Calling swig_sources(args, kwargs) (line 325)
        swig_sources_call_result_56305 = invoke(stypy.reporting.localization.Localization(__file__, 325, 18), swig_sources_56301, *[sources_56302, ext_56303], **kwargs_56304)
        
        # Assigning a type to the variable 'sources' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'sources', swig_sources_call_result_56305)
        
        # Assigning a Call to a Name (line 326):
        
        # Assigning a Call to a Name (line 326):
        
        # Call to f2py_sources(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'sources' (line 326)
        sources_56308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 36), 'sources', False)
        # Getting the type of 'ext' (line 326)
        ext_56309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 45), 'ext', False)
        # Processing the call keyword arguments (line 326)
        kwargs_56310 = {}
        # Getting the type of 'self' (line 326)
        self_56306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 18), 'self', False)
        # Obtaining the member 'f2py_sources' of a type (line 326)
        f2py_sources_56307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 18), self_56306, 'f2py_sources')
        # Calling f2py_sources(args, kwargs) (line 326)
        f2py_sources_call_result_56311 = invoke(stypy.reporting.localization.Localization(__file__, 326, 18), f2py_sources_56307, *[sources_56308, ext_56309], **kwargs_56310)
        
        # Assigning a type to the variable 'sources' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'sources', f2py_sources_call_result_56311)
        
        # Assigning a Call to a Name (line 327):
        
        # Assigning a Call to a Name (line 327):
        
        # Call to pyrex_sources(...): (line 327)
        # Processing the call arguments (line 327)
        # Getting the type of 'sources' (line 327)
        sources_56314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 37), 'sources', False)
        # Getting the type of 'ext' (line 327)
        ext_56315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 46), 'ext', False)
        # Processing the call keyword arguments (line 327)
        kwargs_56316 = {}
        # Getting the type of 'self' (line 327)
        self_56312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 18), 'self', False)
        # Obtaining the member 'pyrex_sources' of a type (line 327)
        pyrex_sources_56313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 18), self_56312, 'pyrex_sources')
        # Calling pyrex_sources(args, kwargs) (line 327)
        pyrex_sources_call_result_56317 = invoke(stypy.reporting.localization.Localization(__file__, 327, 18), pyrex_sources_56313, *[sources_56314, ext_56315], **kwargs_56316)
        
        # Assigning a type to the variable 'sources' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'sources', pyrex_sources_call_result_56317)
        
        # Assigning a Call to a Tuple (line 329):
        
        # Assigning a Call to a Name:
        
        # Call to filter_py_files(...): (line 329)
        # Processing the call arguments (line 329)
        # Getting the type of 'sources' (line 329)
        sources_56320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 49), 'sources', False)
        # Processing the call keyword arguments (line 329)
        kwargs_56321 = {}
        # Getting the type of 'self' (line 329)
        self_56318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 28), 'self', False)
        # Obtaining the member 'filter_py_files' of a type (line 329)
        filter_py_files_56319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 28), self_56318, 'filter_py_files')
        # Calling filter_py_files(args, kwargs) (line 329)
        filter_py_files_call_result_56322 = invoke(stypy.reporting.localization.Localization(__file__, 329, 28), filter_py_files_56319, *[sources_56320], **kwargs_56321)
        
        # Assigning a type to the variable 'call_assignment_55299' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'call_assignment_55299', filter_py_files_call_result_56322)
        
        # Assigning a Call to a Name (line 329):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_56325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 8), 'int')
        # Processing the call keyword arguments
        kwargs_56326 = {}
        # Getting the type of 'call_assignment_55299' (line 329)
        call_assignment_55299_56323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'call_assignment_55299', False)
        # Obtaining the member '__getitem__' of a type (line 329)
        getitem___56324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 8), call_assignment_55299_56323, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_56327 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___56324, *[int_56325], **kwargs_56326)
        
        # Assigning a type to the variable 'call_assignment_55300' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'call_assignment_55300', getitem___call_result_56327)
        
        # Assigning a Name to a Name (line 329):
        # Getting the type of 'call_assignment_55300' (line 329)
        call_assignment_55300_56328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'call_assignment_55300')
        # Assigning a type to the variable 'sources' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'sources', call_assignment_55300_56328)
        
        # Assigning a Call to a Name (line 329):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_56331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 8), 'int')
        # Processing the call keyword arguments
        kwargs_56332 = {}
        # Getting the type of 'call_assignment_55299' (line 329)
        call_assignment_55299_56329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'call_assignment_55299', False)
        # Obtaining the member '__getitem__' of a type (line 329)
        getitem___56330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 8), call_assignment_55299_56329, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_56333 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___56330, *[int_56331], **kwargs_56332)
        
        # Assigning a type to the variable 'call_assignment_55301' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'call_assignment_55301', getitem___call_result_56333)
        
        # Assigning a Name to a Name (line 329):
        # Getting the type of 'call_assignment_55301' (line 329)
        call_assignment_55301_56334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'call_assignment_55301')
        # Assigning a type to the variable 'py_files' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 17), 'py_files', call_assignment_55301_56334)
        
        
        # Getting the type of 'package' (line 331)
        package_56335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 11), 'package')
        # Getting the type of 'self' (line 331)
        self_56336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 26), 'self')
        # Obtaining the member 'py_modules_dict' of a type (line 331)
        py_modules_dict_56337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 26), self_56336, 'py_modules_dict')
        # Applying the binary operator 'notin' (line 331)
        result_contains_56338 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 11), 'notin', package_56335, py_modules_dict_56337)
        
        # Testing the type of an if condition (line 331)
        if_condition_56339 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 331, 8), result_contains_56338)
        # Assigning a type to the variable 'if_condition_56339' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'if_condition_56339', if_condition_56339)
        # SSA begins for if statement (line 331)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Subscript (line 332):
        
        # Assigning a List to a Subscript (line 332):
        
        # Obtaining an instance of the builtin type 'list' (line 332)
        list_56340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 332)
        
        # Getting the type of 'self' (line 332)
        self_56341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'self')
        # Obtaining the member 'py_modules_dict' of a type (line 332)
        py_modules_dict_56342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 12), self_56341, 'py_modules_dict')
        # Getting the type of 'package' (line 332)
        package_56343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 33), 'package')
        # Storing an element on a container (line 332)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 12), py_modules_dict_56342, (package_56343, list_56340))
        # SSA join for if statement (line 331)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 333):
        
        # Assigning a List to a Name (line 333):
        
        # Obtaining an instance of the builtin type 'list' (line 333)
        list_56344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 333)
        
        # Assigning a type to the variable 'modules' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'modules', list_56344)
        
        # Getting the type of 'py_files' (line 334)
        py_files_56345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 17), 'py_files')
        # Testing the type of a for loop iterable (line 334)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 334, 8), py_files_56345)
        # Getting the type of the for loop variable (line 334)
        for_loop_var_56346 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 334, 8), py_files_56345)
        # Assigning a type to the variable 'f' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'f', for_loop_var_56346)
        # SSA begins for a for statement (line 334)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 335):
        
        # Assigning a Subscript to a Name (line 335):
        
        # Obtaining the type of the subscript
        int_56347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 59), 'int')
        
        # Call to splitext(...): (line 335)
        # Processing the call arguments (line 335)
        
        # Call to basename(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'f' (line 335)
        f_56354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 55), 'f', False)
        # Processing the call keyword arguments (line 335)
        kwargs_56355 = {}
        # Getting the type of 'os' (line 335)
        os_56351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 38), 'os', False)
        # Obtaining the member 'path' of a type (line 335)
        path_56352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 38), os_56351, 'path')
        # Obtaining the member 'basename' of a type (line 335)
        basename_56353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 38), path_56352, 'basename')
        # Calling basename(args, kwargs) (line 335)
        basename_call_result_56356 = invoke(stypy.reporting.localization.Localization(__file__, 335, 38), basename_56353, *[f_56354], **kwargs_56355)
        
        # Processing the call keyword arguments (line 335)
        kwargs_56357 = {}
        # Getting the type of 'os' (line 335)
        os_56348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 21), 'os', False)
        # Obtaining the member 'path' of a type (line 335)
        path_56349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 21), os_56348, 'path')
        # Obtaining the member 'splitext' of a type (line 335)
        splitext_56350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 21), path_56349, 'splitext')
        # Calling splitext(args, kwargs) (line 335)
        splitext_call_result_56358 = invoke(stypy.reporting.localization.Localization(__file__, 335, 21), splitext_56350, *[basename_call_result_56356], **kwargs_56357)
        
        # Obtaining the member '__getitem__' of a type (line 335)
        getitem___56359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 21), splitext_call_result_56358, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 335)
        subscript_call_result_56360 = invoke(stypy.reporting.localization.Localization(__file__, 335, 21), getitem___56359, int_56347)
        
        # Assigning a type to the variable 'module' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'module', subscript_call_result_56360)
        
        # Call to append(...): (line 336)
        # Processing the call arguments (line 336)
        
        # Obtaining an instance of the builtin type 'tuple' (line 336)
        tuple_56363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 336)
        # Adding element type (line 336)
        # Getting the type of 'package' (line 336)
        package_56364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 28), 'package', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 28), tuple_56363, package_56364)
        # Adding element type (line 336)
        # Getting the type of 'module' (line 336)
        module_56365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 37), 'module', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 28), tuple_56363, module_56365)
        # Adding element type (line 336)
        # Getting the type of 'f' (line 336)
        f_56366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 45), 'f', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 28), tuple_56363, f_56366)
        
        # Processing the call keyword arguments (line 336)
        kwargs_56367 = {}
        # Getting the type of 'modules' (line 336)
        modules_56361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'modules', False)
        # Obtaining the member 'append' of a type (line 336)
        append_56362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 12), modules_56361, 'append')
        # Calling append(args, kwargs) (line 336)
        append_call_result_56368 = invoke(stypy.reporting.localization.Localization(__file__, 336, 12), append_56362, *[tuple_56363], **kwargs_56367)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 337)
        self_56369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'self')
        # Obtaining the member 'py_modules_dict' of a type (line 337)
        py_modules_dict_56370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 8), self_56369, 'py_modules_dict')
        
        # Obtaining the type of the subscript
        # Getting the type of 'package' (line 337)
        package_56371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 29), 'package')
        # Getting the type of 'self' (line 337)
        self_56372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'self')
        # Obtaining the member 'py_modules_dict' of a type (line 337)
        py_modules_dict_56373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 8), self_56372, 'py_modules_dict')
        # Obtaining the member '__getitem__' of a type (line 337)
        getitem___56374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 8), py_modules_dict_56373, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 337)
        subscript_call_result_56375 = invoke(stypy.reporting.localization.Localization(__file__, 337, 8), getitem___56374, package_56371)
        
        # Getting the type of 'modules' (line 337)
        modules_56376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 41), 'modules')
        # Applying the binary operator '+=' (line 337)
        result_iadd_56377 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 8), '+=', subscript_call_result_56375, modules_56376)
        # Getting the type of 'self' (line 337)
        self_56378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'self')
        # Obtaining the member 'py_modules_dict' of a type (line 337)
        py_modules_dict_56379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 8), self_56378, 'py_modules_dict')
        # Getting the type of 'package' (line 337)
        package_56380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 29), 'package')
        # Storing an element on a container (line 337)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 8), py_modules_dict_56379, (package_56380, result_iadd_56377))
        
        
        # Assigning a Call to a Tuple (line 339):
        
        # Assigning a Call to a Name:
        
        # Call to filter_h_files(...): (line 339)
        # Processing the call arguments (line 339)
        # Getting the type of 'sources' (line 339)
        sources_56383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 47), 'sources', False)
        # Processing the call keyword arguments (line 339)
        kwargs_56384 = {}
        # Getting the type of 'self' (line 339)
        self_56381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 27), 'self', False)
        # Obtaining the member 'filter_h_files' of a type (line 339)
        filter_h_files_56382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 27), self_56381, 'filter_h_files')
        # Calling filter_h_files(args, kwargs) (line 339)
        filter_h_files_call_result_56385 = invoke(stypy.reporting.localization.Localization(__file__, 339, 27), filter_h_files_56382, *[sources_56383], **kwargs_56384)
        
        # Assigning a type to the variable 'call_assignment_55302' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'call_assignment_55302', filter_h_files_call_result_56385)
        
        # Assigning a Call to a Name (line 339):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_56388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 8), 'int')
        # Processing the call keyword arguments
        kwargs_56389 = {}
        # Getting the type of 'call_assignment_55302' (line 339)
        call_assignment_55302_56386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'call_assignment_55302', False)
        # Obtaining the member '__getitem__' of a type (line 339)
        getitem___56387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 8), call_assignment_55302_56386, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_56390 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___56387, *[int_56388], **kwargs_56389)
        
        # Assigning a type to the variable 'call_assignment_55303' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'call_assignment_55303', getitem___call_result_56390)
        
        # Assigning a Name to a Name (line 339):
        # Getting the type of 'call_assignment_55303' (line 339)
        call_assignment_55303_56391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'call_assignment_55303')
        # Assigning a type to the variable 'sources' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'sources', call_assignment_55303_56391)
        
        # Assigning a Call to a Name (line 339):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_56394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 8), 'int')
        # Processing the call keyword arguments
        kwargs_56395 = {}
        # Getting the type of 'call_assignment_55302' (line 339)
        call_assignment_55302_56392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'call_assignment_55302', False)
        # Obtaining the member '__getitem__' of a type (line 339)
        getitem___56393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 8), call_assignment_55302_56392, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_56396 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___56393, *[int_56394], **kwargs_56395)
        
        # Assigning a type to the variable 'call_assignment_55304' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'call_assignment_55304', getitem___call_result_56396)
        
        # Assigning a Name to a Name (line 339):
        # Getting the type of 'call_assignment_55304' (line 339)
        call_assignment_55304_56397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'call_assignment_55304')
        # Assigning a type to the variable 'h_files' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 17), 'h_files', call_assignment_55304_56397)
        
        # Getting the type of 'h_files' (line 341)
        h_files_56398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 11), 'h_files')
        # Testing the type of an if condition (line 341)
        if_condition_56399 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 341, 8), h_files_56398)
        # Assigning a type to the variable 'if_condition_56399' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'if_condition_56399', if_condition_56399)
        # SSA begins for if statement (line 341)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 342)
        # Processing the call arguments (line 342)
        str_56402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 21), 'str', '%s - nothing done with h_files = %s')
        # Getting the type of 'package' (line 343)
        package_56403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 21), 'package', False)
        # Getting the type of 'h_files' (line 343)
        h_files_56404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 30), 'h_files', False)
        # Processing the call keyword arguments (line 342)
        kwargs_56405 = {}
        # Getting the type of 'log' (line 342)
        log_56400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'log', False)
        # Obtaining the member 'info' of a type (line 342)
        info_56401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 12), log_56400, 'info')
        # Calling info(args, kwargs) (line 342)
        info_call_result_56406 = invoke(stypy.reporting.localization.Localization(__file__, 342, 12), info_56401, *[str_56402, package_56403, h_files_56404], **kwargs_56405)
        
        # SSA join for if statement (line 341)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 347):
        
        # Assigning a Name to a Attribute (line 347):
        # Getting the type of 'sources' (line 347)
        sources_56407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 22), 'sources')
        # Getting the type of 'ext' (line 347)
        ext_56408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'ext')
        # Setting the type of the member 'sources' of a type (line 347)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 8), ext_56408, 'sources', sources_56407)
        
        # ################# End of 'build_extension_sources(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'build_extension_sources' in the type store
        # Getting the type of 'stypy_return_type' (line 309)
        stypy_return_type_56409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_56409)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'build_extension_sources'
        return stypy_return_type_56409


    @norecursion
    def generate_sources(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'generate_sources'
        module_type_store = module_type_store.open_function_context('generate_sources', 349, 4, False)
        # Assigning a type to the variable 'self' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_src.generate_sources.__dict__.__setitem__('stypy_localization', localization)
        build_src.generate_sources.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_src.generate_sources.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_src.generate_sources.__dict__.__setitem__('stypy_function_name', 'build_src.generate_sources')
        build_src.generate_sources.__dict__.__setitem__('stypy_param_names_list', ['sources', 'extension'])
        build_src.generate_sources.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_src.generate_sources.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_src.generate_sources.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_src.generate_sources.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_src.generate_sources.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_src.generate_sources.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_src.generate_sources', ['sources', 'extension'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'generate_sources', localization, ['sources', 'extension'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'generate_sources(...)' code ##################

        
        # Assigning a List to a Name (line 350):
        
        # Assigning a List to a Name (line 350):
        
        # Obtaining an instance of the builtin type 'list' (line 350)
        list_56410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 350)
        
        # Assigning a type to the variable 'new_sources' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'new_sources', list_56410)
        
        # Assigning a List to a Name (line 351):
        
        # Assigning a List to a Name (line 351):
        
        # Obtaining an instance of the builtin type 'list' (line 351)
        list_56411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 351)
        
        # Assigning a type to the variable 'func_sources' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'func_sources', list_56411)
        
        # Getting the type of 'sources' (line 352)
        sources_56412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 22), 'sources')
        # Testing the type of a for loop iterable (line 352)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 352, 8), sources_56412)
        # Getting the type of the for loop variable (line 352)
        for_loop_var_56413 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 352, 8), sources_56412)
        # Assigning a type to the variable 'source' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'source', for_loop_var_56413)
        # SSA begins for a for statement (line 352)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to is_string(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 'source' (line 353)
        source_56415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 25), 'source', False)
        # Processing the call keyword arguments (line 353)
        kwargs_56416 = {}
        # Getting the type of 'is_string' (line 353)
        is_string_56414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 15), 'is_string', False)
        # Calling is_string(args, kwargs) (line 353)
        is_string_call_result_56417 = invoke(stypy.reporting.localization.Localization(__file__, 353, 15), is_string_56414, *[source_56415], **kwargs_56416)
        
        # Testing the type of an if condition (line 353)
        if_condition_56418 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 353, 12), is_string_call_result_56417)
        # Assigning a type to the variable 'if_condition_56418' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 12), 'if_condition_56418', if_condition_56418)
        # SSA begins for if statement (line 353)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 354)
        # Processing the call arguments (line 354)
        # Getting the type of 'source' (line 354)
        source_56421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 35), 'source', False)
        # Processing the call keyword arguments (line 354)
        kwargs_56422 = {}
        # Getting the type of 'new_sources' (line 354)
        new_sources_56419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 16), 'new_sources', False)
        # Obtaining the member 'append' of a type (line 354)
        append_56420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 16), new_sources_56419, 'append')
        # Calling append(args, kwargs) (line 354)
        append_call_result_56423 = invoke(stypy.reporting.localization.Localization(__file__, 354, 16), append_56420, *[source_56421], **kwargs_56422)
        
        # SSA branch for the else part of an if statement (line 353)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'source' (line 356)
        source_56426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 36), 'source', False)
        # Processing the call keyword arguments (line 356)
        kwargs_56427 = {}
        # Getting the type of 'func_sources' (line 356)
        func_sources_56424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 16), 'func_sources', False)
        # Obtaining the member 'append' of a type (line 356)
        append_56425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 16), func_sources_56424, 'append')
        # Calling append(args, kwargs) (line 356)
        append_call_result_56428 = invoke(stypy.reporting.localization.Localization(__file__, 356, 16), append_56425, *[source_56426], **kwargs_56427)
        
        # SSA join for if statement (line 353)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'func_sources' (line 357)
        func_sources_56429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 15), 'func_sources')
        # Applying the 'not' unary operator (line 357)
        result_not__56430 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 11), 'not', func_sources_56429)
        
        # Testing the type of an if condition (line 357)
        if_condition_56431 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 357, 8), result_not__56430)
        # Assigning a type to the variable 'if_condition_56431' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'if_condition_56431', if_condition_56431)
        # SSA begins for if statement (line 357)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'new_sources' (line 358)
        new_sources_56432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 19), 'new_sources')
        # Assigning a type to the variable 'stypy_return_type' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'stypy_return_type', new_sources_56432)
        # SSA join for if statement (line 357)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 359)
        self_56433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 11), 'self')
        # Obtaining the member 'inplace' of a type (line 359)
        inplace_56434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 11), self_56433, 'inplace')
        
        
        # Call to is_sequence(...): (line 359)
        # Processing the call arguments (line 359)
        # Getting the type of 'extension' (line 359)
        extension_56436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 44), 'extension', False)
        # Processing the call keyword arguments (line 359)
        kwargs_56437 = {}
        # Getting the type of 'is_sequence' (line 359)
        is_sequence_56435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 32), 'is_sequence', False)
        # Calling is_sequence(args, kwargs) (line 359)
        is_sequence_call_result_56438 = invoke(stypy.reporting.localization.Localization(__file__, 359, 32), is_sequence_56435, *[extension_56436], **kwargs_56437)
        
        # Applying the 'not' unary operator (line 359)
        result_not__56439 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 28), 'not', is_sequence_call_result_56438)
        
        # Applying the binary operator 'and' (line 359)
        result_and_keyword_56440 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 11), 'and', inplace_56434, result_not__56439)
        
        # Testing the type of an if condition (line 359)
        if_condition_56441 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 359, 8), result_and_keyword_56440)
        # Assigning a type to the variable 'if_condition_56441' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'if_condition_56441', if_condition_56441)
        # SSA begins for if statement (line 359)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 360):
        
        # Assigning a Attribute to a Name (line 360):
        # Getting the type of 'self' (line 360)
        self_56442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 24), 'self')
        # Obtaining the member 'ext_target_dir' of a type (line 360)
        ext_target_dir_56443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 24), self_56442, 'ext_target_dir')
        # Assigning a type to the variable 'build_dir' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'build_dir', ext_target_dir_56443)
        # SSA branch for the else part of an if statement (line 359)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to is_sequence(...): (line 362)
        # Processing the call arguments (line 362)
        # Getting the type of 'extension' (line 362)
        extension_56445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 27), 'extension', False)
        # Processing the call keyword arguments (line 362)
        kwargs_56446 = {}
        # Getting the type of 'is_sequence' (line 362)
        is_sequence_56444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 15), 'is_sequence', False)
        # Calling is_sequence(args, kwargs) (line 362)
        is_sequence_call_result_56447 = invoke(stypy.reporting.localization.Localization(__file__, 362, 15), is_sequence_56444, *[extension_56445], **kwargs_56446)
        
        # Testing the type of an if condition (line 362)
        if_condition_56448 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 362, 12), is_sequence_call_result_56447)
        # Assigning a type to the variable 'if_condition_56448' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'if_condition_56448', if_condition_56448)
        # SSA begins for if statement (line 362)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 363):
        
        # Assigning a Subscript to a Name (line 363):
        
        # Obtaining the type of the subscript
        int_56449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 33), 'int')
        # Getting the type of 'extension' (line 363)
        extension_56450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 23), 'extension')
        # Obtaining the member '__getitem__' of a type (line 363)
        getitem___56451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 23), extension_56450, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 363)
        subscript_call_result_56452 = invoke(stypy.reporting.localization.Localization(__file__, 363, 23), getitem___56451, int_56449)
        
        # Assigning a type to the variable 'name' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 16), 'name', subscript_call_result_56452)
        # SSA branch for the else part of an if statement (line 362)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 368):
        
        # Assigning a Attribute to a Name (line 368):
        # Getting the type of 'extension' (line 368)
        extension_56453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 23), 'extension')
        # Obtaining the member 'name' of a type (line 368)
        name_56454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 23), extension_56453, 'name')
        # Assigning a type to the variable 'name' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 16), 'name', name_56454)
        # SSA join for if statement (line 362)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 372):
        
        # Assigning a Call to a Name (line 372):
        
        # Call to join(...): (line 372)
        
        # Obtaining an instance of the builtin type 'list' (line 372)
        list_56458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 372)
        # Adding element type (line 372)
        # Getting the type of 'self' (line 372)
        self_56459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 40), 'self', False)
        # Obtaining the member 'build_src' of a type (line 372)
        build_src_56460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 40), self_56459, 'build_src')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 39), list_56458, build_src_56460)
        
        
        # Obtaining the type of the subscript
        int_56461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 57), 'int')
        slice_56462 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 373, 40), None, int_56461, None)
        
        # Call to split(...): (line 373)
        # Processing the call arguments (line 373)
        str_56465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 51), 'str', '.')
        # Processing the call keyword arguments (line 373)
        kwargs_56466 = {}
        # Getting the type of 'name' (line 373)
        name_56463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 40), 'name', False)
        # Obtaining the member 'split' of a type (line 373)
        split_56464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 40), name_56463, 'split')
        # Calling split(args, kwargs) (line 373)
        split_call_result_56467 = invoke(stypy.reporting.localization.Localization(__file__, 373, 40), split_56464, *[str_56465], **kwargs_56466)
        
        # Obtaining the member '__getitem__' of a type (line 373)
        getitem___56468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 40), split_call_result_56467, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 373)
        subscript_call_result_56469 = invoke(stypy.reporting.localization.Localization(__file__, 373, 40), getitem___56468, slice_56462)
        
        # Applying the binary operator '+' (line 372)
        result_add_56470 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 39), '+', list_56458, subscript_call_result_56469)
        
        # Processing the call keyword arguments (line 372)
        kwargs_56471 = {}
        # Getting the type of 'os' (line 372)
        os_56455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 372)
        path_56456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 24), os_56455, 'path')
        # Obtaining the member 'join' of a type (line 372)
        join_56457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 24), path_56456, 'join')
        # Calling join(args, kwargs) (line 372)
        join_call_result_56472 = invoke(stypy.reporting.localization.Localization(__file__, 372, 24), join_56457, *[result_add_56470], **kwargs_56471)
        
        # Assigning a type to the variable 'build_dir' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'build_dir', join_call_result_56472)
        # SSA join for if statement (line 359)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to mkpath(...): (line 374)
        # Processing the call arguments (line 374)
        # Getting the type of 'build_dir' (line 374)
        build_dir_56475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 20), 'build_dir', False)
        # Processing the call keyword arguments (line 374)
        kwargs_56476 = {}
        # Getting the type of 'self' (line 374)
        self_56473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 374)
        mkpath_56474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 8), self_56473, 'mkpath')
        # Calling mkpath(args, kwargs) (line 374)
        mkpath_call_result_56477 = invoke(stypy.reporting.localization.Localization(__file__, 374, 8), mkpath_56474, *[build_dir_56475], **kwargs_56476)
        
        
        # Getting the type of 'func_sources' (line 375)
        func_sources_56478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 20), 'func_sources')
        # Testing the type of a for loop iterable (line 375)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 375, 8), func_sources_56478)
        # Getting the type of the for loop variable (line 375)
        for_loop_var_56479 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 375, 8), func_sources_56478)
        # Assigning a type to the variable 'func' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'func', for_loop_var_56479)
        # SSA begins for a for statement (line 375)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 376):
        
        # Assigning a Call to a Name (line 376):
        
        # Call to func(...): (line 376)
        # Processing the call arguments (line 376)
        # Getting the type of 'extension' (line 376)
        extension_56481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 26), 'extension', False)
        # Getting the type of 'build_dir' (line 376)
        build_dir_56482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 37), 'build_dir', False)
        # Processing the call keyword arguments (line 376)
        kwargs_56483 = {}
        # Getting the type of 'func' (line 376)
        func_56480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 21), 'func', False)
        # Calling func(args, kwargs) (line 376)
        func_call_result_56484 = invoke(stypy.reporting.localization.Localization(__file__, 376, 21), func_56480, *[extension_56481, build_dir_56482], **kwargs_56483)
        
        # Assigning a type to the variable 'source' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'source', func_call_result_56484)
        
        
        # Getting the type of 'source' (line 377)
        source_56485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 19), 'source')
        # Applying the 'not' unary operator (line 377)
        result_not__56486 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 15), 'not', source_56485)
        
        # Testing the type of an if condition (line 377)
        if_condition_56487 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 377, 12), result_not__56486)
        # Assigning a type to the variable 'if_condition_56487' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 12), 'if_condition_56487', if_condition_56487)
        # SSA begins for if statement (line 377)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 377)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to is_sequence(...): (line 379)
        # Processing the call arguments (line 379)
        # Getting the type of 'source' (line 379)
        source_56489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 27), 'source', False)
        # Processing the call keyword arguments (line 379)
        kwargs_56490 = {}
        # Getting the type of 'is_sequence' (line 379)
        is_sequence_56488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 15), 'is_sequence', False)
        # Calling is_sequence(args, kwargs) (line 379)
        is_sequence_call_result_56491 = invoke(stypy.reporting.localization.Localization(__file__, 379, 15), is_sequence_56488, *[source_56489], **kwargs_56490)
        
        # Testing the type of an if condition (line 379)
        if_condition_56492 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 379, 12), is_sequence_call_result_56491)
        # Assigning a type to the variable 'if_condition_56492' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 12), 'if_condition_56492', if_condition_56492)
        # SSA begins for if statement (line 379)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'source' (line 380)
        source_56501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 71), 'source')
        comprehension_56502 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 17), source_56501)
        # Assigning a type to the variable 's' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 17), 's', comprehension_56502)
        
        # Call to info(...): (line 380)
        # Processing the call arguments (line 380)
        str_56495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 26), 'str', "  adding '%s' to sources.")
        
        # Obtaining an instance of the builtin type 'tuple' (line 380)
        tuple_56496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 57), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 380)
        # Adding element type (line 380)
        # Getting the type of 's' (line 380)
        s_56497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 57), 's', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 57), tuple_56496, s_56497)
        
        # Applying the binary operator '%' (line 380)
        result_mod_56498 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 26), '%', str_56495, tuple_56496)
        
        # Processing the call keyword arguments (line 380)
        kwargs_56499 = {}
        # Getting the type of 'log' (line 380)
        log_56493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 17), 'log', False)
        # Obtaining the member 'info' of a type (line 380)
        info_56494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 17), log_56493, 'info')
        # Calling info(args, kwargs) (line 380)
        info_call_result_56500 = invoke(stypy.reporting.localization.Localization(__file__, 380, 17), info_56494, *[result_mod_56498], **kwargs_56499)
        
        list_56503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 17), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 17), list_56503, info_call_result_56500)
        
        # Call to extend(...): (line 381)
        # Processing the call arguments (line 381)
        # Getting the type of 'source' (line 381)
        source_56506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 35), 'source', False)
        # Processing the call keyword arguments (line 381)
        kwargs_56507 = {}
        # Getting the type of 'new_sources' (line 381)
        new_sources_56504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 16), 'new_sources', False)
        # Obtaining the member 'extend' of a type (line 381)
        extend_56505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 16), new_sources_56504, 'extend')
        # Calling extend(args, kwargs) (line 381)
        extend_call_result_56508 = invoke(stypy.reporting.localization.Localization(__file__, 381, 16), extend_56505, *[source_56506], **kwargs_56507)
        
        # SSA branch for the else part of an if statement (line 379)
        module_type_store.open_ssa_branch('else')
        
        # Call to info(...): (line 383)
        # Processing the call arguments (line 383)
        str_56511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 25), 'str', "  adding '%s' to sources.")
        
        # Obtaining an instance of the builtin type 'tuple' (line 383)
        tuple_56512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 383)
        # Adding element type (line 383)
        # Getting the type of 'source' (line 383)
        source_56513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 56), 'source', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 56), tuple_56512, source_56513)
        
        # Applying the binary operator '%' (line 383)
        result_mod_56514 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 25), '%', str_56511, tuple_56512)
        
        # Processing the call keyword arguments (line 383)
        kwargs_56515 = {}
        # Getting the type of 'log' (line 383)
        log_56509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 16), 'log', False)
        # Obtaining the member 'info' of a type (line 383)
        info_56510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 16), log_56509, 'info')
        # Calling info(args, kwargs) (line 383)
        info_call_result_56516 = invoke(stypy.reporting.localization.Localization(__file__, 383, 16), info_56510, *[result_mod_56514], **kwargs_56515)
        
        
        # Call to append(...): (line 384)
        # Processing the call arguments (line 384)
        # Getting the type of 'source' (line 384)
        source_56519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 35), 'source', False)
        # Processing the call keyword arguments (line 384)
        kwargs_56520 = {}
        # Getting the type of 'new_sources' (line 384)
        new_sources_56517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 16), 'new_sources', False)
        # Obtaining the member 'append' of a type (line 384)
        append_56518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 16), new_sources_56517, 'append')
        # Calling append(args, kwargs) (line 384)
        append_call_result_56521 = invoke(stypy.reporting.localization.Localization(__file__, 384, 16), append_56518, *[source_56519], **kwargs_56520)
        
        # SSA join for if statement (line 379)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'new_sources' (line 386)
        new_sources_56522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 15), 'new_sources')
        # Assigning a type to the variable 'stypy_return_type' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'stypy_return_type', new_sources_56522)
        
        # ################# End of 'generate_sources(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'generate_sources' in the type store
        # Getting the type of 'stypy_return_type' (line 349)
        stypy_return_type_56523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_56523)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'generate_sources'
        return stypy_return_type_56523


    @norecursion
    def filter_py_files(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'filter_py_files'
        module_type_store = module_type_store.open_function_context('filter_py_files', 388, 4, False)
        # Assigning a type to the variable 'self' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_src.filter_py_files.__dict__.__setitem__('stypy_localization', localization)
        build_src.filter_py_files.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_src.filter_py_files.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_src.filter_py_files.__dict__.__setitem__('stypy_function_name', 'build_src.filter_py_files')
        build_src.filter_py_files.__dict__.__setitem__('stypy_param_names_list', ['sources'])
        build_src.filter_py_files.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_src.filter_py_files.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_src.filter_py_files.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_src.filter_py_files.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_src.filter_py_files.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_src.filter_py_files.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_src.filter_py_files', ['sources'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'filter_py_files', localization, ['sources'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'filter_py_files(...)' code ##################

        
        # Call to filter_files(...): (line 389)
        # Processing the call arguments (line 389)
        # Getting the type of 'sources' (line 389)
        sources_56526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 33), 'sources', False)
        
        # Obtaining an instance of the builtin type 'list' (line 389)
        list_56527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 389)
        # Adding element type (line 389)
        str_56528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 43), 'str', '.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 42), list_56527, str_56528)
        
        # Processing the call keyword arguments (line 389)
        kwargs_56529 = {}
        # Getting the type of 'self' (line 389)
        self_56524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 15), 'self', False)
        # Obtaining the member 'filter_files' of a type (line 389)
        filter_files_56525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 15), self_56524, 'filter_files')
        # Calling filter_files(args, kwargs) (line 389)
        filter_files_call_result_56530 = invoke(stypy.reporting.localization.Localization(__file__, 389, 15), filter_files_56525, *[sources_56526, list_56527], **kwargs_56529)
        
        # Assigning a type to the variable 'stypy_return_type' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'stypy_return_type', filter_files_call_result_56530)
        
        # ################# End of 'filter_py_files(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'filter_py_files' in the type store
        # Getting the type of 'stypy_return_type' (line 388)
        stypy_return_type_56531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_56531)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'filter_py_files'
        return stypy_return_type_56531


    @norecursion
    def filter_h_files(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'filter_h_files'
        module_type_store = module_type_store.open_function_context('filter_h_files', 391, 4, False)
        # Assigning a type to the variable 'self' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_src.filter_h_files.__dict__.__setitem__('stypy_localization', localization)
        build_src.filter_h_files.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_src.filter_h_files.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_src.filter_h_files.__dict__.__setitem__('stypy_function_name', 'build_src.filter_h_files')
        build_src.filter_h_files.__dict__.__setitem__('stypy_param_names_list', ['sources'])
        build_src.filter_h_files.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_src.filter_h_files.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_src.filter_h_files.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_src.filter_h_files.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_src.filter_h_files.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_src.filter_h_files.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_src.filter_h_files', ['sources'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'filter_h_files', localization, ['sources'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'filter_h_files(...)' code ##################

        
        # Call to filter_files(...): (line 392)
        # Processing the call arguments (line 392)
        # Getting the type of 'sources' (line 392)
        sources_56534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 33), 'sources', False)
        
        # Obtaining an instance of the builtin type 'list' (line 392)
        list_56535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 392)
        # Adding element type (line 392)
        str_56536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 43), 'str', '.h')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 42), list_56535, str_56536)
        # Adding element type (line 392)
        str_56537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 49), 'str', '.hpp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 42), list_56535, str_56537)
        # Adding element type (line 392)
        str_56538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 57), 'str', '.inc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 42), list_56535, str_56538)
        
        # Processing the call keyword arguments (line 392)
        kwargs_56539 = {}
        # Getting the type of 'self' (line 392)
        self_56532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 15), 'self', False)
        # Obtaining the member 'filter_files' of a type (line 392)
        filter_files_56533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 15), self_56532, 'filter_files')
        # Calling filter_files(args, kwargs) (line 392)
        filter_files_call_result_56540 = invoke(stypy.reporting.localization.Localization(__file__, 392, 15), filter_files_56533, *[sources_56534, list_56535], **kwargs_56539)
        
        # Assigning a type to the variable 'stypy_return_type' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'stypy_return_type', filter_files_call_result_56540)
        
        # ################# End of 'filter_h_files(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'filter_h_files' in the type store
        # Getting the type of 'stypy_return_type' (line 391)
        stypy_return_type_56541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_56541)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'filter_h_files'
        return stypy_return_type_56541


    @norecursion
    def filter_files(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        
        # Obtaining an instance of the builtin type 'list' (line 394)
        list_56542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 394)
        
        defaults = [list_56542]
        # Create a new context for function 'filter_files'
        module_type_store = module_type_store.open_function_context('filter_files', 394, 4, False)
        # Assigning a type to the variable 'self' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_src.filter_files.__dict__.__setitem__('stypy_localization', localization)
        build_src.filter_files.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_src.filter_files.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_src.filter_files.__dict__.__setitem__('stypy_function_name', 'build_src.filter_files')
        build_src.filter_files.__dict__.__setitem__('stypy_param_names_list', ['sources', 'exts'])
        build_src.filter_files.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_src.filter_files.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_src.filter_files.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_src.filter_files.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_src.filter_files.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_src.filter_files.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_src.filter_files', ['sources', 'exts'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'filter_files', localization, ['sources', 'exts'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'filter_files(...)' code ##################

        
        # Assigning a List to a Name (line 395):
        
        # Assigning a List to a Name (line 395):
        
        # Obtaining an instance of the builtin type 'list' (line 395)
        list_56543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 395)
        
        # Assigning a type to the variable 'new_sources' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'new_sources', list_56543)
        
        # Assigning a List to a Name (line 396):
        
        # Assigning a List to a Name (line 396):
        
        # Obtaining an instance of the builtin type 'list' (line 396)
        list_56544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 396)
        
        # Assigning a type to the variable 'files' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'files', list_56544)
        
        # Getting the type of 'sources' (line 397)
        sources_56545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 22), 'sources')
        # Testing the type of a for loop iterable (line 397)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 397, 8), sources_56545)
        # Getting the type of the for loop variable (line 397)
        for_loop_var_56546 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 397, 8), sources_56545)
        # Assigning a type to the variable 'source' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'source', for_loop_var_56546)
        # SSA begins for a for statement (line 397)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 398):
        
        # Assigning a Call to a Name:
        
        # Call to splitext(...): (line 398)
        # Processing the call arguments (line 398)
        # Getting the type of 'source' (line 398)
        source_56550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 43), 'source', False)
        # Processing the call keyword arguments (line 398)
        kwargs_56551 = {}
        # Getting the type of 'os' (line 398)
        os_56547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 398)
        path_56548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 26), os_56547, 'path')
        # Obtaining the member 'splitext' of a type (line 398)
        splitext_56549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 26), path_56548, 'splitext')
        # Calling splitext(args, kwargs) (line 398)
        splitext_call_result_56552 = invoke(stypy.reporting.localization.Localization(__file__, 398, 26), splitext_56549, *[source_56550], **kwargs_56551)
        
        # Assigning a type to the variable 'call_assignment_55305' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'call_assignment_55305', splitext_call_result_56552)
        
        # Assigning a Call to a Name (line 398):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_56555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 12), 'int')
        # Processing the call keyword arguments
        kwargs_56556 = {}
        # Getting the type of 'call_assignment_55305' (line 398)
        call_assignment_55305_56553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'call_assignment_55305', False)
        # Obtaining the member '__getitem__' of a type (line 398)
        getitem___56554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 12), call_assignment_55305_56553, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_56557 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___56554, *[int_56555], **kwargs_56556)
        
        # Assigning a type to the variable 'call_assignment_55306' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'call_assignment_55306', getitem___call_result_56557)
        
        # Assigning a Name to a Name (line 398):
        # Getting the type of 'call_assignment_55306' (line 398)
        call_assignment_55306_56558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'call_assignment_55306')
        # Assigning a type to the variable 'base' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 13), 'base', call_assignment_55306_56558)
        
        # Assigning a Call to a Name (line 398):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_56561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 12), 'int')
        # Processing the call keyword arguments
        kwargs_56562 = {}
        # Getting the type of 'call_assignment_55305' (line 398)
        call_assignment_55305_56559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'call_assignment_55305', False)
        # Obtaining the member '__getitem__' of a type (line 398)
        getitem___56560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 12), call_assignment_55305_56559, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_56563 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___56560, *[int_56561], **kwargs_56562)
        
        # Assigning a type to the variable 'call_assignment_55307' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'call_assignment_55307', getitem___call_result_56563)
        
        # Assigning a Name to a Name (line 398):
        # Getting the type of 'call_assignment_55307' (line 398)
        call_assignment_55307_56564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'call_assignment_55307')
        # Assigning a type to the variable 'ext' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 19), 'ext', call_assignment_55307_56564)
        
        
        # Getting the type of 'ext' (line 399)
        ext_56565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 15), 'ext')
        # Getting the type of 'exts' (line 399)
        exts_56566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 22), 'exts')
        # Applying the binary operator 'in' (line 399)
        result_contains_56567 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 15), 'in', ext_56565, exts_56566)
        
        # Testing the type of an if condition (line 399)
        if_condition_56568 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 399, 12), result_contains_56567)
        # Assigning a type to the variable 'if_condition_56568' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'if_condition_56568', if_condition_56568)
        # SSA begins for if statement (line 399)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 400)
        # Processing the call arguments (line 400)
        # Getting the type of 'source' (line 400)
        source_56571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 29), 'source', False)
        # Processing the call keyword arguments (line 400)
        kwargs_56572 = {}
        # Getting the type of 'files' (line 400)
        files_56569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 16), 'files', False)
        # Obtaining the member 'append' of a type (line 400)
        append_56570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 16), files_56569, 'append')
        # Calling append(args, kwargs) (line 400)
        append_call_result_56573 = invoke(stypy.reporting.localization.Localization(__file__, 400, 16), append_56570, *[source_56571], **kwargs_56572)
        
        # SSA branch for the else part of an if statement (line 399)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 402)
        # Processing the call arguments (line 402)
        # Getting the type of 'source' (line 402)
        source_56576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 35), 'source', False)
        # Processing the call keyword arguments (line 402)
        kwargs_56577 = {}
        # Getting the type of 'new_sources' (line 402)
        new_sources_56574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 16), 'new_sources', False)
        # Obtaining the member 'append' of a type (line 402)
        append_56575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 16), new_sources_56574, 'append')
        # Calling append(args, kwargs) (line 402)
        append_call_result_56578 = invoke(stypy.reporting.localization.Localization(__file__, 402, 16), append_56575, *[source_56576], **kwargs_56577)
        
        # SSA join for if statement (line 399)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 403)
        tuple_56579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 403)
        # Adding element type (line 403)
        # Getting the type of 'new_sources' (line 403)
        new_sources_56580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 15), 'new_sources')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 15), tuple_56579, new_sources_56580)
        # Adding element type (line 403)
        # Getting the type of 'files' (line 403)
        files_56581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 28), 'files')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 15), tuple_56579, files_56581)
        
        # Assigning a type to the variable 'stypy_return_type' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'stypy_return_type', tuple_56579)
        
        # ################# End of 'filter_files(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'filter_files' in the type store
        # Getting the type of 'stypy_return_type' (line 394)
        stypy_return_type_56582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_56582)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'filter_files'
        return stypy_return_type_56582


    @norecursion
    def template_sources(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'template_sources'
        module_type_store = module_type_store.open_function_context('template_sources', 405, 4, False)
        # Assigning a type to the variable 'self' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_src.template_sources.__dict__.__setitem__('stypy_localization', localization)
        build_src.template_sources.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_src.template_sources.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_src.template_sources.__dict__.__setitem__('stypy_function_name', 'build_src.template_sources')
        build_src.template_sources.__dict__.__setitem__('stypy_param_names_list', ['sources', 'extension'])
        build_src.template_sources.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_src.template_sources.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_src.template_sources.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_src.template_sources.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_src.template_sources.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_src.template_sources.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_src.template_sources', ['sources', 'extension'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'template_sources', localization, ['sources', 'extension'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'template_sources(...)' code ##################

        
        # Assigning a List to a Name (line 406):
        
        # Assigning a List to a Name (line 406):
        
        # Obtaining an instance of the builtin type 'list' (line 406)
        list_56583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 406)
        
        # Assigning a type to the variable 'new_sources' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'new_sources', list_56583)
        
        
        # Call to is_sequence(...): (line 407)
        # Processing the call arguments (line 407)
        # Getting the type of 'extension' (line 407)
        extension_56585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 23), 'extension', False)
        # Processing the call keyword arguments (line 407)
        kwargs_56586 = {}
        # Getting the type of 'is_sequence' (line 407)
        is_sequence_56584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 11), 'is_sequence', False)
        # Calling is_sequence(args, kwargs) (line 407)
        is_sequence_call_result_56587 = invoke(stypy.reporting.localization.Localization(__file__, 407, 11), is_sequence_56584, *[extension_56585], **kwargs_56586)
        
        # Testing the type of an if condition (line 407)
        if_condition_56588 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 407, 8), is_sequence_call_result_56587)
        # Assigning a type to the variable 'if_condition_56588' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'if_condition_56588', if_condition_56588)
        # SSA begins for if statement (line 407)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 408):
        
        # Assigning a Call to a Name (line 408):
        
        # Call to get(...): (line 408)
        # Processing the call arguments (line 408)
        str_56594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 39), 'str', 'depends')
        # Processing the call keyword arguments (line 408)
        kwargs_56595 = {}
        
        # Obtaining the type of the subscript
        int_56589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 32), 'int')
        # Getting the type of 'extension' (line 408)
        extension_56590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 22), 'extension', False)
        # Obtaining the member '__getitem__' of a type (line 408)
        getitem___56591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 22), extension_56590, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 408)
        subscript_call_result_56592 = invoke(stypy.reporting.localization.Localization(__file__, 408, 22), getitem___56591, int_56589)
        
        # Obtaining the member 'get' of a type (line 408)
        get_56593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 22), subscript_call_result_56592, 'get')
        # Calling get(args, kwargs) (line 408)
        get_call_result_56596 = invoke(stypy.reporting.localization.Localization(__file__, 408, 22), get_56593, *[str_56594], **kwargs_56595)
        
        # Assigning a type to the variable 'depends' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'depends', get_call_result_56596)
        
        # Assigning a Call to a Name (line 409):
        
        # Assigning a Call to a Name (line 409):
        
        # Call to get(...): (line 409)
        # Processing the call arguments (line 409)
        str_56602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 44), 'str', 'include_dirs')
        # Processing the call keyword arguments (line 409)
        kwargs_56603 = {}
        
        # Obtaining the type of the subscript
        int_56597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 37), 'int')
        # Getting the type of 'extension' (line 409)
        extension_56598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 27), 'extension', False)
        # Obtaining the member '__getitem__' of a type (line 409)
        getitem___56599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 27), extension_56598, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 409)
        subscript_call_result_56600 = invoke(stypy.reporting.localization.Localization(__file__, 409, 27), getitem___56599, int_56597)
        
        # Obtaining the member 'get' of a type (line 409)
        get_56601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 27), subscript_call_result_56600, 'get')
        # Calling get(args, kwargs) (line 409)
        get_call_result_56604 = invoke(stypy.reporting.localization.Localization(__file__, 409, 27), get_56601, *[str_56602], **kwargs_56603)
        
        # Assigning a type to the variable 'include_dirs' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'include_dirs', get_call_result_56604)
        # SSA branch for the else part of an if statement (line 407)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 411):
        
        # Assigning a Attribute to a Name (line 411):
        # Getting the type of 'extension' (line 411)
        extension_56605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 22), 'extension')
        # Obtaining the member 'depends' of a type (line 411)
        depends_56606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 22), extension_56605, 'depends')
        # Assigning a type to the variable 'depends' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 12), 'depends', depends_56606)
        
        # Assigning a Attribute to a Name (line 412):
        
        # Assigning a Attribute to a Name (line 412):
        # Getting the type of 'extension' (line 412)
        extension_56607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 27), 'extension')
        # Obtaining the member 'include_dirs' of a type (line 412)
        include_dirs_56608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 27), extension_56607, 'include_dirs')
        # Assigning a type to the variable 'include_dirs' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 12), 'include_dirs', include_dirs_56608)
        # SSA join for if statement (line 407)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'sources' (line 413)
        sources_56609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 22), 'sources')
        # Testing the type of a for loop iterable (line 413)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 413, 8), sources_56609)
        # Getting the type of the for loop variable (line 413)
        for_loop_var_56610 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 413, 8), sources_56609)
        # Assigning a type to the variable 'source' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'source', for_loop_var_56610)
        # SSA begins for a for statement (line 413)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 414):
        
        # Assigning a Call to a Name:
        
        # Call to splitext(...): (line 414)
        # Processing the call arguments (line 414)
        # Getting the type of 'source' (line 414)
        source_56614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 43), 'source', False)
        # Processing the call keyword arguments (line 414)
        kwargs_56615 = {}
        # Getting the type of 'os' (line 414)
        os_56611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 414)
        path_56612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 26), os_56611, 'path')
        # Obtaining the member 'splitext' of a type (line 414)
        splitext_56613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 26), path_56612, 'splitext')
        # Calling splitext(args, kwargs) (line 414)
        splitext_call_result_56616 = invoke(stypy.reporting.localization.Localization(__file__, 414, 26), splitext_56613, *[source_56614], **kwargs_56615)
        
        # Assigning a type to the variable 'call_assignment_55308' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'call_assignment_55308', splitext_call_result_56616)
        
        # Assigning a Call to a Name (line 414):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_56619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 12), 'int')
        # Processing the call keyword arguments
        kwargs_56620 = {}
        # Getting the type of 'call_assignment_55308' (line 414)
        call_assignment_55308_56617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'call_assignment_55308', False)
        # Obtaining the member '__getitem__' of a type (line 414)
        getitem___56618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 12), call_assignment_55308_56617, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_56621 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___56618, *[int_56619], **kwargs_56620)
        
        # Assigning a type to the variable 'call_assignment_55309' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'call_assignment_55309', getitem___call_result_56621)
        
        # Assigning a Name to a Name (line 414):
        # Getting the type of 'call_assignment_55309' (line 414)
        call_assignment_55309_56622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'call_assignment_55309')
        # Assigning a type to the variable 'base' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 13), 'base', call_assignment_55309_56622)
        
        # Assigning a Call to a Name (line 414):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_56625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 12), 'int')
        # Processing the call keyword arguments
        kwargs_56626 = {}
        # Getting the type of 'call_assignment_55308' (line 414)
        call_assignment_55308_56623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'call_assignment_55308', False)
        # Obtaining the member '__getitem__' of a type (line 414)
        getitem___56624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 12), call_assignment_55308_56623, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_56627 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___56624, *[int_56625], **kwargs_56626)
        
        # Assigning a type to the variable 'call_assignment_55310' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'call_assignment_55310', getitem___call_result_56627)
        
        # Assigning a Name to a Name (line 414):
        # Getting the type of 'call_assignment_55310' (line 414)
        call_assignment_55310_56628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'call_assignment_55310')
        # Assigning a type to the variable 'ext' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 19), 'ext', call_assignment_55310_56628)
        
        
        # Getting the type of 'ext' (line 415)
        ext_56629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 15), 'ext')
        str_56630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 22), 'str', '.src')
        # Applying the binary operator '==' (line 415)
        result_eq_56631 = python_operator(stypy.reporting.localization.Localization(__file__, 415, 15), '==', ext_56629, str_56630)
        
        # Testing the type of an if condition (line 415)
        if_condition_56632 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 415, 12), result_eq_56631)
        # Assigning a type to the variable 'if_condition_56632' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 12), 'if_condition_56632', if_condition_56632)
        # SSA begins for if statement (line 415)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 416)
        self_56633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 19), 'self')
        # Obtaining the member 'inplace' of a type (line 416)
        inplace_56634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 19), self_56633, 'inplace')
        # Testing the type of an if condition (line 416)
        if_condition_56635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 416, 16), inplace_56634)
        # Assigning a type to the variable 'if_condition_56635' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 16), 'if_condition_56635', if_condition_56635)
        # SSA begins for if statement (line 416)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 417):
        
        # Assigning a Call to a Name (line 417):
        
        # Call to dirname(...): (line 417)
        # Processing the call arguments (line 417)
        # Getting the type of 'base' (line 417)
        base_56639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 49), 'base', False)
        # Processing the call keyword arguments (line 417)
        kwargs_56640 = {}
        # Getting the type of 'os' (line 417)
        os_56636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 33), 'os', False)
        # Obtaining the member 'path' of a type (line 417)
        path_56637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 33), os_56636, 'path')
        # Obtaining the member 'dirname' of a type (line 417)
        dirname_56638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 33), path_56637, 'dirname')
        # Calling dirname(args, kwargs) (line 417)
        dirname_call_result_56641 = invoke(stypy.reporting.localization.Localization(__file__, 417, 33), dirname_56638, *[base_56639], **kwargs_56640)
        
        # Assigning a type to the variable 'target_dir' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 20), 'target_dir', dirname_call_result_56641)
        # SSA branch for the else part of an if statement (line 416)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 419):
        
        # Assigning a Call to a Name (line 419):
        
        # Call to appendpath(...): (line 419)
        # Processing the call arguments (line 419)
        # Getting the type of 'self' (line 419)
        self_56643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 44), 'self', False)
        # Obtaining the member 'build_src' of a type (line 419)
        build_src_56644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 44), self_56643, 'build_src')
        
        # Call to dirname(...): (line 419)
        # Processing the call arguments (line 419)
        # Getting the type of 'base' (line 419)
        base_56648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 76), 'base', False)
        # Processing the call keyword arguments (line 419)
        kwargs_56649 = {}
        # Getting the type of 'os' (line 419)
        os_56645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 60), 'os', False)
        # Obtaining the member 'path' of a type (line 419)
        path_56646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 60), os_56645, 'path')
        # Obtaining the member 'dirname' of a type (line 419)
        dirname_56647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 60), path_56646, 'dirname')
        # Calling dirname(args, kwargs) (line 419)
        dirname_call_result_56650 = invoke(stypy.reporting.localization.Localization(__file__, 419, 60), dirname_56647, *[base_56648], **kwargs_56649)
        
        # Processing the call keyword arguments (line 419)
        kwargs_56651 = {}
        # Getting the type of 'appendpath' (line 419)
        appendpath_56642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 33), 'appendpath', False)
        # Calling appendpath(args, kwargs) (line 419)
        appendpath_call_result_56652 = invoke(stypy.reporting.localization.Localization(__file__, 419, 33), appendpath_56642, *[build_src_56644, dirname_call_result_56650], **kwargs_56651)
        
        # Assigning a type to the variable 'target_dir' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 20), 'target_dir', appendpath_call_result_56652)
        # SSA join for if statement (line 416)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to mkpath(...): (line 420)
        # Processing the call arguments (line 420)
        # Getting the type of 'target_dir' (line 420)
        target_dir_56655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 28), 'target_dir', False)
        # Processing the call keyword arguments (line 420)
        kwargs_56656 = {}
        # Getting the type of 'self' (line 420)
        self_56653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 16), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 420)
        mkpath_56654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 16), self_56653, 'mkpath')
        # Calling mkpath(args, kwargs) (line 420)
        mkpath_call_result_56657 = invoke(stypy.reporting.localization.Localization(__file__, 420, 16), mkpath_56654, *[target_dir_56655], **kwargs_56656)
        
        
        # Assigning a Call to a Name (line 421):
        
        # Assigning a Call to a Name (line 421):
        
        # Call to join(...): (line 421)
        # Processing the call arguments (line 421)
        # Getting the type of 'target_dir' (line 421)
        target_dir_56661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 43), 'target_dir', False)
        
        # Call to basename(...): (line 421)
        # Processing the call arguments (line 421)
        # Getting the type of 'base' (line 421)
        base_56665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 72), 'base', False)
        # Processing the call keyword arguments (line 421)
        kwargs_56666 = {}
        # Getting the type of 'os' (line 421)
        os_56662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 55), 'os', False)
        # Obtaining the member 'path' of a type (line 421)
        path_56663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 55), os_56662, 'path')
        # Obtaining the member 'basename' of a type (line 421)
        basename_56664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 55), path_56663, 'basename')
        # Calling basename(args, kwargs) (line 421)
        basename_call_result_56667 = invoke(stypy.reporting.localization.Localization(__file__, 421, 55), basename_56664, *[base_56665], **kwargs_56666)
        
        # Processing the call keyword arguments (line 421)
        kwargs_56668 = {}
        # Getting the type of 'os' (line 421)
        os_56658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 30), 'os', False)
        # Obtaining the member 'path' of a type (line 421)
        path_56659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 30), os_56658, 'path')
        # Obtaining the member 'join' of a type (line 421)
        join_56660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 30), path_56659, 'join')
        # Calling join(args, kwargs) (line 421)
        join_call_result_56669 = invoke(stypy.reporting.localization.Localization(__file__, 421, 30), join_56660, *[target_dir_56661, basename_call_result_56667], **kwargs_56668)
        
        # Assigning a type to the variable 'target_file' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 16), 'target_file', join_call_result_56669)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 422)
        self_56670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 20), 'self')
        # Obtaining the member 'force' of a type (line 422)
        force_56671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 20), self_56670, 'force')
        
        # Call to newer_group(...): (line 422)
        # Processing the call arguments (line 422)
        
        # Obtaining an instance of the builtin type 'list' (line 422)
        list_56673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 422)
        # Adding element type (line 422)
        # Getting the type of 'source' (line 422)
        source_56674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 47), 'source', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 46), list_56673, source_56674)
        
        # Getting the type of 'depends' (line 422)
        depends_56675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 57), 'depends', False)
        # Applying the binary operator '+' (line 422)
        result_add_56676 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 46), '+', list_56673, depends_56675)
        
        # Getting the type of 'target_file' (line 422)
        target_file_56677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 66), 'target_file', False)
        # Processing the call keyword arguments (line 422)
        kwargs_56678 = {}
        # Getting the type of 'newer_group' (line 422)
        newer_group_56672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 34), 'newer_group', False)
        # Calling newer_group(args, kwargs) (line 422)
        newer_group_call_result_56679 = invoke(stypy.reporting.localization.Localization(__file__, 422, 34), newer_group_56672, *[result_add_56676, target_file_56677], **kwargs_56678)
        
        # Applying the binary operator 'or' (line 422)
        result_or_keyword_56680 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 20), 'or', force_56671, newer_group_call_result_56679)
        
        # Testing the type of an if condition (line 422)
        if_condition_56681 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 422, 16), result_or_keyword_56680)
        # Assigning a type to the variable 'if_condition_56681' (line 422)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 16), 'if_condition_56681', if_condition_56681)
        # SSA begins for if statement (line 422)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to _f_pyf_ext_match(...): (line 423)
        # Processing the call arguments (line 423)
        # Getting the type of 'base' (line 423)
        base_56683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 40), 'base', False)
        # Processing the call keyword arguments (line 423)
        kwargs_56684 = {}
        # Getting the type of '_f_pyf_ext_match' (line 423)
        _f_pyf_ext_match_56682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 23), '_f_pyf_ext_match', False)
        # Calling _f_pyf_ext_match(args, kwargs) (line 423)
        _f_pyf_ext_match_call_result_56685 = invoke(stypy.reporting.localization.Localization(__file__, 423, 23), _f_pyf_ext_match_56682, *[base_56683], **kwargs_56684)
        
        # Testing the type of an if condition (line 423)
        if_condition_56686 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 423, 20), _f_pyf_ext_match_call_result_56685)
        # Assigning a type to the variable 'if_condition_56686' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 20), 'if_condition_56686', if_condition_56686)
        # SSA begins for if statement (line 423)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 424)
        # Processing the call arguments (line 424)
        str_56689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 33), 'str', 'from_template:> %s')
        # Getting the type of 'target_file' (line 424)
        target_file_56690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 57), 'target_file', False)
        # Applying the binary operator '%' (line 424)
        result_mod_56691 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 33), '%', str_56689, target_file_56690)
        
        # Processing the call keyword arguments (line 424)
        kwargs_56692 = {}
        # Getting the type of 'log' (line 424)
        log_56687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 24), 'log', False)
        # Obtaining the member 'info' of a type (line 424)
        info_56688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 24), log_56687, 'info')
        # Calling info(args, kwargs) (line 424)
        info_call_result_56693 = invoke(stypy.reporting.localization.Localization(__file__, 424, 24), info_56688, *[result_mod_56691], **kwargs_56692)
        
        
        # Assigning a Call to a Name (line 425):
        
        # Assigning a Call to a Name (line 425):
        
        # Call to process_f_file(...): (line 425)
        # Processing the call arguments (line 425)
        # Getting the type of 'source' (line 425)
        source_56695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 48), 'source', False)
        # Processing the call keyword arguments (line 425)
        kwargs_56696 = {}
        # Getting the type of 'process_f_file' (line 425)
        process_f_file_56694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 33), 'process_f_file', False)
        # Calling process_f_file(args, kwargs) (line 425)
        process_f_file_call_result_56697 = invoke(stypy.reporting.localization.Localization(__file__, 425, 33), process_f_file_56694, *[source_56695], **kwargs_56696)
        
        # Assigning a type to the variable 'outstr' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 24), 'outstr', process_f_file_call_result_56697)
        # SSA branch for the else part of an if statement (line 423)
        module_type_store.open_ssa_branch('else')
        
        # Call to info(...): (line 427)
        # Processing the call arguments (line 427)
        str_56700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 33), 'str', 'conv_template:> %s')
        # Getting the type of 'target_file' (line 427)
        target_file_56701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 57), 'target_file', False)
        # Applying the binary operator '%' (line 427)
        result_mod_56702 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 33), '%', str_56700, target_file_56701)
        
        # Processing the call keyword arguments (line 427)
        kwargs_56703 = {}
        # Getting the type of 'log' (line 427)
        log_56698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 24), 'log', False)
        # Obtaining the member 'info' of a type (line 427)
        info_56699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 24), log_56698, 'info')
        # Calling info(args, kwargs) (line 427)
        info_call_result_56704 = invoke(stypy.reporting.localization.Localization(__file__, 427, 24), info_56699, *[result_mod_56702], **kwargs_56703)
        
        
        # Assigning a Call to a Name (line 428):
        
        # Assigning a Call to a Name (line 428):
        
        # Call to process_c_file(...): (line 428)
        # Processing the call arguments (line 428)
        # Getting the type of 'source' (line 428)
        source_56706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 48), 'source', False)
        # Processing the call keyword arguments (line 428)
        kwargs_56707 = {}
        # Getting the type of 'process_c_file' (line 428)
        process_c_file_56705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 33), 'process_c_file', False)
        # Calling process_c_file(args, kwargs) (line 428)
        process_c_file_call_result_56708 = invoke(stypy.reporting.localization.Localization(__file__, 428, 33), process_c_file_56705, *[source_56706], **kwargs_56707)
        
        # Assigning a type to the variable 'outstr' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 24), 'outstr', process_c_file_call_result_56708)
        # SSA join for if statement (line 423)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 429):
        
        # Assigning a Call to a Name (line 429):
        
        # Call to open(...): (line 429)
        # Processing the call arguments (line 429)
        # Getting the type of 'target_file' (line 429)
        target_file_56710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 31), 'target_file', False)
        str_56711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 44), 'str', 'w')
        # Processing the call keyword arguments (line 429)
        kwargs_56712 = {}
        # Getting the type of 'open' (line 429)
        open_56709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 26), 'open', False)
        # Calling open(args, kwargs) (line 429)
        open_call_result_56713 = invoke(stypy.reporting.localization.Localization(__file__, 429, 26), open_56709, *[target_file_56710, str_56711], **kwargs_56712)
        
        # Assigning a type to the variable 'fid' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 20), 'fid', open_call_result_56713)
        
        # Call to write(...): (line 430)
        # Processing the call arguments (line 430)
        # Getting the type of 'outstr' (line 430)
        outstr_56716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 30), 'outstr', False)
        # Processing the call keyword arguments (line 430)
        kwargs_56717 = {}
        # Getting the type of 'fid' (line 430)
        fid_56714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 20), 'fid', False)
        # Obtaining the member 'write' of a type (line 430)
        write_56715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 20), fid_56714, 'write')
        # Calling write(args, kwargs) (line 430)
        write_call_result_56718 = invoke(stypy.reporting.localization.Localization(__file__, 430, 20), write_56715, *[outstr_56716], **kwargs_56717)
        
        
        # Call to close(...): (line 431)
        # Processing the call keyword arguments (line 431)
        kwargs_56721 = {}
        # Getting the type of 'fid' (line 431)
        fid_56719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 20), 'fid', False)
        # Obtaining the member 'close' of a type (line 431)
        close_56720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 20), fid_56719, 'close')
        # Calling close(args, kwargs) (line 431)
        close_call_result_56722 = invoke(stypy.reporting.localization.Localization(__file__, 431, 20), close_56720, *[], **kwargs_56721)
        
        # SSA join for if statement (line 422)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to _header_ext_match(...): (line 432)
        # Processing the call arguments (line 432)
        # Getting the type of 'target_file' (line 432)
        target_file_56724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 37), 'target_file', False)
        # Processing the call keyword arguments (line 432)
        kwargs_56725 = {}
        # Getting the type of '_header_ext_match' (line 432)
        _header_ext_match_56723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 19), '_header_ext_match', False)
        # Calling _header_ext_match(args, kwargs) (line 432)
        _header_ext_match_call_result_56726 = invoke(stypy.reporting.localization.Localization(__file__, 432, 19), _header_ext_match_56723, *[target_file_56724], **kwargs_56725)
        
        # Testing the type of an if condition (line 432)
        if_condition_56727 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 432, 16), _header_ext_match_call_result_56726)
        # Assigning a type to the variable 'if_condition_56727' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 16), 'if_condition_56727', if_condition_56727)
        # SSA begins for if statement (line 432)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 433):
        
        # Assigning a Call to a Name (line 433):
        
        # Call to dirname(...): (line 433)
        # Processing the call arguments (line 433)
        # Getting the type of 'target_file' (line 433)
        target_file_56731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 40), 'target_file', False)
        # Processing the call keyword arguments (line 433)
        kwargs_56732 = {}
        # Getting the type of 'os' (line 433)
        os_56728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 433)
        path_56729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 24), os_56728, 'path')
        # Obtaining the member 'dirname' of a type (line 433)
        dirname_56730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 24), path_56729, 'dirname')
        # Calling dirname(args, kwargs) (line 433)
        dirname_call_result_56733 = invoke(stypy.reporting.localization.Localization(__file__, 433, 24), dirname_56730, *[target_file_56731], **kwargs_56732)
        
        # Assigning a type to the variable 'd' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 20), 'd', dirname_call_result_56733)
        
        
        # Getting the type of 'd' (line 434)
        d_56734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 23), 'd')
        # Getting the type of 'include_dirs' (line 434)
        include_dirs_56735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 32), 'include_dirs')
        # Applying the binary operator 'notin' (line 434)
        result_contains_56736 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 23), 'notin', d_56734, include_dirs_56735)
        
        # Testing the type of an if condition (line 434)
        if_condition_56737 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 434, 20), result_contains_56736)
        # Assigning a type to the variable 'if_condition_56737' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 20), 'if_condition_56737', if_condition_56737)
        # SSA begins for if statement (line 434)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 435)
        # Processing the call arguments (line 435)
        str_56740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 33), 'str', "  adding '%s' to include_dirs.")
        # Getting the type of 'd' (line 435)
        d_56741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 69), 'd', False)
        # Applying the binary operator '%' (line 435)
        result_mod_56742 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 33), '%', str_56740, d_56741)
        
        # Processing the call keyword arguments (line 435)
        kwargs_56743 = {}
        # Getting the type of 'log' (line 435)
        log_56738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 24), 'log', False)
        # Obtaining the member 'info' of a type (line 435)
        info_56739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 24), log_56738, 'info')
        # Calling info(args, kwargs) (line 435)
        info_call_result_56744 = invoke(stypy.reporting.localization.Localization(__file__, 435, 24), info_56739, *[result_mod_56742], **kwargs_56743)
        
        
        # Call to append(...): (line 436)
        # Processing the call arguments (line 436)
        # Getting the type of 'd' (line 436)
        d_56747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 44), 'd', False)
        # Processing the call keyword arguments (line 436)
        kwargs_56748 = {}
        # Getting the type of 'include_dirs' (line 436)
        include_dirs_56745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 24), 'include_dirs', False)
        # Obtaining the member 'append' of a type (line 436)
        append_56746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 24), include_dirs_56745, 'append')
        # Calling append(args, kwargs) (line 436)
        append_call_result_56749 = invoke(stypy.reporting.localization.Localization(__file__, 436, 24), append_56746, *[d_56747], **kwargs_56748)
        
        # SSA join for if statement (line 434)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 432)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 437)
        # Processing the call arguments (line 437)
        # Getting the type of 'target_file' (line 437)
        target_file_56752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 35), 'target_file', False)
        # Processing the call keyword arguments (line 437)
        kwargs_56753 = {}
        # Getting the type of 'new_sources' (line 437)
        new_sources_56750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 16), 'new_sources', False)
        # Obtaining the member 'append' of a type (line 437)
        append_56751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 16), new_sources_56750, 'append')
        # Calling append(args, kwargs) (line 437)
        append_call_result_56754 = invoke(stypy.reporting.localization.Localization(__file__, 437, 16), append_56751, *[target_file_56752], **kwargs_56753)
        
        # SSA branch for the else part of an if statement (line 415)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 439)
        # Processing the call arguments (line 439)
        # Getting the type of 'source' (line 439)
        source_56757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 35), 'source', False)
        # Processing the call keyword arguments (line 439)
        kwargs_56758 = {}
        # Getting the type of 'new_sources' (line 439)
        new_sources_56755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 16), 'new_sources', False)
        # Obtaining the member 'append' of a type (line 439)
        append_56756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 16), new_sources_56755, 'append')
        # Calling append(args, kwargs) (line 439)
        append_call_result_56759 = invoke(stypy.reporting.localization.Localization(__file__, 439, 16), append_56756, *[source_56757], **kwargs_56758)
        
        # SSA join for if statement (line 415)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'new_sources' (line 440)
        new_sources_56760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 15), 'new_sources')
        # Assigning a type to the variable 'stypy_return_type' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'stypy_return_type', new_sources_56760)
        
        # ################# End of 'template_sources(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'template_sources' in the type store
        # Getting the type of 'stypy_return_type' (line 405)
        stypy_return_type_56761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_56761)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'template_sources'
        return stypy_return_type_56761


    @norecursion
    def pyrex_sources(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'pyrex_sources'
        module_type_store = module_type_store.open_function_context('pyrex_sources', 442, 4, False)
        # Assigning a type to the variable 'self' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_src.pyrex_sources.__dict__.__setitem__('stypy_localization', localization)
        build_src.pyrex_sources.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_src.pyrex_sources.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_src.pyrex_sources.__dict__.__setitem__('stypy_function_name', 'build_src.pyrex_sources')
        build_src.pyrex_sources.__dict__.__setitem__('stypy_param_names_list', ['sources', 'extension'])
        build_src.pyrex_sources.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_src.pyrex_sources.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_src.pyrex_sources.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_src.pyrex_sources.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_src.pyrex_sources.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_src.pyrex_sources.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_src.pyrex_sources', ['sources', 'extension'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'pyrex_sources', localization, ['sources', 'extension'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'pyrex_sources(...)' code ##################

        str_56762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 8), 'str', 'Pyrex not supported; this remains for Cython support (see below)')
        
        # Assigning a List to a Name (line 444):
        
        # Assigning a List to a Name (line 444):
        
        # Obtaining an instance of the builtin type 'list' (line 444)
        list_56763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 444)
        
        # Assigning a type to the variable 'new_sources' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'new_sources', list_56763)
        
        # Assigning a Subscript to a Name (line 445):
        
        # Assigning a Subscript to a Name (line 445):
        
        # Obtaining the type of the subscript
        int_56764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 45), 'int')
        
        # Call to split(...): (line 445)
        # Processing the call arguments (line 445)
        str_56768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 40), 'str', '.')
        # Processing the call keyword arguments (line 445)
        kwargs_56769 = {}
        # Getting the type of 'extension' (line 445)
        extension_56765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 19), 'extension', False)
        # Obtaining the member 'name' of a type (line 445)
        name_56766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 19), extension_56765, 'name')
        # Obtaining the member 'split' of a type (line 445)
        split_56767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 19), name_56766, 'split')
        # Calling split(args, kwargs) (line 445)
        split_call_result_56770 = invoke(stypy.reporting.localization.Localization(__file__, 445, 19), split_56767, *[str_56768], **kwargs_56769)
        
        # Obtaining the member '__getitem__' of a type (line 445)
        getitem___56771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 19), split_call_result_56770, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 445)
        subscript_call_result_56772 = invoke(stypy.reporting.localization.Localization(__file__, 445, 19), getitem___56771, int_56764)
        
        # Assigning a type to the variable 'ext_name' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'ext_name', subscript_call_result_56772)
        
        # Getting the type of 'sources' (line 446)
        sources_56773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 22), 'sources')
        # Testing the type of a for loop iterable (line 446)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 446, 8), sources_56773)
        # Getting the type of the for loop variable (line 446)
        for_loop_var_56774 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 446, 8), sources_56773)
        # Assigning a type to the variable 'source' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'source', for_loop_var_56774)
        # SSA begins for a for statement (line 446)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 447):
        
        # Assigning a Call to a Name:
        
        # Call to splitext(...): (line 447)
        # Processing the call arguments (line 447)
        # Getting the type of 'source' (line 447)
        source_56778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 43), 'source', False)
        # Processing the call keyword arguments (line 447)
        kwargs_56779 = {}
        # Getting the type of 'os' (line 447)
        os_56775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 447)
        path_56776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 26), os_56775, 'path')
        # Obtaining the member 'splitext' of a type (line 447)
        splitext_56777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 26), path_56776, 'splitext')
        # Calling splitext(args, kwargs) (line 447)
        splitext_call_result_56780 = invoke(stypy.reporting.localization.Localization(__file__, 447, 26), splitext_56777, *[source_56778], **kwargs_56779)
        
        # Assigning a type to the variable 'call_assignment_55311' (line 447)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'call_assignment_55311', splitext_call_result_56780)
        
        # Assigning a Call to a Name (line 447):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_56783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 12), 'int')
        # Processing the call keyword arguments
        kwargs_56784 = {}
        # Getting the type of 'call_assignment_55311' (line 447)
        call_assignment_55311_56781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'call_assignment_55311', False)
        # Obtaining the member '__getitem__' of a type (line 447)
        getitem___56782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 12), call_assignment_55311_56781, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_56785 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___56782, *[int_56783], **kwargs_56784)
        
        # Assigning a type to the variable 'call_assignment_55312' (line 447)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'call_assignment_55312', getitem___call_result_56785)
        
        # Assigning a Name to a Name (line 447):
        # Getting the type of 'call_assignment_55312' (line 447)
        call_assignment_55312_56786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'call_assignment_55312')
        # Assigning a type to the variable 'base' (line 447)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 13), 'base', call_assignment_55312_56786)
        
        # Assigning a Call to a Name (line 447):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_56789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 12), 'int')
        # Processing the call keyword arguments
        kwargs_56790 = {}
        # Getting the type of 'call_assignment_55311' (line 447)
        call_assignment_55311_56787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'call_assignment_55311', False)
        # Obtaining the member '__getitem__' of a type (line 447)
        getitem___56788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 12), call_assignment_55311_56787, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_56791 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___56788, *[int_56789], **kwargs_56790)
        
        # Assigning a type to the variable 'call_assignment_55313' (line 447)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'call_assignment_55313', getitem___call_result_56791)
        
        # Assigning a Name to a Name (line 447):
        # Getting the type of 'call_assignment_55313' (line 447)
        call_assignment_55313_56792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'call_assignment_55313')
        # Assigning a type to the variable 'ext' (line 447)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 19), 'ext', call_assignment_55313_56792)
        
        
        # Getting the type of 'ext' (line 448)
        ext_56793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 15), 'ext')
        str_56794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 22), 'str', '.pyx')
        # Applying the binary operator '==' (line 448)
        result_eq_56795 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 15), '==', ext_56793, str_56794)
        
        # Testing the type of an if condition (line 448)
        if_condition_56796 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 448, 12), result_eq_56795)
        # Assigning a type to the variable 'if_condition_56796' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 12), 'if_condition_56796', if_condition_56796)
        # SSA begins for if statement (line 448)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 449):
        
        # Assigning a Call to a Name (line 449):
        
        # Call to generate_a_pyrex_source(...): (line 449)
        # Processing the call arguments (line 449)
        # Getting the type of 'base' (line 449)
        base_56799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 59), 'base', False)
        # Getting the type of 'ext_name' (line 449)
        ext_name_56800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 65), 'ext_name', False)
        # Getting the type of 'source' (line 450)
        source_56801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 59), 'source', False)
        # Getting the type of 'extension' (line 451)
        extension_56802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 59), 'extension', False)
        # Processing the call keyword arguments (line 449)
        kwargs_56803 = {}
        # Getting the type of 'self' (line 449)
        self_56797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 30), 'self', False)
        # Obtaining the member 'generate_a_pyrex_source' of a type (line 449)
        generate_a_pyrex_source_56798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 30), self_56797, 'generate_a_pyrex_source')
        # Calling generate_a_pyrex_source(args, kwargs) (line 449)
        generate_a_pyrex_source_call_result_56804 = invoke(stypy.reporting.localization.Localization(__file__, 449, 30), generate_a_pyrex_source_56798, *[base_56799, ext_name_56800, source_56801, extension_56802], **kwargs_56803)
        
        # Assigning a type to the variable 'target_file' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 16), 'target_file', generate_a_pyrex_source_call_result_56804)
        
        # Call to append(...): (line 452)
        # Processing the call arguments (line 452)
        # Getting the type of 'target_file' (line 452)
        target_file_56807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 35), 'target_file', False)
        # Processing the call keyword arguments (line 452)
        kwargs_56808 = {}
        # Getting the type of 'new_sources' (line 452)
        new_sources_56805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 16), 'new_sources', False)
        # Obtaining the member 'append' of a type (line 452)
        append_56806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 16), new_sources_56805, 'append')
        # Calling append(args, kwargs) (line 452)
        append_call_result_56809 = invoke(stypy.reporting.localization.Localization(__file__, 452, 16), append_56806, *[target_file_56807], **kwargs_56808)
        
        # SSA branch for the else part of an if statement (line 448)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 454)
        # Processing the call arguments (line 454)
        # Getting the type of 'source' (line 454)
        source_56812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 35), 'source', False)
        # Processing the call keyword arguments (line 454)
        kwargs_56813 = {}
        # Getting the type of 'new_sources' (line 454)
        new_sources_56810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 16), 'new_sources', False)
        # Obtaining the member 'append' of a type (line 454)
        append_56811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 16), new_sources_56810, 'append')
        # Calling append(args, kwargs) (line 454)
        append_call_result_56814 = invoke(stypy.reporting.localization.Localization(__file__, 454, 16), append_56811, *[source_56812], **kwargs_56813)
        
        # SSA join for if statement (line 448)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'new_sources' (line 455)
        new_sources_56815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 15), 'new_sources')
        # Assigning a type to the variable 'stypy_return_type' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'stypy_return_type', new_sources_56815)
        
        # ################# End of 'pyrex_sources(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'pyrex_sources' in the type store
        # Getting the type of 'stypy_return_type' (line 442)
        stypy_return_type_56816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_56816)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'pyrex_sources'
        return stypy_return_type_56816


    @norecursion
    def generate_a_pyrex_source(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'generate_a_pyrex_source'
        module_type_store = module_type_store.open_function_context('generate_a_pyrex_source', 457, 4, False)
        # Assigning a type to the variable 'self' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_src.generate_a_pyrex_source.__dict__.__setitem__('stypy_localization', localization)
        build_src.generate_a_pyrex_source.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_src.generate_a_pyrex_source.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_src.generate_a_pyrex_source.__dict__.__setitem__('stypy_function_name', 'build_src.generate_a_pyrex_source')
        build_src.generate_a_pyrex_source.__dict__.__setitem__('stypy_param_names_list', ['base', 'ext_name', 'source', 'extension'])
        build_src.generate_a_pyrex_source.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_src.generate_a_pyrex_source.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_src.generate_a_pyrex_source.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_src.generate_a_pyrex_source.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_src.generate_a_pyrex_source.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_src.generate_a_pyrex_source.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_src.generate_a_pyrex_source', ['base', 'ext_name', 'source', 'extension'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'generate_a_pyrex_source', localization, ['base', 'ext_name', 'source', 'extension'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'generate_a_pyrex_source(...)' code ##################

        str_56817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, (-1)), 'str', 'Pyrex is not supported, but some projects monkeypatch this method.\n\n        That allows compiling Cython code, see gh-6955.\n        This method will remain here for compatibility reasons.\n        ')
        
        # Obtaining an instance of the builtin type 'list' (line 463)
        list_56818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 463)
        
        # Assigning a type to the variable 'stypy_return_type' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'stypy_return_type', list_56818)
        
        # ################# End of 'generate_a_pyrex_source(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'generate_a_pyrex_source' in the type store
        # Getting the type of 'stypy_return_type' (line 457)
        stypy_return_type_56819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_56819)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'generate_a_pyrex_source'
        return stypy_return_type_56819


    @norecursion
    def f2py_sources(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f2py_sources'
        module_type_store = module_type_store.open_function_context('f2py_sources', 465, 4, False)
        # Assigning a type to the variable 'self' (line 466)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_src.f2py_sources.__dict__.__setitem__('stypy_localization', localization)
        build_src.f2py_sources.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_src.f2py_sources.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_src.f2py_sources.__dict__.__setitem__('stypy_function_name', 'build_src.f2py_sources')
        build_src.f2py_sources.__dict__.__setitem__('stypy_param_names_list', ['sources', 'extension'])
        build_src.f2py_sources.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_src.f2py_sources.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_src.f2py_sources.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_src.f2py_sources.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_src.f2py_sources.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_src.f2py_sources.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_src.f2py_sources', ['sources', 'extension'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f2py_sources', localization, ['sources', 'extension'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f2py_sources(...)' code ##################

        
        # Assigning a List to a Name (line 466):
        
        # Assigning a List to a Name (line 466):
        
        # Obtaining an instance of the builtin type 'list' (line 466)
        list_56820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 466)
        
        # Assigning a type to the variable 'new_sources' (line 466)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'new_sources', list_56820)
        
        # Assigning a List to a Name (line 467):
        
        # Assigning a List to a Name (line 467):
        
        # Obtaining an instance of the builtin type 'list' (line 467)
        list_56821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 467)
        
        # Assigning a type to the variable 'f2py_sources' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'f2py_sources', list_56821)
        
        # Assigning a List to a Name (line 468):
        
        # Assigning a List to a Name (line 468):
        
        # Obtaining an instance of the builtin type 'list' (line 468)
        list_56822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 468)
        
        # Assigning a type to the variable 'f_sources' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'f_sources', list_56822)
        
        # Assigning a Dict to a Name (line 469):
        
        # Assigning a Dict to a Name (line 469):
        
        # Obtaining an instance of the builtin type 'dict' (line 469)
        dict_56823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 23), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 469)
        
        # Assigning a type to the variable 'f2py_targets' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'f2py_targets', dict_56823)
        
        # Assigning a List to a Name (line 470):
        
        # Assigning a List to a Name (line 470):
        
        # Obtaining an instance of the builtin type 'list' (line 470)
        list_56824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 470)
        
        # Assigning a type to the variable 'target_dirs' (line 470)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'target_dirs', list_56824)
        
        # Assigning a Subscript to a Name (line 471):
        
        # Assigning a Subscript to a Name (line 471):
        
        # Obtaining the type of the subscript
        int_56825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 45), 'int')
        
        # Call to split(...): (line 471)
        # Processing the call arguments (line 471)
        str_56829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 40), 'str', '.')
        # Processing the call keyword arguments (line 471)
        kwargs_56830 = {}
        # Getting the type of 'extension' (line 471)
        extension_56826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 19), 'extension', False)
        # Obtaining the member 'name' of a type (line 471)
        name_56827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 19), extension_56826, 'name')
        # Obtaining the member 'split' of a type (line 471)
        split_56828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 19), name_56827, 'split')
        # Calling split(args, kwargs) (line 471)
        split_call_result_56831 = invoke(stypy.reporting.localization.Localization(__file__, 471, 19), split_56828, *[str_56829], **kwargs_56830)
        
        # Obtaining the member '__getitem__' of a type (line 471)
        getitem___56832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 19), split_call_result_56831, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 471)
        subscript_call_result_56833 = invoke(stypy.reporting.localization.Localization(__file__, 471, 19), getitem___56832, int_56825)
        
        # Assigning a type to the variable 'ext_name' (line 471)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'ext_name', subscript_call_result_56833)
        
        # Assigning a Num to a Name (line 472):
        
        # Assigning a Num to a Name (line 472):
        int_56834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 20), 'int')
        # Assigning a type to the variable 'skip_f2py' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'skip_f2py', int_56834)
        
        # Getting the type of 'sources' (line 474)
        sources_56835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 22), 'sources')
        # Testing the type of a for loop iterable (line 474)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 474, 8), sources_56835)
        # Getting the type of the for loop variable (line 474)
        for_loop_var_56836 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 474, 8), sources_56835)
        # Assigning a type to the variable 'source' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'source', for_loop_var_56836)
        # SSA begins for a for statement (line 474)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 475):
        
        # Assigning a Call to a Name:
        
        # Call to splitext(...): (line 475)
        # Processing the call arguments (line 475)
        # Getting the type of 'source' (line 475)
        source_56840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 43), 'source', False)
        # Processing the call keyword arguments (line 475)
        kwargs_56841 = {}
        # Getting the type of 'os' (line 475)
        os_56837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 475)
        path_56838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 26), os_56837, 'path')
        # Obtaining the member 'splitext' of a type (line 475)
        splitext_56839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 26), path_56838, 'splitext')
        # Calling splitext(args, kwargs) (line 475)
        splitext_call_result_56842 = invoke(stypy.reporting.localization.Localization(__file__, 475, 26), splitext_56839, *[source_56840], **kwargs_56841)
        
        # Assigning a type to the variable 'call_assignment_55314' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'call_assignment_55314', splitext_call_result_56842)
        
        # Assigning a Call to a Name (line 475):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_56845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 12), 'int')
        # Processing the call keyword arguments
        kwargs_56846 = {}
        # Getting the type of 'call_assignment_55314' (line 475)
        call_assignment_55314_56843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'call_assignment_55314', False)
        # Obtaining the member '__getitem__' of a type (line 475)
        getitem___56844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 12), call_assignment_55314_56843, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_56847 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___56844, *[int_56845], **kwargs_56846)
        
        # Assigning a type to the variable 'call_assignment_55315' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'call_assignment_55315', getitem___call_result_56847)
        
        # Assigning a Name to a Name (line 475):
        # Getting the type of 'call_assignment_55315' (line 475)
        call_assignment_55315_56848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'call_assignment_55315')
        # Assigning a type to the variable 'base' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 13), 'base', call_assignment_55315_56848)
        
        # Assigning a Call to a Name (line 475):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_56851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 12), 'int')
        # Processing the call keyword arguments
        kwargs_56852 = {}
        # Getting the type of 'call_assignment_55314' (line 475)
        call_assignment_55314_56849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'call_assignment_55314', False)
        # Obtaining the member '__getitem__' of a type (line 475)
        getitem___56850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 12), call_assignment_55314_56849, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_56853 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___56850, *[int_56851], **kwargs_56852)
        
        # Assigning a type to the variable 'call_assignment_55316' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'call_assignment_55316', getitem___call_result_56853)
        
        # Assigning a Name to a Name (line 475):
        # Getting the type of 'call_assignment_55316' (line 475)
        call_assignment_55316_56854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'call_assignment_55316')
        # Assigning a type to the variable 'ext' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 19), 'ext', call_assignment_55316_56854)
        
        
        # Getting the type of 'ext' (line 476)
        ext_56855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 15), 'ext')
        str_56856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 22), 'str', '.pyf')
        # Applying the binary operator '==' (line 476)
        result_eq_56857 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 15), '==', ext_56855, str_56856)
        
        # Testing the type of an if condition (line 476)
        if_condition_56858 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 476, 12), result_eq_56857)
        # Assigning a type to the variable 'if_condition_56858' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 12), 'if_condition_56858', if_condition_56858)
        # SSA begins for if statement (line 476)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 477)
        self_56859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 19), 'self')
        # Obtaining the member 'inplace' of a type (line 477)
        inplace_56860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 19), self_56859, 'inplace')
        # Testing the type of an if condition (line 477)
        if_condition_56861 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 477, 16), inplace_56860)
        # Assigning a type to the variable 'if_condition_56861' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 16), 'if_condition_56861', if_condition_56861)
        # SSA begins for if statement (line 477)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 478):
        
        # Assigning a Call to a Name (line 478):
        
        # Call to dirname(...): (line 478)
        # Processing the call arguments (line 478)
        # Getting the type of 'base' (line 478)
        base_56865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 49), 'base', False)
        # Processing the call keyword arguments (line 478)
        kwargs_56866 = {}
        # Getting the type of 'os' (line 478)
        os_56862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 33), 'os', False)
        # Obtaining the member 'path' of a type (line 478)
        path_56863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 33), os_56862, 'path')
        # Obtaining the member 'dirname' of a type (line 478)
        dirname_56864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 33), path_56863, 'dirname')
        # Calling dirname(args, kwargs) (line 478)
        dirname_call_result_56867 = invoke(stypy.reporting.localization.Localization(__file__, 478, 33), dirname_56864, *[base_56865], **kwargs_56866)
        
        # Assigning a type to the variable 'target_dir' (line 478)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 20), 'target_dir', dirname_call_result_56867)
        # SSA branch for the else part of an if statement (line 477)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 480):
        
        # Assigning a Call to a Name (line 480):
        
        # Call to appendpath(...): (line 480)
        # Processing the call arguments (line 480)
        # Getting the type of 'self' (line 480)
        self_56869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 44), 'self', False)
        # Obtaining the member 'build_src' of a type (line 480)
        build_src_56870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 44), self_56869, 'build_src')
        
        # Call to dirname(...): (line 480)
        # Processing the call arguments (line 480)
        # Getting the type of 'base' (line 480)
        base_56874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 76), 'base', False)
        # Processing the call keyword arguments (line 480)
        kwargs_56875 = {}
        # Getting the type of 'os' (line 480)
        os_56871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 60), 'os', False)
        # Obtaining the member 'path' of a type (line 480)
        path_56872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 60), os_56871, 'path')
        # Obtaining the member 'dirname' of a type (line 480)
        dirname_56873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 60), path_56872, 'dirname')
        # Calling dirname(args, kwargs) (line 480)
        dirname_call_result_56876 = invoke(stypy.reporting.localization.Localization(__file__, 480, 60), dirname_56873, *[base_56874], **kwargs_56875)
        
        # Processing the call keyword arguments (line 480)
        kwargs_56877 = {}
        # Getting the type of 'appendpath' (line 480)
        appendpath_56868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 33), 'appendpath', False)
        # Calling appendpath(args, kwargs) (line 480)
        appendpath_call_result_56878 = invoke(stypy.reporting.localization.Localization(__file__, 480, 33), appendpath_56868, *[build_src_56870, dirname_call_result_56876], **kwargs_56877)
        
        # Assigning a type to the variable 'target_dir' (line 480)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 20), 'target_dir', appendpath_call_result_56878)
        # SSA join for if statement (line 477)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to isfile(...): (line 481)
        # Processing the call arguments (line 481)
        # Getting the type of 'source' (line 481)
        source_56882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 34), 'source', False)
        # Processing the call keyword arguments (line 481)
        kwargs_56883 = {}
        # Getting the type of 'os' (line 481)
        os_56879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 481)
        path_56880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 19), os_56879, 'path')
        # Obtaining the member 'isfile' of a type (line 481)
        isfile_56881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 19), path_56880, 'isfile')
        # Calling isfile(args, kwargs) (line 481)
        isfile_call_result_56884 = invoke(stypy.reporting.localization.Localization(__file__, 481, 19), isfile_56881, *[source_56882], **kwargs_56883)
        
        # Testing the type of an if condition (line 481)
        if_condition_56885 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 481, 16), isfile_call_result_56884)
        # Assigning a type to the variable 'if_condition_56885' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 16), 'if_condition_56885', if_condition_56885)
        # SSA begins for if statement (line 481)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 482):
        
        # Assigning a Call to a Name (line 482):
        
        # Call to get_f2py_modulename(...): (line 482)
        # Processing the call arguments (line 482)
        # Getting the type of 'source' (line 482)
        source_56887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 47), 'source', False)
        # Processing the call keyword arguments (line 482)
        kwargs_56888 = {}
        # Getting the type of 'get_f2py_modulename' (line 482)
        get_f2py_modulename_56886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 27), 'get_f2py_modulename', False)
        # Calling get_f2py_modulename(args, kwargs) (line 482)
        get_f2py_modulename_call_result_56889 = invoke(stypy.reporting.localization.Localization(__file__, 482, 27), get_f2py_modulename_56886, *[source_56887], **kwargs_56888)
        
        # Assigning a type to the variable 'name' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 20), 'name', get_f2py_modulename_call_result_56889)
        
        
        # Getting the type of 'name' (line 483)
        name_56890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 23), 'name')
        # Getting the type of 'ext_name' (line 483)
        ext_name_56891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 31), 'ext_name')
        # Applying the binary operator '!=' (line 483)
        result_ne_56892 = python_operator(stypy.reporting.localization.Localization(__file__, 483, 23), '!=', name_56890, ext_name_56891)
        
        # Testing the type of an if condition (line 483)
        if_condition_56893 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 483, 20), result_ne_56892)
        # Assigning a type to the variable 'if_condition_56893' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 20), 'if_condition_56893', if_condition_56893)
        # SSA begins for if statement (line 483)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to DistutilsSetupError(...): (line 484)
        # Processing the call arguments (line 484)
        str_56895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 50), 'str', 'mismatch of extension names: %s provides %r but expected %r')
        
        # Obtaining an instance of the builtin type 'tuple' (line 486)
        tuple_56896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 486)
        # Adding element type (line 486)
        # Getting the type of 'source' (line 486)
        source_56897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 28), 'source', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 486, 28), tuple_56896, source_56897)
        # Adding element type (line 486)
        # Getting the type of 'name' (line 486)
        name_56898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 36), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 486, 28), tuple_56896, name_56898)
        # Adding element type (line 486)
        # Getting the type of 'ext_name' (line 486)
        ext_name_56899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 42), 'ext_name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 486, 28), tuple_56896, ext_name_56899)
        
        # Applying the binary operator '%' (line 484)
        result_mod_56900 = python_operator(stypy.reporting.localization.Localization(__file__, 484, 50), '%', str_56895, tuple_56896)
        
        # Processing the call keyword arguments (line 484)
        kwargs_56901 = {}
        # Getting the type of 'DistutilsSetupError' (line 484)
        DistutilsSetupError_56894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 30), 'DistutilsSetupError', False)
        # Calling DistutilsSetupError(args, kwargs) (line 484)
        DistutilsSetupError_call_result_56902 = invoke(stypy.reporting.localization.Localization(__file__, 484, 30), DistutilsSetupError_56894, *[result_mod_56900], **kwargs_56901)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 484, 24), DistutilsSetupError_call_result_56902, 'raise parameter', BaseException)
        # SSA join for if statement (line 483)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 487):
        
        # Assigning a Call to a Name (line 487):
        
        # Call to join(...): (line 487)
        # Processing the call arguments (line 487)
        # Getting the type of 'target_dir' (line 487)
        target_dir_56906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 47), 'target_dir', False)
        # Getting the type of 'name' (line 487)
        name_56907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 59), 'name', False)
        str_56908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 64), 'str', 'module.c')
        # Applying the binary operator '+' (line 487)
        result_add_56909 = python_operator(stypy.reporting.localization.Localization(__file__, 487, 59), '+', name_56907, str_56908)
        
        # Processing the call keyword arguments (line 487)
        kwargs_56910 = {}
        # Getting the type of 'os' (line 487)
        os_56903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 34), 'os', False)
        # Obtaining the member 'path' of a type (line 487)
        path_56904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 34), os_56903, 'path')
        # Obtaining the member 'join' of a type (line 487)
        join_56905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 34), path_56904, 'join')
        # Calling join(args, kwargs) (line 487)
        join_call_result_56911 = invoke(stypy.reporting.localization.Localization(__file__, 487, 34), join_56905, *[target_dir_56906, result_add_56909], **kwargs_56910)
        
        # Assigning a type to the variable 'target_file' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 20), 'target_file', join_call_result_56911)
        # SSA branch for the else part of an if statement (line 481)
        module_type_store.open_ssa_branch('else')
        
        # Call to debug(...): (line 489)
        # Processing the call arguments (line 489)
        str_56914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 30), 'str', "  source %s does not exist: skipping f2py'ing.")
        # Getting the type of 'source' (line 490)
        source_56915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 33), 'source', False)
        # Applying the binary operator '%' (line 489)
        result_mod_56916 = python_operator(stypy.reporting.localization.Localization(__file__, 489, 30), '%', str_56914, source_56915)
        
        # Processing the call keyword arguments (line 489)
        kwargs_56917 = {}
        # Getting the type of 'log' (line 489)
        log_56912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 20), 'log', False)
        # Obtaining the member 'debug' of a type (line 489)
        debug_56913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 20), log_56912, 'debug')
        # Calling debug(args, kwargs) (line 489)
        debug_call_result_56918 = invoke(stypy.reporting.localization.Localization(__file__, 489, 20), debug_56913, *[result_mod_56916], **kwargs_56917)
        
        
        # Assigning a Name to a Name (line 491):
        
        # Assigning a Name to a Name (line 491):
        # Getting the type of 'ext_name' (line 491)
        ext_name_56919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 27), 'ext_name')
        # Assigning a type to the variable 'name' (line 491)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 20), 'name', ext_name_56919)
        
        # Assigning a Num to a Name (line 492):
        
        # Assigning a Num to a Name (line 492):
        int_56920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 32), 'int')
        # Assigning a type to the variable 'skip_f2py' (line 492)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 20), 'skip_f2py', int_56920)
        
        # Assigning a Call to a Name (line 493):
        
        # Assigning a Call to a Name (line 493):
        
        # Call to join(...): (line 493)
        # Processing the call arguments (line 493)
        # Getting the type of 'target_dir' (line 493)
        target_dir_56924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 47), 'target_dir', False)
        # Getting the type of 'name' (line 493)
        name_56925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 59), 'name', False)
        str_56926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 64), 'str', 'module.c')
        # Applying the binary operator '+' (line 493)
        result_add_56927 = python_operator(stypy.reporting.localization.Localization(__file__, 493, 59), '+', name_56925, str_56926)
        
        # Processing the call keyword arguments (line 493)
        kwargs_56928 = {}
        # Getting the type of 'os' (line 493)
        os_56921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 34), 'os', False)
        # Obtaining the member 'path' of a type (line 493)
        path_56922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 34), os_56921, 'path')
        # Obtaining the member 'join' of a type (line 493)
        join_56923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 34), path_56922, 'join')
        # Calling join(args, kwargs) (line 493)
        join_call_result_56929 = invoke(stypy.reporting.localization.Localization(__file__, 493, 34), join_56923, *[target_dir_56924, result_add_56927], **kwargs_56928)
        
        # Assigning a type to the variable 'target_file' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 20), 'target_file', join_call_result_56929)
        
        
        
        # Call to isfile(...): (line 494)
        # Processing the call arguments (line 494)
        # Getting the type of 'target_file' (line 494)
        target_file_56933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 42), 'target_file', False)
        # Processing the call keyword arguments (line 494)
        kwargs_56934 = {}
        # Getting the type of 'os' (line 494)
        os_56930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 494)
        path_56931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 27), os_56930, 'path')
        # Obtaining the member 'isfile' of a type (line 494)
        isfile_56932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 27), path_56931, 'isfile')
        # Calling isfile(args, kwargs) (line 494)
        isfile_call_result_56935 = invoke(stypy.reporting.localization.Localization(__file__, 494, 27), isfile_56932, *[target_file_56933], **kwargs_56934)
        
        # Applying the 'not' unary operator (line 494)
        result_not__56936 = python_operator(stypy.reporting.localization.Localization(__file__, 494, 23), 'not', isfile_call_result_56935)
        
        # Testing the type of an if condition (line 494)
        if_condition_56937 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 494, 20), result_not__56936)
        # Assigning a type to the variable 'if_condition_56937' (line 494)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 20), 'if_condition_56937', if_condition_56937)
        # SSA begins for if statement (line 494)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 495)
        # Processing the call arguments (line 495)
        str_56940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 33), 'str', '  target %s does not exist:\n   Assuming %smodule.c was generated with "build_src --inplace" command.')
        
        # Obtaining an instance of the builtin type 'tuple' (line 498)
        tuple_56941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 498)
        # Adding element type (line 498)
        # Getting the type of 'target_file' (line 498)
        target_file_56942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 36), 'target_file', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 498, 36), tuple_56941, target_file_56942)
        # Adding element type (line 498)
        # Getting the type of 'name' (line 498)
        name_56943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 49), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 498, 36), tuple_56941, name_56943)
        
        # Applying the binary operator '%' (line 495)
        result_mod_56944 = python_operator(stypy.reporting.localization.Localization(__file__, 495, 33), '%', str_56940, tuple_56941)
        
        # Processing the call keyword arguments (line 495)
        kwargs_56945 = {}
        # Getting the type of 'log' (line 495)
        log_56938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 24), 'log', False)
        # Obtaining the member 'warn' of a type (line 495)
        warn_56939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 24), log_56938, 'warn')
        # Calling warn(args, kwargs) (line 495)
        warn_call_result_56946 = invoke(stypy.reporting.localization.Localization(__file__, 495, 24), warn_56939, *[result_mod_56944], **kwargs_56945)
        
        
        # Assigning a Call to a Name (line 499):
        
        # Assigning a Call to a Name (line 499):
        
        # Call to dirname(...): (line 499)
        # Processing the call arguments (line 499)
        # Getting the type of 'base' (line 499)
        base_56950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 53), 'base', False)
        # Processing the call keyword arguments (line 499)
        kwargs_56951 = {}
        # Getting the type of 'os' (line 499)
        os_56947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 37), 'os', False)
        # Obtaining the member 'path' of a type (line 499)
        path_56948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 37), os_56947, 'path')
        # Obtaining the member 'dirname' of a type (line 499)
        dirname_56949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 37), path_56948, 'dirname')
        # Calling dirname(args, kwargs) (line 499)
        dirname_call_result_56952 = invoke(stypy.reporting.localization.Localization(__file__, 499, 37), dirname_56949, *[base_56950], **kwargs_56951)
        
        # Assigning a type to the variable 'target_dir' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 24), 'target_dir', dirname_call_result_56952)
        
        # Assigning a Call to a Name (line 500):
        
        # Assigning a Call to a Name (line 500):
        
        # Call to join(...): (line 500)
        # Processing the call arguments (line 500)
        # Getting the type of 'target_dir' (line 500)
        target_dir_56956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 51), 'target_dir', False)
        # Getting the type of 'name' (line 500)
        name_56957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 63), 'name', False)
        str_56958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 68), 'str', 'module.c')
        # Applying the binary operator '+' (line 500)
        result_add_56959 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 63), '+', name_56957, str_56958)
        
        # Processing the call keyword arguments (line 500)
        kwargs_56960 = {}
        # Getting the type of 'os' (line 500)
        os_56953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 38), 'os', False)
        # Obtaining the member 'path' of a type (line 500)
        path_56954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 38), os_56953, 'path')
        # Obtaining the member 'join' of a type (line 500)
        join_56955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 38), path_56954, 'join')
        # Calling join(args, kwargs) (line 500)
        join_call_result_56961 = invoke(stypy.reporting.localization.Localization(__file__, 500, 38), join_56955, *[target_dir_56956, result_add_56959], **kwargs_56960)
        
        # Assigning a type to the variable 'target_file' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 24), 'target_file', join_call_result_56961)
        
        
        
        # Call to isfile(...): (line 501)
        # Processing the call arguments (line 501)
        # Getting the type of 'target_file' (line 501)
        target_file_56965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 46), 'target_file', False)
        # Processing the call keyword arguments (line 501)
        kwargs_56966 = {}
        # Getting the type of 'os' (line 501)
        os_56962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 31), 'os', False)
        # Obtaining the member 'path' of a type (line 501)
        path_56963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 31), os_56962, 'path')
        # Obtaining the member 'isfile' of a type (line 501)
        isfile_56964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 31), path_56963, 'isfile')
        # Calling isfile(args, kwargs) (line 501)
        isfile_call_result_56967 = invoke(stypy.reporting.localization.Localization(__file__, 501, 31), isfile_56964, *[target_file_56965], **kwargs_56966)
        
        # Applying the 'not' unary operator (line 501)
        result_not__56968 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 27), 'not', isfile_call_result_56967)
        
        # Testing the type of an if condition (line 501)
        if_condition_56969 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 501, 24), result_not__56968)
        # Assigning a type to the variable 'if_condition_56969' (line 501)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 24), 'if_condition_56969', if_condition_56969)
        # SSA begins for if statement (line 501)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to DistutilsSetupError(...): (line 502)
        # Processing the call arguments (line 502)
        str_56971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 54), 'str', '%r missing')
        
        # Obtaining an instance of the builtin type 'tuple' (line 502)
        tuple_56972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 70), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 502)
        # Adding element type (line 502)
        # Getting the type of 'target_file' (line 502)
        target_file_56973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 70), 'target_file', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 70), tuple_56972, target_file_56973)
        
        # Applying the binary operator '%' (line 502)
        result_mod_56974 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 54), '%', str_56971, tuple_56972)
        
        # Processing the call keyword arguments (line 502)
        kwargs_56975 = {}
        # Getting the type of 'DistutilsSetupError' (line 502)
        DistutilsSetupError_56970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 34), 'DistutilsSetupError', False)
        # Calling DistutilsSetupError(args, kwargs) (line 502)
        DistutilsSetupError_call_result_56976 = invoke(stypy.reporting.localization.Localization(__file__, 502, 34), DistutilsSetupError_56970, *[result_mod_56974], **kwargs_56975)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 502, 28), DistutilsSetupError_call_result_56976, 'raise parameter', BaseException)
        # SSA join for if statement (line 501)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to info(...): (line 503)
        # Processing the call arguments (line 503)
        str_56979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 33), 'str', '   Yes! Using %r as up-to-date target.')
        # Getting the type of 'target_file' (line 504)
        target_file_56980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 36), 'target_file', False)
        # Applying the binary operator '%' (line 503)
        result_mod_56981 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 33), '%', str_56979, target_file_56980)
        
        # Processing the call keyword arguments (line 503)
        kwargs_56982 = {}
        # Getting the type of 'log' (line 503)
        log_56977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 24), 'log', False)
        # Obtaining the member 'info' of a type (line 503)
        info_56978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 24), log_56977, 'info')
        # Calling info(args, kwargs) (line 503)
        info_call_result_56983 = invoke(stypy.reporting.localization.Localization(__file__, 503, 24), info_56978, *[result_mod_56981], **kwargs_56982)
        
        # SSA join for if statement (line 494)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 481)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 505)
        # Processing the call arguments (line 505)
        # Getting the type of 'target_dir' (line 505)
        target_dir_56986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 35), 'target_dir', False)
        # Processing the call keyword arguments (line 505)
        kwargs_56987 = {}
        # Getting the type of 'target_dirs' (line 505)
        target_dirs_56984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 16), 'target_dirs', False)
        # Obtaining the member 'append' of a type (line 505)
        append_56985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 16), target_dirs_56984, 'append')
        # Calling append(args, kwargs) (line 505)
        append_call_result_56988 = invoke(stypy.reporting.localization.Localization(__file__, 505, 16), append_56985, *[target_dir_56986], **kwargs_56987)
        
        
        # Call to append(...): (line 506)
        # Processing the call arguments (line 506)
        # Getting the type of 'source' (line 506)
        source_56991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 36), 'source', False)
        # Processing the call keyword arguments (line 506)
        kwargs_56992 = {}
        # Getting the type of 'f2py_sources' (line 506)
        f2py_sources_56989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 16), 'f2py_sources', False)
        # Obtaining the member 'append' of a type (line 506)
        append_56990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 16), f2py_sources_56989, 'append')
        # Calling append(args, kwargs) (line 506)
        append_call_result_56993 = invoke(stypy.reporting.localization.Localization(__file__, 506, 16), append_56990, *[source_56991], **kwargs_56992)
        
        
        # Assigning a Name to a Subscript (line 507):
        
        # Assigning a Name to a Subscript (line 507):
        # Getting the type of 'target_file' (line 507)
        target_file_56994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 39), 'target_file')
        # Getting the type of 'f2py_targets' (line 507)
        f2py_targets_56995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 16), 'f2py_targets')
        # Getting the type of 'source' (line 507)
        source_56996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 29), 'source')
        # Storing an element on a container (line 507)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 16), f2py_targets_56995, (source_56996, target_file_56994))
        
        # Call to append(...): (line 508)
        # Processing the call arguments (line 508)
        # Getting the type of 'target_file' (line 508)
        target_file_56999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 35), 'target_file', False)
        # Processing the call keyword arguments (line 508)
        kwargs_57000 = {}
        # Getting the type of 'new_sources' (line 508)
        new_sources_56997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 16), 'new_sources', False)
        # Obtaining the member 'append' of a type (line 508)
        append_56998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 16), new_sources_56997, 'append')
        # Calling append(args, kwargs) (line 508)
        append_call_result_57001 = invoke(stypy.reporting.localization.Localization(__file__, 508, 16), append_56998, *[target_file_56999], **kwargs_57000)
        
        # SSA branch for the else part of an if statement (line 476)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to fortran_ext_match(...): (line 509)
        # Processing the call arguments (line 509)
        # Getting the type of 'ext' (line 509)
        ext_57003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 35), 'ext', False)
        # Processing the call keyword arguments (line 509)
        kwargs_57004 = {}
        # Getting the type of 'fortran_ext_match' (line 509)
        fortran_ext_match_57002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 17), 'fortran_ext_match', False)
        # Calling fortran_ext_match(args, kwargs) (line 509)
        fortran_ext_match_call_result_57005 = invoke(stypy.reporting.localization.Localization(__file__, 509, 17), fortran_ext_match_57002, *[ext_57003], **kwargs_57004)
        
        # Testing the type of an if condition (line 509)
        if_condition_57006 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 509, 17), fortran_ext_match_call_result_57005)
        # Assigning a type to the variable 'if_condition_57006' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 17), 'if_condition_57006', if_condition_57006)
        # SSA begins for if statement (line 509)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 510)
        # Processing the call arguments (line 510)
        # Getting the type of 'source' (line 510)
        source_57009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 33), 'source', False)
        # Processing the call keyword arguments (line 510)
        kwargs_57010 = {}
        # Getting the type of 'f_sources' (line 510)
        f_sources_57007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 16), 'f_sources', False)
        # Obtaining the member 'append' of a type (line 510)
        append_57008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 16), f_sources_57007, 'append')
        # Calling append(args, kwargs) (line 510)
        append_call_result_57011 = invoke(stypy.reporting.localization.Localization(__file__, 510, 16), append_57008, *[source_57009], **kwargs_57010)
        
        # SSA branch for the else part of an if statement (line 509)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 512)
        # Processing the call arguments (line 512)
        # Getting the type of 'source' (line 512)
        source_57014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 35), 'source', False)
        # Processing the call keyword arguments (line 512)
        kwargs_57015 = {}
        # Getting the type of 'new_sources' (line 512)
        new_sources_57012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 16), 'new_sources', False)
        # Obtaining the member 'append' of a type (line 512)
        append_57013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 16), new_sources_57012, 'append')
        # Calling append(args, kwargs) (line 512)
        append_call_result_57016 = invoke(stypy.reporting.localization.Localization(__file__, 512, 16), append_57013, *[source_57014], **kwargs_57015)
        
        # SSA join for if statement (line 509)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 476)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'f2py_sources' (line 514)
        f2py_sources_57017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 16), 'f2py_sources')
        # Getting the type of 'f_sources' (line 514)
        f_sources_57018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 32), 'f_sources')
        # Applying the binary operator 'or' (line 514)
        result_or_keyword_57019 = python_operator(stypy.reporting.localization.Localization(__file__, 514, 16), 'or', f2py_sources_57017, f_sources_57018)
        
        # Applying the 'not' unary operator (line 514)
        result_not__57020 = python_operator(stypy.reporting.localization.Localization(__file__, 514, 11), 'not', result_or_keyword_57019)
        
        # Testing the type of an if condition (line 514)
        if_condition_57021 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 514, 8), result_not__57020)
        # Assigning a type to the variable 'if_condition_57021' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 8), 'if_condition_57021', if_condition_57021)
        # SSA begins for if statement (line 514)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'new_sources' (line 515)
        new_sources_57022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 19), 'new_sources')
        # Assigning a type to the variable 'stypy_return_type' (line 515)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 12), 'stypy_return_type', new_sources_57022)
        # SSA join for if statement (line 514)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'target_dirs' (line 517)
        target_dirs_57023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 17), 'target_dirs')
        # Testing the type of a for loop iterable (line 517)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 517, 8), target_dirs_57023)
        # Getting the type of the for loop variable (line 517)
        for_loop_var_57024 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 517, 8), target_dirs_57023)
        # Assigning a type to the variable 'd' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'd', for_loop_var_57024)
        # SSA begins for a for statement (line 517)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to mkpath(...): (line 518)
        # Processing the call arguments (line 518)
        # Getting the type of 'd' (line 518)
        d_57027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 24), 'd', False)
        # Processing the call keyword arguments (line 518)
        kwargs_57028 = {}
        # Getting the type of 'self' (line 518)
        self_57025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 12), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 518)
        mkpath_57026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 12), self_57025, 'mkpath')
        # Calling mkpath(args, kwargs) (line 518)
        mkpath_call_result_57029 = invoke(stypy.reporting.localization.Localization(__file__, 518, 12), mkpath_57026, *[d_57027], **kwargs_57028)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 520):
        
        # Assigning a BinOp to a Name (line 520):
        # Getting the type of 'extension' (line 520)
        extension_57030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 23), 'extension')
        # Obtaining the member 'f2py_options' of a type (line 520)
        f2py_options_57031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 23), extension_57030, 'f2py_options')
        # Getting the type of 'self' (line 520)
        self_57032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 48), 'self')
        # Obtaining the member 'f2py_opts' of a type (line 520)
        f2py_opts_57033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 48), self_57032, 'f2py_opts')
        # Applying the binary operator '+' (line 520)
        result_add_57034 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 23), '+', f2py_options_57031, f2py_opts_57033)
        
        # Assigning a type to the variable 'f2py_options' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'f2py_options', result_add_57034)
        
        # Getting the type of 'self' (line 522)
        self_57035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 11), 'self')
        # Obtaining the member 'distribution' of a type (line 522)
        distribution_57036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 11), self_57035, 'distribution')
        # Obtaining the member 'libraries' of a type (line 522)
        libraries_57037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 11), distribution_57036, 'libraries')
        # Testing the type of an if condition (line 522)
        if_condition_57038 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 522, 8), libraries_57037)
        # Assigning a type to the variable 'if_condition_57038' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'if_condition_57038', if_condition_57038)
        # SSA begins for if statement (line 522)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 523)
        self_57039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 36), 'self')
        # Obtaining the member 'distribution' of a type (line 523)
        distribution_57040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 36), self_57039, 'distribution')
        # Obtaining the member 'libraries' of a type (line 523)
        libraries_57041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 36), distribution_57040, 'libraries')
        # Testing the type of a for loop iterable (line 523)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 523, 12), libraries_57041)
        # Getting the type of the for loop variable (line 523)
        for_loop_var_57042 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 523, 12), libraries_57041)
        # Assigning a type to the variable 'name' (line 523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 12), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 12), for_loop_var_57042))
        # Assigning a type to the variable 'build_info' (line 523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 12), 'build_info', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 12), for_loop_var_57042))
        # SSA begins for a for statement (line 523)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'name' (line 524)
        name_57043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 19), 'name')
        # Getting the type of 'extension' (line 524)
        extension_57044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 27), 'extension')
        # Obtaining the member 'libraries' of a type (line 524)
        libraries_57045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 27), extension_57044, 'libraries')
        # Applying the binary operator 'in' (line 524)
        result_contains_57046 = python_operator(stypy.reporting.localization.Localization(__file__, 524, 19), 'in', name_57043, libraries_57045)
        
        # Testing the type of an if condition (line 524)
        if_condition_57047 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 524, 16), result_contains_57046)
        # Assigning a type to the variable 'if_condition_57047' (line 524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 16), 'if_condition_57047', if_condition_57047)
        # SSA begins for if statement (line 524)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 525)
        # Processing the call arguments (line 525)
        
        # Call to get(...): (line 525)
        # Processing the call arguments (line 525)
        str_57052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 55), 'str', 'f2py_options')
        
        # Obtaining an instance of the builtin type 'list' (line 525)
        list_57053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 71), 'list')
        # Adding type elements to the builtin type 'list' instance (line 525)
        
        # Processing the call keyword arguments (line 525)
        kwargs_57054 = {}
        # Getting the type of 'build_info' (line 525)
        build_info_57050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 40), 'build_info', False)
        # Obtaining the member 'get' of a type (line 525)
        get_57051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 40), build_info_57050, 'get')
        # Calling get(args, kwargs) (line 525)
        get_call_result_57055 = invoke(stypy.reporting.localization.Localization(__file__, 525, 40), get_57051, *[str_57052, list_57053], **kwargs_57054)
        
        # Processing the call keyword arguments (line 525)
        kwargs_57056 = {}
        # Getting the type of 'f2py_options' (line 525)
        f2py_options_57048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 20), 'f2py_options', False)
        # Obtaining the member 'extend' of a type (line 525)
        extend_57049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 20), f2py_options_57048, 'extend')
        # Calling extend(args, kwargs) (line 525)
        extend_call_result_57057 = invoke(stypy.reporting.localization.Localization(__file__, 525, 20), extend_57049, *[get_call_result_57055], **kwargs_57056)
        
        # SSA join for if statement (line 524)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 522)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to info(...): (line 527)
        # Processing the call arguments (line 527)
        str_57060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 17), 'str', 'f2py options: %s')
        # Getting the type of 'f2py_options' (line 527)
        f2py_options_57061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 39), 'f2py_options', False)
        # Applying the binary operator '%' (line 527)
        result_mod_57062 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 17), '%', str_57060, f2py_options_57061)
        
        # Processing the call keyword arguments (line 527)
        kwargs_57063 = {}
        # Getting the type of 'log' (line 527)
        log_57058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 527)
        info_57059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 8), log_57058, 'info')
        # Calling info(args, kwargs) (line 527)
        info_call_result_57064 = invoke(stypy.reporting.localization.Localization(__file__, 527, 8), info_57059, *[result_mod_57062], **kwargs_57063)
        
        
        # Getting the type of 'f2py_sources' (line 529)
        f2py_sources_57065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 11), 'f2py_sources')
        # Testing the type of an if condition (line 529)
        if_condition_57066 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 529, 8), f2py_sources_57065)
        # Assigning a type to the variable 'if_condition_57066' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 8), 'if_condition_57066', if_condition_57066)
        # SSA begins for if statement (line 529)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to len(...): (line 530)
        # Processing the call arguments (line 530)
        # Getting the type of 'f2py_sources' (line 530)
        f2py_sources_57068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 19), 'f2py_sources', False)
        # Processing the call keyword arguments (line 530)
        kwargs_57069 = {}
        # Getting the type of 'len' (line 530)
        len_57067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 15), 'len', False)
        # Calling len(args, kwargs) (line 530)
        len_call_result_57070 = invoke(stypy.reporting.localization.Localization(__file__, 530, 15), len_57067, *[f2py_sources_57068], **kwargs_57069)
        
        int_57071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 36), 'int')
        # Applying the binary operator '!=' (line 530)
        result_ne_57072 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 15), '!=', len_call_result_57070, int_57071)
        
        # Testing the type of an if condition (line 530)
        if_condition_57073 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 530, 12), result_ne_57072)
        # Assigning a type to the variable 'if_condition_57073' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'if_condition_57073', if_condition_57073)
        # SSA begins for if statement (line 530)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to DistutilsSetupError(...): (line 531)
        # Processing the call arguments (line 531)
        str_57075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 20), 'str', 'only one .pyf file is allowed per extension module but got more: %r')
        
        # Obtaining an instance of the builtin type 'tuple' (line 533)
        tuple_57076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 533)
        # Adding element type (line 533)
        # Getting the type of 'f2py_sources' (line 533)
        f2py_sources_57077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 35), 'f2py_sources', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 35), tuple_57076, f2py_sources_57077)
        
        # Applying the binary operator '%' (line 532)
        result_mod_57078 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 20), '%', str_57075, tuple_57076)
        
        # Processing the call keyword arguments (line 531)
        kwargs_57079 = {}
        # Getting the type of 'DistutilsSetupError' (line 531)
        DistutilsSetupError_57074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 22), 'DistutilsSetupError', False)
        # Calling DistutilsSetupError(args, kwargs) (line 531)
        DistutilsSetupError_call_result_57080 = invoke(stypy.reporting.localization.Localization(__file__, 531, 22), DistutilsSetupError_57074, *[result_mod_57078], **kwargs_57079)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 531, 16), DistutilsSetupError_call_result_57080, 'raise parameter', BaseException)
        # SSA join for if statement (line 530)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 534):
        
        # Assigning a Subscript to a Name (line 534):
        
        # Obtaining the type of the subscript
        int_57081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 34), 'int')
        # Getting the type of 'f2py_sources' (line 534)
        f2py_sources_57082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 21), 'f2py_sources')
        # Obtaining the member '__getitem__' of a type (line 534)
        getitem___57083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 21), f2py_sources_57082, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 534)
        subscript_call_result_57084 = invoke(stypy.reporting.localization.Localization(__file__, 534, 21), getitem___57083, int_57081)
        
        # Assigning a type to the variable 'source' (line 534)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 12), 'source', subscript_call_result_57084)
        
        # Assigning a Subscript to a Name (line 535):
        
        # Assigning a Subscript to a Name (line 535):
        
        # Obtaining the type of the subscript
        # Getting the type of 'source' (line 535)
        source_57085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 39), 'source')
        # Getting the type of 'f2py_targets' (line 535)
        f2py_targets_57086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 26), 'f2py_targets')
        # Obtaining the member '__getitem__' of a type (line 535)
        getitem___57087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 26), f2py_targets_57086, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 535)
        subscript_call_result_57088 = invoke(stypy.reporting.localization.Localization(__file__, 535, 26), getitem___57087, source_57085)
        
        # Assigning a type to the variable 'target_file' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 12), 'target_file', subscript_call_result_57088)
        
        # Assigning a BoolOp to a Name (line 536):
        
        # Assigning a BoolOp to a Name (line 536):
        
        # Evaluating a boolean operation
        
        # Call to dirname(...): (line 536)
        # Processing the call arguments (line 536)
        # Getting the type of 'target_file' (line 536)
        target_file_57092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 41), 'target_file', False)
        # Processing the call keyword arguments (line 536)
        kwargs_57093 = {}
        # Getting the type of 'os' (line 536)
        os_57089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 536)
        path_57090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 25), os_57089, 'path')
        # Obtaining the member 'dirname' of a type (line 536)
        dirname_57091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 25), path_57090, 'dirname')
        # Calling dirname(args, kwargs) (line 536)
        dirname_call_result_57094 = invoke(stypy.reporting.localization.Localization(__file__, 536, 25), dirname_57091, *[target_file_57092], **kwargs_57093)
        
        str_57095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 57), 'str', '.')
        # Applying the binary operator 'or' (line 536)
        result_or_keyword_57096 = python_operator(stypy.reporting.localization.Localization(__file__, 536, 25), 'or', dirname_call_result_57094, str_57095)
        
        # Assigning a type to the variable 'target_dir' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 12), 'target_dir', result_or_keyword_57096)
        
        # Assigning a BinOp to a Name (line 537):
        
        # Assigning a BinOp to a Name (line 537):
        
        # Obtaining an instance of the builtin type 'list' (line 537)
        list_57097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 537)
        # Adding element type (line 537)
        # Getting the type of 'source' (line 537)
        source_57098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 23), 'source')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 22), list_57097, source_57098)
        
        # Getting the type of 'extension' (line 537)
        extension_57099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 33), 'extension')
        # Obtaining the member 'depends' of a type (line 537)
        depends_57100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 33), extension_57099, 'depends')
        # Applying the binary operator '+' (line 537)
        result_add_57101 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 22), '+', list_57097, depends_57100)
        
        # Assigning a type to the variable 'depends' (line 537)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 12), 'depends', result_add_57101)
        
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 538)
        self_57102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 16), 'self')
        # Obtaining the member 'force' of a type (line 538)
        force_57103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 16), self_57102, 'force')
        
        # Call to newer_group(...): (line 538)
        # Processing the call arguments (line 538)
        # Getting the type of 'depends' (line 538)
        depends_57105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 42), 'depends', False)
        # Getting the type of 'target_file' (line 538)
        target_file_57106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 51), 'target_file', False)
        str_57107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 64), 'str', 'newer')
        # Processing the call keyword arguments (line 538)
        kwargs_57108 = {}
        # Getting the type of 'newer_group' (line 538)
        newer_group_57104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 30), 'newer_group', False)
        # Calling newer_group(args, kwargs) (line 538)
        newer_group_call_result_57109 = invoke(stypy.reporting.localization.Localization(__file__, 538, 30), newer_group_57104, *[depends_57105, target_file_57106, str_57107], **kwargs_57108)
        
        # Applying the binary operator 'or' (line 538)
        result_or_keyword_57110 = python_operator(stypy.reporting.localization.Localization(__file__, 538, 16), 'or', force_57103, newer_group_call_result_57109)
        
        
        # Getting the type of 'skip_f2py' (line 539)
        skip_f2py_57111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 27), 'skip_f2py')
        # Applying the 'not' unary operator (line 539)
        result_not__57112 = python_operator(stypy.reporting.localization.Localization(__file__, 539, 23), 'not', skip_f2py_57111)
        
        # Applying the binary operator 'and' (line 538)
        result_and_keyword_57113 = python_operator(stypy.reporting.localization.Localization(__file__, 538, 15), 'and', result_or_keyword_57110, result_not__57112)
        
        # Testing the type of an if condition (line 538)
        if_condition_57114 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 538, 12), result_and_keyword_57113)
        # Assigning a type to the variable 'if_condition_57114' (line 538)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 12), 'if_condition_57114', if_condition_57114)
        # SSA begins for if statement (line 538)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 540)
        # Processing the call arguments (line 540)
        str_57117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 25), 'str', 'f2py: %s')
        # Getting the type of 'source' (line 540)
        source_57118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 39), 'source', False)
        # Applying the binary operator '%' (line 540)
        result_mod_57119 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 25), '%', str_57117, source_57118)
        
        # Processing the call keyword arguments (line 540)
        kwargs_57120 = {}
        # Getting the type of 'log' (line 540)
        log_57115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 16), 'log', False)
        # Obtaining the member 'info' of a type (line 540)
        info_57116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 16), log_57115, 'info')
        # Calling info(args, kwargs) (line 540)
        info_call_result_57121 = invoke(stypy.reporting.localization.Localization(__file__, 540, 16), info_57116, *[result_mod_57119], **kwargs_57120)
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 541, 16))
        
        # 'import numpy.f2py' statement (line 541)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
        import_57122 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 541, 16), 'numpy.f2py')

        if (type(import_57122) is not StypyTypeError):

            if (import_57122 != 'pyd_module'):
                __import__(import_57122)
                sys_modules_57123 = sys.modules[import_57122]
                import_module(stypy.reporting.localization.Localization(__file__, 541, 16), 'numpy.f2py', sys_modules_57123.module_type_store, module_type_store)
            else:
                import numpy.f2py

                import_module(stypy.reporting.localization.Localization(__file__, 541, 16), 'numpy.f2py', numpy.f2py, module_type_store)

        else:
            # Assigning a type to the variable 'numpy.f2py' (line 541)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 16), 'numpy.f2py', import_57122)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')
        
        
        # Call to run_main(...): (line 542)
        # Processing the call arguments (line 542)
        # Getting the type of 'f2py_options' (line 542)
        f2py_options_57127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 36), 'f2py_options', False)
        
        # Obtaining an instance of the builtin type 'list' (line 543)
        list_57128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 543)
        # Adding element type (line 543)
        str_57129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 39), 'str', '--build-dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 543, 38), list_57128, str_57129)
        # Adding element type (line 543)
        # Getting the type of 'target_dir' (line 543)
        target_dir_57130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 54), 'target_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 543, 38), list_57128, target_dir_57130)
        # Adding element type (line 543)
        # Getting the type of 'source' (line 543)
        source_57131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 66), 'source', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 543, 38), list_57128, source_57131)
        
        # Applying the binary operator '+' (line 542)
        result_add_57132 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 36), '+', f2py_options_57127, list_57128)
        
        # Processing the call keyword arguments (line 542)
        kwargs_57133 = {}
        # Getting the type of 'numpy' (line 542)
        numpy_57124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 16), 'numpy', False)
        # Obtaining the member 'f2py' of a type (line 542)
        f2py_57125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 16), numpy_57124, 'f2py')
        # Obtaining the member 'run_main' of a type (line 542)
        run_main_57126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 16), f2py_57125, 'run_main')
        # Calling run_main(args, kwargs) (line 542)
        run_main_call_result_57134 = invoke(stypy.reporting.localization.Localization(__file__, 542, 16), run_main_57126, *[result_add_57132], **kwargs_57133)
        
        # SSA branch for the else part of an if statement (line 538)
        module_type_store.open_ssa_branch('else')
        
        # Call to debug(...): (line 545)
        # Processing the call arguments (line 545)
        str_57137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 26), 'str', "  skipping '%s' f2py interface (up-to-date)")
        # Getting the type of 'source' (line 545)
        source_57138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 75), 'source', False)
        # Applying the binary operator '%' (line 545)
        result_mod_57139 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 26), '%', str_57137, source_57138)
        
        # Processing the call keyword arguments (line 545)
        kwargs_57140 = {}
        # Getting the type of 'log' (line 545)
        log_57135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 16), 'log', False)
        # Obtaining the member 'debug' of a type (line 545)
        debug_57136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 16), log_57135, 'debug')
        # Calling debug(args, kwargs) (line 545)
        debug_call_result_57141 = invoke(stypy.reporting.localization.Localization(__file__, 545, 16), debug_57136, *[result_mod_57139], **kwargs_57140)
        
        # SSA join for if statement (line 538)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 529)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to is_sequence(...): (line 548)
        # Processing the call arguments (line 548)
        # Getting the type of 'extension' (line 548)
        extension_57143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 27), 'extension', False)
        # Processing the call keyword arguments (line 548)
        kwargs_57144 = {}
        # Getting the type of 'is_sequence' (line 548)
        is_sequence_57142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 15), 'is_sequence', False)
        # Calling is_sequence(args, kwargs) (line 548)
        is_sequence_call_result_57145 = invoke(stypy.reporting.localization.Localization(__file__, 548, 15), is_sequence_57142, *[extension_57143], **kwargs_57144)
        
        # Testing the type of an if condition (line 548)
        if_condition_57146 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 548, 12), is_sequence_call_result_57145)
        # Assigning a type to the variable 'if_condition_57146' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 12), 'if_condition_57146', if_condition_57146)
        # SSA begins for if statement (line 548)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 549):
        
        # Assigning a Subscript to a Name (line 549):
        
        # Obtaining the type of the subscript
        int_57147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 33), 'int')
        # Getting the type of 'extension' (line 549)
        extension_57148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 23), 'extension')
        # Obtaining the member '__getitem__' of a type (line 549)
        getitem___57149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 23), extension_57148, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 549)
        subscript_call_result_57150 = invoke(stypy.reporting.localization.Localization(__file__, 549, 23), getitem___57149, int_57147)
        
        # Assigning a type to the variable 'name' (line 549)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 16), 'name', subscript_call_result_57150)
        # SSA branch for the else part of an if statement (line 548)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 550):
        
        # Assigning a Attribute to a Name (line 550):
        # Getting the type of 'extension' (line 550)
        extension_57151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 25), 'extension')
        # Obtaining the member 'name' of a type (line 550)
        name_57152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 25), extension_57151, 'name')
        # Assigning a type to the variable 'name' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 18), 'name', name_57152)
        # SSA join for if statement (line 548)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 551):
        
        # Assigning a Call to a Name (line 551):
        
        # Call to join(...): (line 551)
        
        # Obtaining an instance of the builtin type 'list' (line 551)
        list_57156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 551)
        # Adding element type (line 551)
        # Getting the type of 'self' (line 551)
        self_57157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 41), 'self', False)
        # Obtaining the member 'build_src' of a type (line 551)
        build_src_57158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 41), self_57157, 'build_src')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 551, 40), list_57156, build_src_57158)
        
        
        # Obtaining the type of the subscript
        int_57159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 58), 'int')
        slice_57160 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 552, 41), None, int_57159, None)
        
        # Call to split(...): (line 552)
        # Processing the call arguments (line 552)
        str_57163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 52), 'str', '.')
        # Processing the call keyword arguments (line 552)
        kwargs_57164 = {}
        # Getting the type of 'name' (line 552)
        name_57161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 41), 'name', False)
        # Obtaining the member 'split' of a type (line 552)
        split_57162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 41), name_57161, 'split')
        # Calling split(args, kwargs) (line 552)
        split_call_result_57165 = invoke(stypy.reporting.localization.Localization(__file__, 552, 41), split_57162, *[str_57163], **kwargs_57164)
        
        # Obtaining the member '__getitem__' of a type (line 552)
        getitem___57166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 41), split_call_result_57165, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 552)
        subscript_call_result_57167 = invoke(stypy.reporting.localization.Localization(__file__, 552, 41), getitem___57166, slice_57160)
        
        # Applying the binary operator '+' (line 551)
        result_add_57168 = python_operator(stypy.reporting.localization.Localization(__file__, 551, 40), '+', list_57156, subscript_call_result_57167)
        
        # Processing the call keyword arguments (line 551)
        kwargs_57169 = {}
        # Getting the type of 'os' (line 551)
        os_57153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 551)
        path_57154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 25), os_57153, 'path')
        # Obtaining the member 'join' of a type (line 551)
        join_57155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 25), path_57154, 'join')
        # Calling join(args, kwargs) (line 551)
        join_call_result_57170 = invoke(stypy.reporting.localization.Localization(__file__, 551, 25), join_57155, *[result_add_57168], **kwargs_57169)
        
        # Assigning a type to the variable 'target_dir' (line 551)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 12), 'target_dir', join_call_result_57170)
        
        # Assigning a Call to a Name (line 553):
        
        # Assigning a Call to a Name (line 553):
        
        # Call to join(...): (line 553)
        # Processing the call arguments (line 553)
        # Getting the type of 'target_dir' (line 553)
        target_dir_57174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 39), 'target_dir', False)
        # Getting the type of 'ext_name' (line 553)
        ext_name_57175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 51), 'ext_name', False)
        str_57176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 62), 'str', 'module.c')
        # Applying the binary operator '+' (line 553)
        result_add_57177 = python_operator(stypy.reporting.localization.Localization(__file__, 553, 51), '+', ext_name_57175, str_57176)
        
        # Processing the call keyword arguments (line 553)
        kwargs_57178 = {}
        # Getting the type of 'os' (line 553)
        os_57171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 553)
        path_57172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 26), os_57171, 'path')
        # Obtaining the member 'join' of a type (line 553)
        join_57173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 26), path_57172, 'join')
        # Calling join(args, kwargs) (line 553)
        join_call_result_57179 = invoke(stypy.reporting.localization.Localization(__file__, 553, 26), join_57173, *[target_dir_57174, result_add_57177], **kwargs_57178)
        
        # Assigning a type to the variable 'target_file' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 12), 'target_file', join_call_result_57179)
        
        # Call to append(...): (line 554)
        # Processing the call arguments (line 554)
        # Getting the type of 'target_file' (line 554)
        target_file_57182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 31), 'target_file', False)
        # Processing the call keyword arguments (line 554)
        kwargs_57183 = {}
        # Getting the type of 'new_sources' (line 554)
        new_sources_57180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 12), 'new_sources', False)
        # Obtaining the member 'append' of a type (line 554)
        append_57181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 12), new_sources_57180, 'append')
        # Calling append(args, kwargs) (line 554)
        append_call_result_57184 = invoke(stypy.reporting.localization.Localization(__file__, 554, 12), append_57181, *[target_file_57182], **kwargs_57183)
        
        
        # Assigning a BinOp to a Name (line 555):
        
        # Assigning a BinOp to a Name (line 555):
        # Getting the type of 'f_sources' (line 555)
        f_sources_57185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 22), 'f_sources')
        # Getting the type of 'extension' (line 555)
        extension_57186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 34), 'extension')
        # Obtaining the member 'depends' of a type (line 555)
        depends_57187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 34), extension_57186, 'depends')
        # Applying the binary operator '+' (line 555)
        result_add_57188 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 22), '+', f_sources_57185, depends_57187)
        
        # Assigning a type to the variable 'depends' (line 555)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 12), 'depends', result_add_57188)
        
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 556)
        self_57189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 16), 'self')
        # Obtaining the member 'force' of a type (line 556)
        force_57190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 16), self_57189, 'force')
        
        # Call to newer_group(...): (line 556)
        # Processing the call arguments (line 556)
        # Getting the type of 'depends' (line 556)
        depends_57192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 42), 'depends', False)
        # Getting the type of 'target_file' (line 556)
        target_file_57193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 51), 'target_file', False)
        str_57194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 64), 'str', 'newer')
        # Processing the call keyword arguments (line 556)
        kwargs_57195 = {}
        # Getting the type of 'newer_group' (line 556)
        newer_group_57191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 30), 'newer_group', False)
        # Calling newer_group(args, kwargs) (line 556)
        newer_group_call_result_57196 = invoke(stypy.reporting.localization.Localization(__file__, 556, 30), newer_group_57191, *[depends_57192, target_file_57193, str_57194], **kwargs_57195)
        
        # Applying the binary operator 'or' (line 556)
        result_or_keyword_57197 = python_operator(stypy.reporting.localization.Localization(__file__, 556, 16), 'or', force_57190, newer_group_call_result_57196)
        
        
        # Getting the type of 'skip_f2py' (line 557)
        skip_f2py_57198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 27), 'skip_f2py')
        # Applying the 'not' unary operator (line 557)
        result_not__57199 = python_operator(stypy.reporting.localization.Localization(__file__, 557, 23), 'not', skip_f2py_57198)
        
        # Applying the binary operator 'and' (line 556)
        result_and_keyword_57200 = python_operator(stypy.reporting.localization.Localization(__file__, 556, 15), 'and', result_or_keyword_57197, result_not__57199)
        
        # Testing the type of an if condition (line 556)
        if_condition_57201 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 556, 12), result_and_keyword_57200)
        # Assigning a type to the variable 'if_condition_57201' (line 556)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 12), 'if_condition_57201', if_condition_57201)
        # SSA begins for if statement (line 556)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 558)
        # Processing the call arguments (line 558)
        str_57204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 25), 'str', 'f2py:> %s')
        # Getting the type of 'target_file' (line 558)
        target_file_57205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 40), 'target_file', False)
        # Applying the binary operator '%' (line 558)
        result_mod_57206 = python_operator(stypy.reporting.localization.Localization(__file__, 558, 25), '%', str_57204, target_file_57205)
        
        # Processing the call keyword arguments (line 558)
        kwargs_57207 = {}
        # Getting the type of 'log' (line 558)
        log_57202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 16), 'log', False)
        # Obtaining the member 'info' of a type (line 558)
        info_57203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 16), log_57202, 'info')
        # Calling info(args, kwargs) (line 558)
        info_call_result_57208 = invoke(stypy.reporting.localization.Localization(__file__, 558, 16), info_57203, *[result_mod_57206], **kwargs_57207)
        
        
        # Call to mkpath(...): (line 559)
        # Processing the call arguments (line 559)
        # Getting the type of 'target_dir' (line 559)
        target_dir_57211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 28), 'target_dir', False)
        # Processing the call keyword arguments (line 559)
        kwargs_57212 = {}
        # Getting the type of 'self' (line 559)
        self_57209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 16), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 559)
        mkpath_57210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 16), self_57209, 'mkpath')
        # Calling mkpath(args, kwargs) (line 559)
        mkpath_call_result_57213 = invoke(stypy.reporting.localization.Localization(__file__, 559, 16), mkpath_57210, *[target_dir_57211], **kwargs_57212)
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 560, 16))
        
        # 'import numpy.f2py' statement (line 560)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
        import_57214 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 560, 16), 'numpy.f2py')

        if (type(import_57214) is not StypyTypeError):

            if (import_57214 != 'pyd_module'):
                __import__(import_57214)
                sys_modules_57215 = sys.modules[import_57214]
                import_module(stypy.reporting.localization.Localization(__file__, 560, 16), 'numpy.f2py', sys_modules_57215.module_type_store, module_type_store)
            else:
                import numpy.f2py

                import_module(stypy.reporting.localization.Localization(__file__, 560, 16), 'numpy.f2py', numpy.f2py, module_type_store)

        else:
            # Assigning a type to the variable 'numpy.f2py' (line 560)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 16), 'numpy.f2py', import_57214)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')
        
        
        # Call to run_main(...): (line 561)
        # Processing the call arguments (line 561)
        # Getting the type of 'f2py_options' (line 561)
        f2py_options_57219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 36), 'f2py_options', False)
        
        # Obtaining an instance of the builtin type 'list' (line 561)
        list_57220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 561)
        # Adding element type (line 561)
        str_57221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 52), 'str', '--lower')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 51), list_57220, str_57221)
        # Adding element type (line 561)
        str_57222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 48), 'str', '--build-dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 51), list_57220, str_57222)
        # Adding element type (line 561)
        # Getting the type of 'target_dir' (line 562)
        target_dir_57223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 63), 'target_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 51), list_57220, target_dir_57223)
        
        # Applying the binary operator '+' (line 561)
        result_add_57224 = python_operator(stypy.reporting.localization.Localization(__file__, 561, 36), '+', f2py_options_57219, list_57220)
        
        
        # Obtaining an instance of the builtin type 'list' (line 563)
        list_57225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 563)
        # Adding element type (line 563)
        str_57226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 33), 'str', '-m')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 32), list_57225, str_57226)
        # Adding element type (line 563)
        # Getting the type of 'ext_name' (line 563)
        ext_name_57227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 39), 'ext_name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 32), list_57225, ext_name_57227)
        
        # Applying the binary operator '+' (line 562)
        result_add_57228 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 74), '+', result_add_57224, list_57225)
        
        # Getting the type of 'f_sources' (line 563)
        f_sources_57229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 49), 'f_sources', False)
        # Applying the binary operator '+' (line 563)
        result_add_57230 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 48), '+', result_add_57228, f_sources_57229)
        
        # Processing the call keyword arguments (line 561)
        kwargs_57231 = {}
        # Getting the type of 'numpy' (line 561)
        numpy_57216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 16), 'numpy', False)
        # Obtaining the member 'f2py' of a type (line 561)
        f2py_57217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 16), numpy_57216, 'f2py')
        # Obtaining the member 'run_main' of a type (line 561)
        run_main_57218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 16), f2py_57217, 'run_main')
        # Calling run_main(args, kwargs) (line 561)
        run_main_call_result_57232 = invoke(stypy.reporting.localization.Localization(__file__, 561, 16), run_main_57218, *[result_add_57230], **kwargs_57231)
        
        # SSA branch for the else part of an if statement (line 556)
        module_type_store.open_ssa_branch('else')
        
        # Call to debug(...): (line 565)
        # Processing the call arguments (line 565)
        str_57235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 26), 'str', "  skipping f2py fortran files for '%s' (up-to-date)")
        # Getting the type of 'target_file' (line 566)
        target_file_57236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 29), 'target_file', False)
        # Applying the binary operator '%' (line 565)
        result_mod_57237 = python_operator(stypy.reporting.localization.Localization(__file__, 565, 26), '%', str_57235, target_file_57236)
        
        # Processing the call keyword arguments (line 565)
        kwargs_57238 = {}
        # Getting the type of 'log' (line 565)
        log_57233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 16), 'log', False)
        # Obtaining the member 'debug' of a type (line 565)
        debug_57234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 16), log_57233, 'debug')
        # Calling debug(args, kwargs) (line 565)
        debug_call_result_57239 = invoke(stypy.reporting.localization.Localization(__file__, 565, 16), debug_57234, *[result_mod_57237], **kwargs_57238)
        
        # SSA join for if statement (line 556)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 529)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to isfile(...): (line 568)
        # Processing the call arguments (line 568)
        # Getting the type of 'target_file' (line 568)
        target_file_57243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 30), 'target_file', False)
        # Processing the call keyword arguments (line 568)
        kwargs_57244 = {}
        # Getting the type of 'os' (line 568)
        os_57240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 568)
        path_57241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 15), os_57240, 'path')
        # Obtaining the member 'isfile' of a type (line 568)
        isfile_57242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 15), path_57241, 'isfile')
        # Calling isfile(args, kwargs) (line 568)
        isfile_call_result_57245 = invoke(stypy.reporting.localization.Localization(__file__, 568, 15), isfile_57242, *[target_file_57243], **kwargs_57244)
        
        # Applying the 'not' unary operator (line 568)
        result_not__57246 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 11), 'not', isfile_call_result_57245)
        
        # Testing the type of an if condition (line 568)
        if_condition_57247 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 568, 8), result_not__57246)
        # Assigning a type to the variable 'if_condition_57247' (line 568)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 8), 'if_condition_57247', if_condition_57247)
        # SSA begins for if statement (line 568)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to DistutilsError(...): (line 569)
        # Processing the call arguments (line 569)
        str_57249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 33), 'str', 'f2py target file %r not generated')
        
        # Obtaining an instance of the builtin type 'tuple' (line 569)
        tuple_57250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 72), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 569)
        # Adding element type (line 569)
        # Getting the type of 'target_file' (line 569)
        target_file_57251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 72), 'target_file', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 569, 72), tuple_57250, target_file_57251)
        
        # Applying the binary operator '%' (line 569)
        result_mod_57252 = python_operator(stypy.reporting.localization.Localization(__file__, 569, 33), '%', str_57249, tuple_57250)
        
        # Processing the call keyword arguments (line 569)
        kwargs_57253 = {}
        # Getting the type of 'DistutilsError' (line 569)
        DistutilsError_57248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 18), 'DistutilsError', False)
        # Calling DistutilsError(args, kwargs) (line 569)
        DistutilsError_call_result_57254 = invoke(stypy.reporting.localization.Localization(__file__, 569, 18), DistutilsError_57248, *[result_mod_57252], **kwargs_57253)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 569, 12), DistutilsError_call_result_57254, 'raise parameter', BaseException)
        # SSA join for if statement (line 568)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 571):
        
        # Assigning a Call to a Name (line 571):
        
        # Call to join(...): (line 571)
        # Processing the call arguments (line 571)
        # Getting the type of 'self' (line 571)
        self_57258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 32), 'self', False)
        # Obtaining the member 'build_src' of a type (line 571)
        build_src_57259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 32), self_57258, 'build_src')
        str_57260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 48), 'str', 'fortranobject.c')
        # Processing the call keyword arguments (line 571)
        kwargs_57261 = {}
        # Getting the type of 'os' (line 571)
        os_57255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 571)
        path_57256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 19), os_57255, 'path')
        # Obtaining the member 'join' of a type (line 571)
        join_57257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 19), path_57256, 'join')
        # Calling join(args, kwargs) (line 571)
        join_call_result_57262 = invoke(stypy.reporting.localization.Localization(__file__, 571, 19), join_57257, *[build_src_57259, str_57260], **kwargs_57261)
        
        # Assigning a type to the variable 'target_c' (line 571)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'target_c', join_call_result_57262)
        
        # Assigning a Call to a Name (line 572):
        
        # Assigning a Call to a Name (line 572):
        
        # Call to join(...): (line 572)
        # Processing the call arguments (line 572)
        # Getting the type of 'self' (line 572)
        self_57266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 32), 'self', False)
        # Obtaining the member 'build_src' of a type (line 572)
        build_src_57267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 32), self_57266, 'build_src')
        str_57268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 48), 'str', 'fortranobject.h')
        # Processing the call keyword arguments (line 572)
        kwargs_57269 = {}
        # Getting the type of 'os' (line 572)
        os_57263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 572)
        path_57264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 19), os_57263, 'path')
        # Obtaining the member 'join' of a type (line 572)
        join_57265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 19), path_57264, 'join')
        # Calling join(args, kwargs) (line 572)
        join_call_result_57270 = invoke(stypy.reporting.localization.Localization(__file__, 572, 19), join_57265, *[build_src_57267, str_57268], **kwargs_57269)
        
        # Assigning a type to the variable 'target_h' (line 572)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'target_h', join_call_result_57270)
        
        # Call to info(...): (line 573)
        # Processing the call arguments (line 573)
        str_57273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 17), 'str', "  adding '%s' to sources.")
        # Getting the type of 'target_c' (line 573)
        target_c_57274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 48), 'target_c', False)
        # Applying the binary operator '%' (line 573)
        result_mod_57275 = python_operator(stypy.reporting.localization.Localization(__file__, 573, 17), '%', str_57273, target_c_57274)
        
        # Processing the call keyword arguments (line 573)
        kwargs_57276 = {}
        # Getting the type of 'log' (line 573)
        log_57271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 573)
        info_57272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 8), log_57271, 'info')
        # Calling info(args, kwargs) (line 573)
        info_call_result_57277 = invoke(stypy.reporting.localization.Localization(__file__, 573, 8), info_57272, *[result_mod_57275], **kwargs_57276)
        
        
        # Call to append(...): (line 574)
        # Processing the call arguments (line 574)
        # Getting the type of 'target_c' (line 574)
        target_c_57280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 27), 'target_c', False)
        # Processing the call keyword arguments (line 574)
        kwargs_57281 = {}
        # Getting the type of 'new_sources' (line 574)
        new_sources_57278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'new_sources', False)
        # Obtaining the member 'append' of a type (line 574)
        append_57279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 8), new_sources_57278, 'append')
        # Calling append(args, kwargs) (line 574)
        append_call_result_57282 = invoke(stypy.reporting.localization.Localization(__file__, 574, 8), append_57279, *[target_c_57280], **kwargs_57281)
        
        
        
        # Getting the type of 'self' (line 575)
        self_57283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 11), 'self')
        # Obtaining the member 'build_src' of a type (line 575)
        build_src_57284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 11), self_57283, 'build_src')
        # Getting the type of 'extension' (line 575)
        extension_57285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 33), 'extension')
        # Obtaining the member 'include_dirs' of a type (line 575)
        include_dirs_57286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 33), extension_57285, 'include_dirs')
        # Applying the binary operator 'notin' (line 575)
        result_contains_57287 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 11), 'notin', build_src_57284, include_dirs_57286)
        
        # Testing the type of an if condition (line 575)
        if_condition_57288 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 575, 8), result_contains_57287)
        # Assigning a type to the variable 'if_condition_57288' (line 575)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 8), 'if_condition_57288', if_condition_57288)
        # SSA begins for if statement (line 575)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 576)
        # Processing the call arguments (line 576)
        str_57291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 21), 'str', "  adding '%s' to include_dirs.")
        # Getting the type of 'self' (line 577)
        self_57292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 24), 'self', False)
        # Obtaining the member 'build_src' of a type (line 577)
        build_src_57293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 24), self_57292, 'build_src')
        # Applying the binary operator '%' (line 576)
        result_mod_57294 = python_operator(stypy.reporting.localization.Localization(__file__, 576, 21), '%', str_57291, build_src_57293)
        
        # Processing the call keyword arguments (line 576)
        kwargs_57295 = {}
        # Getting the type of 'log' (line 576)
        log_57289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 12), 'log', False)
        # Obtaining the member 'info' of a type (line 576)
        info_57290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 12), log_57289, 'info')
        # Calling info(args, kwargs) (line 576)
        info_call_result_57296 = invoke(stypy.reporting.localization.Localization(__file__, 576, 12), info_57290, *[result_mod_57294], **kwargs_57295)
        
        
        # Call to append(...): (line 578)
        # Processing the call arguments (line 578)
        # Getting the type of 'self' (line 578)
        self_57300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 42), 'self', False)
        # Obtaining the member 'build_src' of a type (line 578)
        build_src_57301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 42), self_57300, 'build_src')
        # Processing the call keyword arguments (line 578)
        kwargs_57302 = {}
        # Getting the type of 'extension' (line 578)
        extension_57297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 12), 'extension', False)
        # Obtaining the member 'include_dirs' of a type (line 578)
        include_dirs_57298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 12), extension_57297, 'include_dirs')
        # Obtaining the member 'append' of a type (line 578)
        append_57299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 12), include_dirs_57298, 'append')
        # Calling append(args, kwargs) (line 578)
        append_call_result_57303 = invoke(stypy.reporting.localization.Localization(__file__, 578, 12), append_57299, *[build_src_57301], **kwargs_57302)
        
        # SSA join for if statement (line 575)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'skip_f2py' (line 580)
        skip_f2py_57304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 15), 'skip_f2py')
        # Applying the 'not' unary operator (line 580)
        result_not__57305 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 11), 'not', skip_f2py_57304)
        
        # Testing the type of an if condition (line 580)
        if_condition_57306 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 580, 8), result_not__57305)
        # Assigning a type to the variable 'if_condition_57306' (line 580)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 8), 'if_condition_57306', if_condition_57306)
        # SSA begins for if statement (line 580)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 581, 12))
        
        # 'import numpy.f2py' statement (line 581)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
        import_57307 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 581, 12), 'numpy.f2py')

        if (type(import_57307) is not StypyTypeError):

            if (import_57307 != 'pyd_module'):
                __import__(import_57307)
                sys_modules_57308 = sys.modules[import_57307]
                import_module(stypy.reporting.localization.Localization(__file__, 581, 12), 'numpy.f2py', sys_modules_57308.module_type_store, module_type_store)
            else:
                import numpy.f2py

                import_module(stypy.reporting.localization.Localization(__file__, 581, 12), 'numpy.f2py', numpy.f2py, module_type_store)

        else:
            # Assigning a type to the variable 'numpy.f2py' (line 581)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 12), 'numpy.f2py', import_57307)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')
        
        
        # Assigning a Call to a Name (line 582):
        
        # Assigning a Call to a Name (line 582):
        
        # Call to dirname(...): (line 582)
        # Processing the call arguments (line 582)
        # Getting the type of 'numpy' (line 582)
        numpy_57312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 32), 'numpy', False)
        # Obtaining the member 'f2py' of a type (line 582)
        f2py_57313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 32), numpy_57312, 'f2py')
        # Obtaining the member '__file__' of a type (line 582)
        file___57314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 32), f2py_57313, '__file__')
        # Processing the call keyword arguments (line 582)
        kwargs_57315 = {}
        # Getting the type of 'os' (line 582)
        os_57309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 16), 'os', False)
        # Obtaining the member 'path' of a type (line 582)
        path_57310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 16), os_57309, 'path')
        # Obtaining the member 'dirname' of a type (line 582)
        dirname_57311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 16), path_57310, 'dirname')
        # Calling dirname(args, kwargs) (line 582)
        dirname_call_result_57316 = invoke(stypy.reporting.localization.Localization(__file__, 582, 16), dirname_57311, *[file___57314], **kwargs_57315)
        
        # Assigning a type to the variable 'd' (line 582)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 12), 'd', dirname_call_result_57316)
        
        # Assigning a Call to a Name (line 583):
        
        # Assigning a Call to a Name (line 583):
        
        # Call to join(...): (line 583)
        # Processing the call arguments (line 583)
        # Getting the type of 'd' (line 583)
        d_57320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 36), 'd', False)
        str_57321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 39), 'str', 'src')
        str_57322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 46), 'str', 'fortranobject.c')
        # Processing the call keyword arguments (line 583)
        kwargs_57323 = {}
        # Getting the type of 'os' (line 583)
        os_57317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 583)
        path_57318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 23), os_57317, 'path')
        # Obtaining the member 'join' of a type (line 583)
        join_57319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 23), path_57318, 'join')
        # Calling join(args, kwargs) (line 583)
        join_call_result_57324 = invoke(stypy.reporting.localization.Localization(__file__, 583, 23), join_57319, *[d_57320, str_57321, str_57322], **kwargs_57323)
        
        # Assigning a type to the variable 'source_c' (line 583)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 12), 'source_c', join_call_result_57324)
        
        # Assigning a Call to a Name (line 584):
        
        # Assigning a Call to a Name (line 584):
        
        # Call to join(...): (line 584)
        # Processing the call arguments (line 584)
        # Getting the type of 'd' (line 584)
        d_57328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 36), 'd', False)
        str_57329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 39), 'str', 'src')
        str_57330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 46), 'str', 'fortranobject.h')
        # Processing the call keyword arguments (line 584)
        kwargs_57331 = {}
        # Getting the type of 'os' (line 584)
        os_57325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 584)
        path_57326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 23), os_57325, 'path')
        # Obtaining the member 'join' of a type (line 584)
        join_57327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 23), path_57326, 'join')
        # Calling join(args, kwargs) (line 584)
        join_call_result_57332 = invoke(stypy.reporting.localization.Localization(__file__, 584, 23), join_57327, *[d_57328, str_57329, str_57330], **kwargs_57331)
        
        # Assigning a type to the variable 'source_h' (line 584)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 12), 'source_h', join_call_result_57332)
        
        
        # Evaluating a boolean operation
        
        # Call to newer(...): (line 585)
        # Processing the call arguments (line 585)
        # Getting the type of 'source_c' (line 585)
        source_c_57334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 21), 'source_c', False)
        # Getting the type of 'target_c' (line 585)
        target_c_57335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 31), 'target_c', False)
        # Processing the call keyword arguments (line 585)
        kwargs_57336 = {}
        # Getting the type of 'newer' (line 585)
        newer_57333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 15), 'newer', False)
        # Calling newer(args, kwargs) (line 585)
        newer_call_result_57337 = invoke(stypy.reporting.localization.Localization(__file__, 585, 15), newer_57333, *[source_c_57334, target_c_57335], **kwargs_57336)
        
        
        # Call to newer(...): (line 585)
        # Processing the call arguments (line 585)
        # Getting the type of 'source_h' (line 585)
        source_h_57339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 50), 'source_h', False)
        # Getting the type of 'target_h' (line 585)
        target_h_57340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 60), 'target_h', False)
        # Processing the call keyword arguments (line 585)
        kwargs_57341 = {}
        # Getting the type of 'newer' (line 585)
        newer_57338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 44), 'newer', False)
        # Calling newer(args, kwargs) (line 585)
        newer_call_result_57342 = invoke(stypy.reporting.localization.Localization(__file__, 585, 44), newer_57338, *[source_h_57339, target_h_57340], **kwargs_57341)
        
        # Applying the binary operator 'or' (line 585)
        result_or_keyword_57343 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 15), 'or', newer_call_result_57337, newer_call_result_57342)
        
        # Testing the type of an if condition (line 585)
        if_condition_57344 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 585, 12), result_or_keyword_57343)
        # Assigning a type to the variable 'if_condition_57344' (line 585)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 12), 'if_condition_57344', if_condition_57344)
        # SSA begins for if statement (line 585)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to mkpath(...): (line 586)
        # Processing the call arguments (line 586)
        
        # Call to dirname(...): (line 586)
        # Processing the call arguments (line 586)
        # Getting the type of 'target_c' (line 586)
        target_c_57350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 44), 'target_c', False)
        # Processing the call keyword arguments (line 586)
        kwargs_57351 = {}
        # Getting the type of 'os' (line 586)
        os_57347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 28), 'os', False)
        # Obtaining the member 'path' of a type (line 586)
        path_57348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 28), os_57347, 'path')
        # Obtaining the member 'dirname' of a type (line 586)
        dirname_57349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 28), path_57348, 'dirname')
        # Calling dirname(args, kwargs) (line 586)
        dirname_call_result_57352 = invoke(stypy.reporting.localization.Localization(__file__, 586, 28), dirname_57349, *[target_c_57350], **kwargs_57351)
        
        # Processing the call keyword arguments (line 586)
        kwargs_57353 = {}
        # Getting the type of 'self' (line 586)
        self_57345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 16), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 586)
        mkpath_57346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 16), self_57345, 'mkpath')
        # Calling mkpath(args, kwargs) (line 586)
        mkpath_call_result_57354 = invoke(stypy.reporting.localization.Localization(__file__, 586, 16), mkpath_57346, *[dirname_call_result_57352], **kwargs_57353)
        
        
        # Call to copy_file(...): (line 587)
        # Processing the call arguments (line 587)
        # Getting the type of 'source_c' (line 587)
        source_c_57357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 31), 'source_c', False)
        # Getting the type of 'target_c' (line 587)
        target_c_57358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 41), 'target_c', False)
        # Processing the call keyword arguments (line 587)
        kwargs_57359 = {}
        # Getting the type of 'self' (line 587)
        self_57355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 16), 'self', False)
        # Obtaining the member 'copy_file' of a type (line 587)
        copy_file_57356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 16), self_57355, 'copy_file')
        # Calling copy_file(args, kwargs) (line 587)
        copy_file_call_result_57360 = invoke(stypy.reporting.localization.Localization(__file__, 587, 16), copy_file_57356, *[source_c_57357, target_c_57358], **kwargs_57359)
        
        
        # Call to copy_file(...): (line 588)
        # Processing the call arguments (line 588)
        # Getting the type of 'source_h' (line 588)
        source_h_57363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 31), 'source_h', False)
        # Getting the type of 'target_h' (line 588)
        target_h_57364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 41), 'target_h', False)
        # Processing the call keyword arguments (line 588)
        kwargs_57365 = {}
        # Getting the type of 'self' (line 588)
        self_57361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 16), 'self', False)
        # Obtaining the member 'copy_file' of a type (line 588)
        copy_file_57362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 16), self_57361, 'copy_file')
        # Calling copy_file(args, kwargs) (line 588)
        copy_file_call_result_57366 = invoke(stypy.reporting.localization.Localization(__file__, 588, 16), copy_file_57362, *[source_h_57363, target_h_57364], **kwargs_57365)
        
        # SSA join for if statement (line 585)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 580)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to isfile(...): (line 590)
        # Processing the call arguments (line 590)
        # Getting the type of 'target_c' (line 590)
        target_c_57370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 34), 'target_c', False)
        # Processing the call keyword arguments (line 590)
        kwargs_57371 = {}
        # Getting the type of 'os' (line 590)
        os_57367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 590)
        path_57368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 19), os_57367, 'path')
        # Obtaining the member 'isfile' of a type (line 590)
        isfile_57369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 19), path_57368, 'isfile')
        # Calling isfile(args, kwargs) (line 590)
        isfile_call_result_57372 = invoke(stypy.reporting.localization.Localization(__file__, 590, 19), isfile_57369, *[target_c_57370], **kwargs_57371)
        
        # Applying the 'not' unary operator (line 590)
        result_not__57373 = python_operator(stypy.reporting.localization.Localization(__file__, 590, 15), 'not', isfile_call_result_57372)
        
        # Testing the type of an if condition (line 590)
        if_condition_57374 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 590, 12), result_not__57373)
        # Assigning a type to the variable 'if_condition_57374' (line 590)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 12), 'if_condition_57374', if_condition_57374)
        # SSA begins for if statement (line 590)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to DistutilsSetupError(...): (line 591)
        # Processing the call arguments (line 591)
        str_57376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 42), 'str', 'f2py target_c file %r not found')
        
        # Obtaining an instance of the builtin type 'tuple' (line 591)
        tuple_57377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 79), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 591)
        # Adding element type (line 591)
        # Getting the type of 'target_c' (line 591)
        target_c_57378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 79), 'target_c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 79), tuple_57377, target_c_57378)
        
        # Applying the binary operator '%' (line 591)
        result_mod_57379 = python_operator(stypy.reporting.localization.Localization(__file__, 591, 42), '%', str_57376, tuple_57377)
        
        # Processing the call keyword arguments (line 591)
        kwargs_57380 = {}
        # Getting the type of 'DistutilsSetupError' (line 591)
        DistutilsSetupError_57375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 22), 'DistutilsSetupError', False)
        # Calling DistutilsSetupError(args, kwargs) (line 591)
        DistutilsSetupError_call_result_57381 = invoke(stypy.reporting.localization.Localization(__file__, 591, 22), DistutilsSetupError_57375, *[result_mod_57379], **kwargs_57380)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 591, 16), DistutilsSetupError_call_result_57381, 'raise parameter', BaseException)
        # SSA join for if statement (line 590)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to isfile(...): (line 592)
        # Processing the call arguments (line 592)
        # Getting the type of 'target_h' (line 592)
        target_h_57385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 34), 'target_h', False)
        # Processing the call keyword arguments (line 592)
        kwargs_57386 = {}
        # Getting the type of 'os' (line 592)
        os_57382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 592)
        path_57383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 19), os_57382, 'path')
        # Obtaining the member 'isfile' of a type (line 592)
        isfile_57384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 19), path_57383, 'isfile')
        # Calling isfile(args, kwargs) (line 592)
        isfile_call_result_57387 = invoke(stypy.reporting.localization.Localization(__file__, 592, 19), isfile_57384, *[target_h_57385], **kwargs_57386)
        
        # Applying the 'not' unary operator (line 592)
        result_not__57388 = python_operator(stypy.reporting.localization.Localization(__file__, 592, 15), 'not', isfile_call_result_57387)
        
        # Testing the type of an if condition (line 592)
        if_condition_57389 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 592, 12), result_not__57388)
        # Assigning a type to the variable 'if_condition_57389' (line 592)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 12), 'if_condition_57389', if_condition_57389)
        # SSA begins for if statement (line 592)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to DistutilsSetupError(...): (line 593)
        # Processing the call arguments (line 593)
        str_57391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 42), 'str', 'f2py target_h file %r not found')
        
        # Obtaining an instance of the builtin type 'tuple' (line 593)
        tuple_57392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 79), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 593)
        # Adding element type (line 593)
        # Getting the type of 'target_h' (line 593)
        target_h_57393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 79), 'target_h', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 593, 79), tuple_57392, target_h_57393)
        
        # Applying the binary operator '%' (line 593)
        result_mod_57394 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 42), '%', str_57391, tuple_57392)
        
        # Processing the call keyword arguments (line 593)
        kwargs_57395 = {}
        # Getting the type of 'DistutilsSetupError' (line 593)
        DistutilsSetupError_57390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 22), 'DistutilsSetupError', False)
        # Calling DistutilsSetupError(args, kwargs) (line 593)
        DistutilsSetupError_call_result_57396 = invoke(stypy.reporting.localization.Localization(__file__, 593, 22), DistutilsSetupError_57390, *[result_mod_57394], **kwargs_57395)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 593, 16), DistutilsSetupError_call_result_57396, 'raise parameter', BaseException)
        # SSA join for if statement (line 592)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 580)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Obtaining an instance of the builtin type 'list' (line 595)
        list_57397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 595)
        # Adding element type (line 595)
        str_57398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 25), 'str', '-f2pywrappers.f')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 595, 24), list_57397, str_57398)
        # Adding element type (line 595)
        str_57399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 44), 'str', '-f2pywrappers2.f90')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 595, 24), list_57397, str_57399)
        
        # Testing the type of a for loop iterable (line 595)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 595, 8), list_57397)
        # Getting the type of the for loop variable (line 595)
        for_loop_var_57400 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 595, 8), list_57397)
        # Assigning a type to the variable 'name_ext' (line 595)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 8), 'name_ext', for_loop_var_57400)
        # SSA begins for a for statement (line 595)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 596):
        
        # Assigning a Call to a Name (line 596):
        
        # Call to join(...): (line 596)
        # Processing the call arguments (line 596)
        # Getting the type of 'target_dir' (line 596)
        target_dir_57404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 36), 'target_dir', False)
        # Getting the type of 'ext_name' (line 596)
        ext_name_57405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 48), 'ext_name', False)
        # Getting the type of 'name_ext' (line 596)
        name_ext_57406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 59), 'name_ext', False)
        # Applying the binary operator '+' (line 596)
        result_add_57407 = python_operator(stypy.reporting.localization.Localization(__file__, 596, 48), '+', ext_name_57405, name_ext_57406)
        
        # Processing the call keyword arguments (line 596)
        kwargs_57408 = {}
        # Getting the type of 'os' (line 596)
        os_57401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 596)
        path_57402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 23), os_57401, 'path')
        # Obtaining the member 'join' of a type (line 596)
        join_57403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 23), path_57402, 'join')
        # Calling join(args, kwargs) (line 596)
        join_call_result_57409 = invoke(stypy.reporting.localization.Localization(__file__, 596, 23), join_57403, *[target_dir_57404, result_add_57407], **kwargs_57408)
        
        # Assigning a type to the variable 'filename' (line 596)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 12), 'filename', join_call_result_57409)
        
        
        # Call to isfile(...): (line 597)
        # Processing the call arguments (line 597)
        # Getting the type of 'filename' (line 597)
        filename_57413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 30), 'filename', False)
        # Processing the call keyword arguments (line 597)
        kwargs_57414 = {}
        # Getting the type of 'os' (line 597)
        os_57410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 597)
        path_57411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 15), os_57410, 'path')
        # Obtaining the member 'isfile' of a type (line 597)
        isfile_57412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 15), path_57411, 'isfile')
        # Calling isfile(args, kwargs) (line 597)
        isfile_call_result_57415 = invoke(stypy.reporting.localization.Localization(__file__, 597, 15), isfile_57412, *[filename_57413], **kwargs_57414)
        
        # Testing the type of an if condition (line 597)
        if_condition_57416 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 597, 12), isfile_call_result_57415)
        # Assigning a type to the variable 'if_condition_57416' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 12), 'if_condition_57416', if_condition_57416)
        # SSA begins for if statement (line 597)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 598)
        # Processing the call arguments (line 598)
        str_57419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 25), 'str', "  adding '%s' to sources.")
        # Getting the type of 'filename' (line 598)
        filename_57420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 56), 'filename', False)
        # Applying the binary operator '%' (line 598)
        result_mod_57421 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 25), '%', str_57419, filename_57420)
        
        # Processing the call keyword arguments (line 598)
        kwargs_57422 = {}
        # Getting the type of 'log' (line 598)
        log_57417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 16), 'log', False)
        # Obtaining the member 'info' of a type (line 598)
        info_57418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 16), log_57417, 'info')
        # Calling info(args, kwargs) (line 598)
        info_call_result_57423 = invoke(stypy.reporting.localization.Localization(__file__, 598, 16), info_57418, *[result_mod_57421], **kwargs_57422)
        
        
        # Call to append(...): (line 599)
        # Processing the call arguments (line 599)
        # Getting the type of 'filename' (line 599)
        filename_57426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 33), 'filename', False)
        # Processing the call keyword arguments (line 599)
        kwargs_57427 = {}
        # Getting the type of 'f_sources' (line 599)
        f_sources_57424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 16), 'f_sources', False)
        # Obtaining the member 'append' of a type (line 599)
        append_57425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 16), f_sources_57424, 'append')
        # Calling append(args, kwargs) (line 599)
        append_call_result_57428 = invoke(stypy.reporting.localization.Localization(__file__, 599, 16), append_57425, *[filename_57426], **kwargs_57427)
        
        # SSA join for if statement (line 597)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'new_sources' (line 601)
        new_sources_57429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 15), 'new_sources')
        # Getting the type of 'f_sources' (line 601)
        f_sources_57430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 29), 'f_sources')
        # Applying the binary operator '+' (line 601)
        result_add_57431 = python_operator(stypy.reporting.localization.Localization(__file__, 601, 15), '+', new_sources_57429, f_sources_57430)
        
        # Assigning a type to the variable 'stypy_return_type' (line 601)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 8), 'stypy_return_type', result_add_57431)
        
        # ################# End of 'f2py_sources(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f2py_sources' in the type store
        # Getting the type of 'stypy_return_type' (line 465)
        stypy_return_type_57432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_57432)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f2py_sources'
        return stypy_return_type_57432


    @norecursion
    def swig_sources(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'swig_sources'
        module_type_store = module_type_store.open_function_context('swig_sources', 603, 4, False)
        # Assigning a type to the variable 'self' (line 604)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_src.swig_sources.__dict__.__setitem__('stypy_localization', localization)
        build_src.swig_sources.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_src.swig_sources.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_src.swig_sources.__dict__.__setitem__('stypy_function_name', 'build_src.swig_sources')
        build_src.swig_sources.__dict__.__setitem__('stypy_param_names_list', ['sources', 'extension'])
        build_src.swig_sources.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_src.swig_sources.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_src.swig_sources.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_src.swig_sources.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_src.swig_sources.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_src.swig_sources.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_src.swig_sources', ['sources', 'extension'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'swig_sources', localization, ['sources', 'extension'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'swig_sources(...)' code ##################

        
        # Assigning a List to a Name (line 607):
        
        # Assigning a List to a Name (line 607):
        
        # Obtaining an instance of the builtin type 'list' (line 607)
        list_57433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 607)
        
        # Assigning a type to the variable 'new_sources' (line 607)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 8), 'new_sources', list_57433)
        
        # Assigning a List to a Name (line 608):
        
        # Assigning a List to a Name (line 608):
        
        # Obtaining an instance of the builtin type 'list' (line 608)
        list_57434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 608)
        
        # Assigning a type to the variable 'swig_sources' (line 608)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 8), 'swig_sources', list_57434)
        
        # Assigning a Dict to a Name (line 609):
        
        # Assigning a Dict to a Name (line 609):
        
        # Obtaining an instance of the builtin type 'dict' (line 609)
        dict_57435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 23), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 609)
        
        # Assigning a type to the variable 'swig_targets' (line 609)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 8), 'swig_targets', dict_57435)
        
        # Assigning a List to a Name (line 610):
        
        # Assigning a List to a Name (line 610):
        
        # Obtaining an instance of the builtin type 'list' (line 610)
        list_57436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 610)
        
        # Assigning a type to the variable 'target_dirs' (line 610)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 8), 'target_dirs', list_57436)
        
        # Assigning a List to a Name (line 611):
        
        # Assigning a List to a Name (line 611):
        
        # Obtaining an instance of the builtin type 'list' (line 611)
        list_57437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 611)
        
        # Assigning a type to the variable 'py_files' (line 611)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 8), 'py_files', list_57437)
        
        # Assigning a Str to a Name (line 612):
        
        # Assigning a Str to a Name (line 612):
        str_57438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 21), 'str', '.c')
        # Assigning a type to the variable 'target_ext' (line 612)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 8), 'target_ext', str_57438)
        
        
        str_57439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 11), 'str', '-c++')
        # Getting the type of 'extension' (line 613)
        extension_57440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 21), 'extension')
        # Obtaining the member 'swig_opts' of a type (line 613)
        swig_opts_57441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 21), extension_57440, 'swig_opts')
        # Applying the binary operator 'in' (line 613)
        result_contains_57442 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 11), 'in', str_57439, swig_opts_57441)
        
        # Testing the type of an if condition (line 613)
        if_condition_57443 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 613, 8), result_contains_57442)
        # Assigning a type to the variable 'if_condition_57443' (line 613)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 8), 'if_condition_57443', if_condition_57443)
        # SSA begins for if statement (line 613)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 614):
        
        # Assigning a Str to a Name (line 614):
        str_57444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 18), 'str', 'c++')
        # Assigning a type to the variable 'typ' (line 614)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 12), 'typ', str_57444)
        
        # Assigning a Name to a Name (line 615):
        
        # Assigning a Name to a Name (line 615):
        # Getting the type of 'True' (line 615)
        True_57445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 21), 'True')
        # Assigning a type to the variable 'is_cpp' (line 615)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 12), 'is_cpp', True_57445)
        
        # Call to remove(...): (line 616)
        # Processing the call arguments (line 616)
        str_57449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 39), 'str', '-c++')
        # Processing the call keyword arguments (line 616)
        kwargs_57450 = {}
        # Getting the type of 'extension' (line 616)
        extension_57446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 12), 'extension', False)
        # Obtaining the member 'swig_opts' of a type (line 616)
        swig_opts_57447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 12), extension_57446, 'swig_opts')
        # Obtaining the member 'remove' of a type (line 616)
        remove_57448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 12), swig_opts_57447, 'remove')
        # Calling remove(args, kwargs) (line 616)
        remove_call_result_57451 = invoke(stypy.reporting.localization.Localization(__file__, 616, 12), remove_57448, *[str_57449], **kwargs_57450)
        
        # SSA branch for the else part of an if statement (line 613)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'self' (line 617)
        self_57452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 13), 'self')
        # Obtaining the member 'swig_cpp' of a type (line 617)
        swig_cpp_57453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 13), self_57452, 'swig_cpp')
        # Testing the type of an if condition (line 617)
        if_condition_57454 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 617, 13), swig_cpp_57453)
        # Assigning a type to the variable 'if_condition_57454' (line 617)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 13), 'if_condition_57454', if_condition_57454)
        # SSA begins for if statement (line 617)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 618):
        
        # Assigning a Str to a Name (line 618):
        str_57455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 18), 'str', 'c++')
        # Assigning a type to the variable 'typ' (line 618)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 12), 'typ', str_57455)
        
        # Assigning a Name to a Name (line 619):
        
        # Assigning a Name to a Name (line 619):
        # Getting the type of 'True' (line 619)
        True_57456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 21), 'True')
        # Assigning a type to the variable 'is_cpp' (line 619)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 12), 'is_cpp', True_57456)
        # SSA branch for the else part of an if statement (line 617)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 621):
        
        # Assigning a Name to a Name (line 621):
        # Getting the type of 'None' (line 621)
        None_57457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 18), 'None')
        # Assigning a type to the variable 'typ' (line 621)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 12), 'typ', None_57457)
        
        # Assigning a Name to a Name (line 622):
        
        # Assigning a Name to a Name (line 622):
        # Getting the type of 'False' (line 622)
        False_57458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 21), 'False')
        # Assigning a type to the variable 'is_cpp' (line 622)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 12), 'is_cpp', False_57458)
        # SSA join for if statement (line 617)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 613)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Num to a Name (line 623):
        
        # Assigning a Num to a Name (line 623):
        int_57459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 20), 'int')
        # Assigning a type to the variable 'skip_swig' (line 623)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 8), 'skip_swig', int_57459)
        
        # Assigning a Subscript to a Name (line 624):
        
        # Assigning a Subscript to a Name (line 624):
        
        # Obtaining the type of the subscript
        int_57460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 45), 'int')
        
        # Call to split(...): (line 624)
        # Processing the call arguments (line 624)
        str_57464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 40), 'str', '.')
        # Processing the call keyword arguments (line 624)
        kwargs_57465 = {}
        # Getting the type of 'extension' (line 624)
        extension_57461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 19), 'extension', False)
        # Obtaining the member 'name' of a type (line 624)
        name_57462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 19), extension_57461, 'name')
        # Obtaining the member 'split' of a type (line 624)
        split_57463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 19), name_57462, 'split')
        # Calling split(args, kwargs) (line 624)
        split_call_result_57466 = invoke(stypy.reporting.localization.Localization(__file__, 624, 19), split_57463, *[str_57464], **kwargs_57465)
        
        # Obtaining the member '__getitem__' of a type (line 624)
        getitem___57467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 19), split_call_result_57466, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 624)
        subscript_call_result_57468 = invoke(stypy.reporting.localization.Localization(__file__, 624, 19), getitem___57467, int_57460)
        
        # Assigning a type to the variable 'ext_name' (line 624)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 8), 'ext_name', subscript_call_result_57468)
        
        # Getting the type of 'sources' (line 626)
        sources_57469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 22), 'sources')
        # Testing the type of a for loop iterable (line 626)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 626, 8), sources_57469)
        # Getting the type of the for loop variable (line 626)
        for_loop_var_57470 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 626, 8), sources_57469)
        # Assigning a type to the variable 'source' (line 626)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 8), 'source', for_loop_var_57470)
        # SSA begins for a for statement (line 626)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 627):
        
        # Assigning a Call to a Name:
        
        # Call to splitext(...): (line 627)
        # Processing the call arguments (line 627)
        # Getting the type of 'source' (line 627)
        source_57474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 43), 'source', False)
        # Processing the call keyword arguments (line 627)
        kwargs_57475 = {}
        # Getting the type of 'os' (line 627)
        os_57471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 627)
        path_57472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 26), os_57471, 'path')
        # Obtaining the member 'splitext' of a type (line 627)
        splitext_57473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 26), path_57472, 'splitext')
        # Calling splitext(args, kwargs) (line 627)
        splitext_call_result_57476 = invoke(stypy.reporting.localization.Localization(__file__, 627, 26), splitext_57473, *[source_57474], **kwargs_57475)
        
        # Assigning a type to the variable 'call_assignment_55317' (line 627)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 12), 'call_assignment_55317', splitext_call_result_57476)
        
        # Assigning a Call to a Name (line 627):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_57479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 12), 'int')
        # Processing the call keyword arguments
        kwargs_57480 = {}
        # Getting the type of 'call_assignment_55317' (line 627)
        call_assignment_55317_57477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 12), 'call_assignment_55317', False)
        # Obtaining the member '__getitem__' of a type (line 627)
        getitem___57478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 12), call_assignment_55317_57477, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_57481 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___57478, *[int_57479], **kwargs_57480)
        
        # Assigning a type to the variable 'call_assignment_55318' (line 627)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 12), 'call_assignment_55318', getitem___call_result_57481)
        
        # Assigning a Name to a Name (line 627):
        # Getting the type of 'call_assignment_55318' (line 627)
        call_assignment_55318_57482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 12), 'call_assignment_55318')
        # Assigning a type to the variable 'base' (line 627)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 13), 'base', call_assignment_55318_57482)
        
        # Assigning a Call to a Name (line 627):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_57485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 12), 'int')
        # Processing the call keyword arguments
        kwargs_57486 = {}
        # Getting the type of 'call_assignment_55317' (line 627)
        call_assignment_55317_57483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 12), 'call_assignment_55317', False)
        # Obtaining the member '__getitem__' of a type (line 627)
        getitem___57484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 12), call_assignment_55317_57483, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_57487 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___57484, *[int_57485], **kwargs_57486)
        
        # Assigning a type to the variable 'call_assignment_55319' (line 627)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 12), 'call_assignment_55319', getitem___call_result_57487)
        
        # Assigning a Name to a Name (line 627):
        # Getting the type of 'call_assignment_55319' (line 627)
        call_assignment_55319_57488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 12), 'call_assignment_55319')
        # Assigning a type to the variable 'ext' (line 627)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 19), 'ext', call_assignment_55319_57488)
        
        
        # Getting the type of 'ext' (line 628)
        ext_57489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 15), 'ext')
        str_57490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 22), 'str', '.i')
        # Applying the binary operator '==' (line 628)
        result_eq_57491 = python_operator(stypy.reporting.localization.Localization(__file__, 628, 15), '==', ext_57489, str_57490)
        
        # Testing the type of an if condition (line 628)
        if_condition_57492 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 628, 12), result_eq_57491)
        # Assigning a type to the variable 'if_condition_57492' (line 628)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 12), 'if_condition_57492', if_condition_57492)
        # SSA begins for if statement (line 628)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 631)
        self_57493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 19), 'self')
        # Obtaining the member 'inplace' of a type (line 631)
        inplace_57494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 19), self_57493, 'inplace')
        # Testing the type of an if condition (line 631)
        if_condition_57495 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 631, 16), inplace_57494)
        # Assigning a type to the variable 'if_condition_57495' (line 631)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 16), 'if_condition_57495', if_condition_57495)
        # SSA begins for if statement (line 631)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 632):
        
        # Assigning a Call to a Name (line 632):
        
        # Call to dirname(...): (line 632)
        # Processing the call arguments (line 632)
        # Getting the type of 'base' (line 632)
        base_57499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 49), 'base', False)
        # Processing the call keyword arguments (line 632)
        kwargs_57500 = {}
        # Getting the type of 'os' (line 632)
        os_57496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 33), 'os', False)
        # Obtaining the member 'path' of a type (line 632)
        path_57497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 33), os_57496, 'path')
        # Obtaining the member 'dirname' of a type (line 632)
        dirname_57498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 33), path_57497, 'dirname')
        # Calling dirname(args, kwargs) (line 632)
        dirname_call_result_57501 = invoke(stypy.reporting.localization.Localization(__file__, 632, 33), dirname_57498, *[base_57499], **kwargs_57500)
        
        # Assigning a type to the variable 'target_dir' (line 632)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 20), 'target_dir', dirname_call_result_57501)
        
        # Assigning a Attribute to a Name (line 633):
        
        # Assigning a Attribute to a Name (line 633):
        # Getting the type of 'self' (line 633)
        self_57502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 36), 'self')
        # Obtaining the member 'ext_target_dir' of a type (line 633)
        ext_target_dir_57503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 36), self_57502, 'ext_target_dir')
        # Assigning a type to the variable 'py_target_dir' (line 633)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 20), 'py_target_dir', ext_target_dir_57503)
        # SSA branch for the else part of an if statement (line 631)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 635):
        
        # Assigning a Call to a Name (line 635):
        
        # Call to appendpath(...): (line 635)
        # Processing the call arguments (line 635)
        # Getting the type of 'self' (line 635)
        self_57505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 44), 'self', False)
        # Obtaining the member 'build_src' of a type (line 635)
        build_src_57506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 44), self_57505, 'build_src')
        
        # Call to dirname(...): (line 635)
        # Processing the call arguments (line 635)
        # Getting the type of 'base' (line 635)
        base_57510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 76), 'base', False)
        # Processing the call keyword arguments (line 635)
        kwargs_57511 = {}
        # Getting the type of 'os' (line 635)
        os_57507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 60), 'os', False)
        # Obtaining the member 'path' of a type (line 635)
        path_57508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 60), os_57507, 'path')
        # Obtaining the member 'dirname' of a type (line 635)
        dirname_57509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 60), path_57508, 'dirname')
        # Calling dirname(args, kwargs) (line 635)
        dirname_call_result_57512 = invoke(stypy.reporting.localization.Localization(__file__, 635, 60), dirname_57509, *[base_57510], **kwargs_57511)
        
        # Processing the call keyword arguments (line 635)
        kwargs_57513 = {}
        # Getting the type of 'appendpath' (line 635)
        appendpath_57504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 33), 'appendpath', False)
        # Calling appendpath(args, kwargs) (line 635)
        appendpath_call_result_57514 = invoke(stypy.reporting.localization.Localization(__file__, 635, 33), appendpath_57504, *[build_src_57506, dirname_call_result_57512], **kwargs_57513)
        
        # Assigning a type to the variable 'target_dir' (line 635)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 20), 'target_dir', appendpath_call_result_57514)
        
        # Assigning a Name to a Name (line 636):
        
        # Assigning a Name to a Name (line 636):
        # Getting the type of 'target_dir' (line 636)
        target_dir_57515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 36), 'target_dir')
        # Assigning a type to the variable 'py_target_dir' (line 636)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 20), 'py_target_dir', target_dir_57515)
        # SSA join for if statement (line 631)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to isfile(...): (line 637)
        # Processing the call arguments (line 637)
        # Getting the type of 'source' (line 637)
        source_57519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 34), 'source', False)
        # Processing the call keyword arguments (line 637)
        kwargs_57520 = {}
        # Getting the type of 'os' (line 637)
        os_57516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 637)
        path_57517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 19), os_57516, 'path')
        # Obtaining the member 'isfile' of a type (line 637)
        isfile_57518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 19), path_57517, 'isfile')
        # Calling isfile(args, kwargs) (line 637)
        isfile_call_result_57521 = invoke(stypy.reporting.localization.Localization(__file__, 637, 19), isfile_57518, *[source_57519], **kwargs_57520)
        
        # Testing the type of an if condition (line 637)
        if_condition_57522 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 637, 16), isfile_call_result_57521)
        # Assigning a type to the variable 'if_condition_57522' (line 637)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 16), 'if_condition_57522', if_condition_57522)
        # SSA begins for if statement (line 637)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 638):
        
        # Assigning a Call to a Name (line 638):
        
        # Call to get_swig_modulename(...): (line 638)
        # Processing the call arguments (line 638)
        # Getting the type of 'source' (line 638)
        source_57524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 47), 'source', False)
        # Processing the call keyword arguments (line 638)
        kwargs_57525 = {}
        # Getting the type of 'get_swig_modulename' (line 638)
        get_swig_modulename_57523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 27), 'get_swig_modulename', False)
        # Calling get_swig_modulename(args, kwargs) (line 638)
        get_swig_modulename_call_result_57526 = invoke(stypy.reporting.localization.Localization(__file__, 638, 27), get_swig_modulename_57523, *[source_57524], **kwargs_57525)
        
        # Assigning a type to the variable 'name' (line 638)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 20), 'name', get_swig_modulename_call_result_57526)
        
        
        # Getting the type of 'name' (line 639)
        name_57527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 23), 'name')
        
        # Obtaining the type of the subscript
        int_57528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 40), 'int')
        slice_57529 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 639, 31), int_57528, None, None)
        # Getting the type of 'ext_name' (line 639)
        ext_name_57530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 31), 'ext_name')
        # Obtaining the member '__getitem__' of a type (line 639)
        getitem___57531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 31), ext_name_57530, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 639)
        subscript_call_result_57532 = invoke(stypy.reporting.localization.Localization(__file__, 639, 31), getitem___57531, slice_57529)
        
        # Applying the binary operator '!=' (line 639)
        result_ne_57533 = python_operator(stypy.reporting.localization.Localization(__file__, 639, 23), '!=', name_57527, subscript_call_result_57532)
        
        # Testing the type of an if condition (line 639)
        if_condition_57534 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 639, 20), result_ne_57533)
        # Assigning a type to the variable 'if_condition_57534' (line 639)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 20), 'if_condition_57534', if_condition_57534)
        # SSA begins for if statement (line 639)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to DistutilsSetupError(...): (line 640)
        # Processing the call arguments (line 640)
        str_57536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 28), 'str', 'mismatch of extension names: %s provides %r but expected %r')
        
        # Obtaining an instance of the builtin type 'tuple' (line 642)
        tuple_57537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 642)
        # Adding element type (line 642)
        # Getting the type of 'source' (line 642)
        source_57538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 50), 'source', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 642, 50), tuple_57537, source_57538)
        # Adding element type (line 642)
        # Getting the type of 'name' (line 642)
        name_57539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 58), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 642, 50), tuple_57537, name_57539)
        # Adding element type (line 642)
        
        # Obtaining the type of the subscript
        int_57540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, 73), 'int')
        slice_57541 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 642, 64), int_57540, None, None)
        # Getting the type of 'ext_name' (line 642)
        ext_name_57542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 64), 'ext_name', False)
        # Obtaining the member '__getitem__' of a type (line 642)
        getitem___57543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 64), ext_name_57542, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 642)
        subscript_call_result_57544 = invoke(stypy.reporting.localization.Localization(__file__, 642, 64), getitem___57543, slice_57541)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 642, 50), tuple_57537, subscript_call_result_57544)
        
        # Applying the binary operator '%' (line 641)
        result_mod_57545 = python_operator(stypy.reporting.localization.Localization(__file__, 641, 28), '%', str_57536, tuple_57537)
        
        # Processing the call keyword arguments (line 640)
        kwargs_57546 = {}
        # Getting the type of 'DistutilsSetupError' (line 640)
        DistutilsSetupError_57535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 30), 'DistutilsSetupError', False)
        # Calling DistutilsSetupError(args, kwargs) (line 640)
        DistutilsSetupError_call_result_57547 = invoke(stypy.reporting.localization.Localization(__file__, 640, 30), DistutilsSetupError_57535, *[result_mod_57545], **kwargs_57546)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 640, 24), DistutilsSetupError_call_result_57547, 'raise parameter', BaseException)
        # SSA join for if statement (line 639)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 643)
        # Getting the type of 'typ' (line 643)
        typ_57548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 23), 'typ')
        # Getting the type of 'None' (line 643)
        None_57549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 30), 'None')
        
        (may_be_57550, more_types_in_union_57551) = may_be_none(typ_57548, None_57549)

        if may_be_57550:

            if more_types_in_union_57551:
                # Runtime conditional SSA (line 643)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 644):
            
            # Assigning a Call to a Name (line 644):
            
            # Call to get_swig_target(...): (line 644)
            # Processing the call arguments (line 644)
            # Getting the type of 'source' (line 644)
            source_57553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 46), 'source', False)
            # Processing the call keyword arguments (line 644)
            kwargs_57554 = {}
            # Getting the type of 'get_swig_target' (line 644)
            get_swig_target_57552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 30), 'get_swig_target', False)
            # Calling get_swig_target(args, kwargs) (line 644)
            get_swig_target_call_result_57555 = invoke(stypy.reporting.localization.Localization(__file__, 644, 30), get_swig_target_57552, *[source_57553], **kwargs_57554)
            
            # Assigning a type to the variable 'typ' (line 644)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 24), 'typ', get_swig_target_call_result_57555)
            
            # Assigning a Compare to a Name (line 645):
            
            # Assigning a Compare to a Name (line 645):
            
            # Getting the type of 'typ' (line 645)
            typ_57556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 33), 'typ')
            str_57557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 645, 38), 'str', 'c++')
            # Applying the binary operator '==' (line 645)
            result_eq_57558 = python_operator(stypy.reporting.localization.Localization(__file__, 645, 33), '==', typ_57556, str_57557)
            
            # Assigning a type to the variable 'is_cpp' (line 645)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 24), 'is_cpp', result_eq_57558)

            if more_types_in_union_57551:
                # Runtime conditional SSA for else branch (line 643)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_57550) or more_types_in_union_57551):
            
            # Assigning a Call to a Name (line 647):
            
            # Assigning a Call to a Name (line 647):
            
            # Call to get_swig_target(...): (line 647)
            # Processing the call arguments (line 647)
            # Getting the type of 'source' (line 647)
            source_57560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 47), 'source', False)
            # Processing the call keyword arguments (line 647)
            kwargs_57561 = {}
            # Getting the type of 'get_swig_target' (line 647)
            get_swig_target_57559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 31), 'get_swig_target', False)
            # Calling get_swig_target(args, kwargs) (line 647)
            get_swig_target_call_result_57562 = invoke(stypy.reporting.localization.Localization(__file__, 647, 31), get_swig_target_57559, *[source_57560], **kwargs_57561)
            
            # Assigning a type to the variable 'typ2' (line 647)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 24), 'typ2', get_swig_target_call_result_57562)
            
            # Type idiom detected: calculating its left and rigth part (line 648)
            # Getting the type of 'typ2' (line 648)
            typ2_57563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 27), 'typ2')
            # Getting the type of 'None' (line 648)
            None_57564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 35), 'None')
            
            (may_be_57565, more_types_in_union_57566) = may_be_none(typ2_57563, None_57564)

            if may_be_57565:

                if more_types_in_union_57566:
                    # Runtime conditional SSA (line 648)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to warn(...): (line 649)
                # Processing the call arguments (line 649)
                str_57569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 37), 'str', 'source %r does not define swig target, assuming %s swig target')
                
                # Obtaining an instance of the builtin type 'tuple' (line 650)
                tuple_57570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 40), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 650)
                # Adding element type (line 650)
                # Getting the type of 'source' (line 650)
                source_57571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 40), 'source', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 650, 40), tuple_57570, source_57571)
                # Adding element type (line 650)
                # Getting the type of 'typ' (line 650)
                typ_57572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 48), 'typ', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 650, 40), tuple_57570, typ_57572)
                
                # Applying the binary operator '%' (line 649)
                result_mod_57573 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 37), '%', str_57569, tuple_57570)
                
                # Processing the call keyword arguments (line 649)
                kwargs_57574 = {}
                # Getting the type of 'log' (line 649)
                log_57567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 28), 'log', False)
                # Obtaining the member 'warn' of a type (line 649)
                warn_57568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 28), log_57567, 'warn')
                # Calling warn(args, kwargs) (line 649)
                warn_call_result_57575 = invoke(stypy.reporting.localization.Localization(__file__, 649, 28), warn_57568, *[result_mod_57573], **kwargs_57574)
                

                if more_types_in_union_57566:
                    # Runtime conditional SSA for else branch (line 648)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_57565) or more_types_in_union_57566):
                
                
                # Getting the type of 'typ' (line 651)
                typ_57576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 29), 'typ')
                # Getting the type of 'typ2' (line 651)
                typ2_57577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 34), 'typ2')
                # Applying the binary operator '!=' (line 651)
                result_ne_57578 = python_operator(stypy.reporting.localization.Localization(__file__, 651, 29), '!=', typ_57576, typ2_57577)
                
                # Testing the type of an if condition (line 651)
                if_condition_57579 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 651, 29), result_ne_57578)
                # Assigning a type to the variable 'if_condition_57579' (line 651)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 29), 'if_condition_57579', if_condition_57579)
                # SSA begins for if statement (line 651)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to warn(...): (line 652)
                # Processing the call arguments (line 652)
                str_57582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 37), 'str', 'expected %r but source %r defines %r swig target')
                
                # Obtaining an instance of the builtin type 'tuple' (line 653)
                tuple_57583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 40), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 653)
                # Adding element type (line 653)
                # Getting the type of 'typ' (line 653)
                typ_57584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 40), 'typ', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 653, 40), tuple_57583, typ_57584)
                # Adding element type (line 653)
                # Getting the type of 'source' (line 653)
                source_57585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 45), 'source', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 653, 40), tuple_57583, source_57585)
                # Adding element type (line 653)
                # Getting the type of 'typ2' (line 653)
                typ2_57586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 53), 'typ2', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 653, 40), tuple_57583, typ2_57586)
                
                # Applying the binary operator '%' (line 652)
                result_mod_57587 = python_operator(stypy.reporting.localization.Localization(__file__, 652, 37), '%', str_57582, tuple_57583)
                
                # Processing the call keyword arguments (line 652)
                kwargs_57588 = {}
                # Getting the type of 'log' (line 652)
                log_57580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 28), 'log', False)
                # Obtaining the member 'warn' of a type (line 652)
                warn_57581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 28), log_57580, 'warn')
                # Calling warn(args, kwargs) (line 652)
                warn_call_result_57589 = invoke(stypy.reporting.localization.Localization(__file__, 652, 28), warn_57581, *[result_mod_57587], **kwargs_57588)
                
                
                
                # Getting the type of 'typ2' (line 654)
                typ2_57590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 31), 'typ2')
                str_57591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 37), 'str', 'c++')
                # Applying the binary operator '==' (line 654)
                result_eq_57592 = python_operator(stypy.reporting.localization.Localization(__file__, 654, 31), '==', typ2_57590, str_57591)
                
                # Testing the type of an if condition (line 654)
                if_condition_57593 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 654, 28), result_eq_57592)
                # Assigning a type to the variable 'if_condition_57593' (line 654)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 28), 'if_condition_57593', if_condition_57593)
                # SSA begins for if statement (line 654)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to warn(...): (line 655)
                # Processing the call arguments (line 655)
                str_57596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 41), 'str', 'resetting swig target to c++ (some targets may have .c extension)')
                # Processing the call keyword arguments (line 655)
                kwargs_57597 = {}
                # Getting the type of 'log' (line 655)
                log_57594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 32), 'log', False)
                # Obtaining the member 'warn' of a type (line 655)
                warn_57595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 32), log_57594, 'warn')
                # Calling warn(args, kwargs) (line 655)
                warn_call_result_57598 = invoke(stypy.reporting.localization.Localization(__file__, 655, 32), warn_57595, *[str_57596], **kwargs_57597)
                
                
                # Assigning a Name to a Name (line 656):
                
                # Assigning a Name to a Name (line 656):
                # Getting the type of 'True' (line 656)
                True_57599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 41), 'True')
                # Assigning a type to the variable 'is_cpp' (line 656)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 32), 'is_cpp', True_57599)
                # SSA branch for the else part of an if statement (line 654)
                module_type_store.open_ssa_branch('else')
                
                # Call to warn(...): (line 658)
                # Processing the call arguments (line 658)
                str_57602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 41), 'str', 'assuming that %r has c++ swig target')
                # Getting the type of 'source' (line 658)
                source_57603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 83), 'source', False)
                # Applying the binary operator '%' (line 658)
                result_mod_57604 = python_operator(stypy.reporting.localization.Localization(__file__, 658, 41), '%', str_57602, source_57603)
                
                # Processing the call keyword arguments (line 658)
                kwargs_57605 = {}
                # Getting the type of 'log' (line 658)
                log_57600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 32), 'log', False)
                # Obtaining the member 'warn' of a type (line 658)
                warn_57601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 32), log_57600, 'warn')
                # Calling warn(args, kwargs) (line 658)
                warn_call_result_57606 = invoke(stypy.reporting.localization.Localization(__file__, 658, 32), warn_57601, *[result_mod_57604], **kwargs_57605)
                
                # SSA join for if statement (line 654)
                module_type_store = module_type_store.join_ssa_context()
                
                # SSA join for if statement (line 651)
                module_type_store = module_type_store.join_ssa_context()
                

                if (may_be_57565 and more_types_in_union_57566):
                    # SSA join for if statement (line 648)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_57550 and more_types_in_union_57551):
                # SSA join for if statement (line 643)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'is_cpp' (line 659)
        is_cpp_57607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 23), 'is_cpp')
        # Testing the type of an if condition (line 659)
        if_condition_57608 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 659, 20), is_cpp_57607)
        # Assigning a type to the variable 'if_condition_57608' (line 659)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 20), 'if_condition_57608', if_condition_57608)
        # SSA begins for if statement (line 659)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 660):
        
        # Assigning a Str to a Name (line 660):
        str_57609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 37), 'str', '.cpp')
        # Assigning a type to the variable 'target_ext' (line 660)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 24), 'target_ext', str_57609)
        # SSA join for if statement (line 659)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 661):
        
        # Assigning a Call to a Name (line 661):
        
        # Call to join(...): (line 661)
        # Processing the call arguments (line 661)
        # Getting the type of 'target_dir' (line 661)
        target_dir_57613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 47), 'target_dir', False)
        str_57614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 59), 'str', '%s_wrap%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 662)
        tuple_57615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 662)
        # Adding element type (line 662)
        # Getting the type of 'name' (line 662)
        name_57616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 50), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 662, 50), tuple_57615, name_57616)
        # Adding element type (line 662)
        # Getting the type of 'target_ext' (line 662)
        target_ext_57617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 56), 'target_ext', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 662, 50), tuple_57615, target_ext_57617)
        
        # Applying the binary operator '%' (line 661)
        result_mod_57618 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 59), '%', str_57614, tuple_57615)
        
        # Processing the call keyword arguments (line 661)
        kwargs_57619 = {}
        # Getting the type of 'os' (line 661)
        os_57610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 34), 'os', False)
        # Obtaining the member 'path' of a type (line 661)
        path_57611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 34), os_57610, 'path')
        # Obtaining the member 'join' of a type (line 661)
        join_57612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 34), path_57611, 'join')
        # Calling join(args, kwargs) (line 661)
        join_call_result_57620 = invoke(stypy.reporting.localization.Localization(__file__, 661, 34), join_57612, *[target_dir_57613, result_mod_57618], **kwargs_57619)
        
        # Assigning a type to the variable 'target_file' (line 661)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 20), 'target_file', join_call_result_57620)
        # SSA branch for the else part of an if statement (line 637)
        module_type_store.open_ssa_branch('else')
        
        # Call to warn(...): (line 664)
        # Processing the call arguments (line 664)
        str_57623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 29), 'str', "  source %s does not exist: skipping swig'ing.")
        # Getting the type of 'source' (line 665)
        source_57624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 32), 'source', False)
        # Applying the binary operator '%' (line 664)
        result_mod_57625 = python_operator(stypy.reporting.localization.Localization(__file__, 664, 29), '%', str_57623, source_57624)
        
        # Processing the call keyword arguments (line 664)
        kwargs_57626 = {}
        # Getting the type of 'log' (line 664)
        log_57621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 20), 'log', False)
        # Obtaining the member 'warn' of a type (line 664)
        warn_57622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 20), log_57621, 'warn')
        # Calling warn(args, kwargs) (line 664)
        warn_call_result_57627 = invoke(stypy.reporting.localization.Localization(__file__, 664, 20), warn_57622, *[result_mod_57625], **kwargs_57626)
        
        
        # Assigning a Subscript to a Name (line 666):
        
        # Assigning a Subscript to a Name (line 666):
        
        # Obtaining the type of the subscript
        int_57628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 36), 'int')
        slice_57629 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 666, 27), int_57628, None, None)
        # Getting the type of 'ext_name' (line 666)
        ext_name_57630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 27), 'ext_name')
        # Obtaining the member '__getitem__' of a type (line 666)
        getitem___57631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 27), ext_name_57630, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 666)
        subscript_call_result_57632 = invoke(stypy.reporting.localization.Localization(__file__, 666, 27), getitem___57631, slice_57629)
        
        # Assigning a type to the variable 'name' (line 666)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 20), 'name', subscript_call_result_57632)
        
        # Assigning a Num to a Name (line 667):
        
        # Assigning a Num to a Name (line 667):
        int_57633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 32), 'int')
        # Assigning a type to the variable 'skip_swig' (line 667)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 20), 'skip_swig', int_57633)
        
        # Assigning a Call to a Name (line 668):
        
        # Assigning a Call to a Name (line 668):
        
        # Call to _find_swig_target(...): (line 668)
        # Processing the call arguments (line 668)
        # Getting the type of 'target_dir' (line 668)
        target_dir_57635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 52), 'target_dir', False)
        # Getting the type of 'name' (line 668)
        name_57636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 64), 'name', False)
        # Processing the call keyword arguments (line 668)
        kwargs_57637 = {}
        # Getting the type of '_find_swig_target' (line 668)
        _find_swig_target_57634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 34), '_find_swig_target', False)
        # Calling _find_swig_target(args, kwargs) (line 668)
        _find_swig_target_call_result_57638 = invoke(stypy.reporting.localization.Localization(__file__, 668, 34), _find_swig_target_57634, *[target_dir_57635, name_57636], **kwargs_57637)
        
        # Assigning a type to the variable 'target_file' (line 668)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 20), 'target_file', _find_swig_target_call_result_57638)
        
        
        
        # Call to isfile(...): (line 669)
        # Processing the call arguments (line 669)
        # Getting the type of 'target_file' (line 669)
        target_file_57642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 42), 'target_file', False)
        # Processing the call keyword arguments (line 669)
        kwargs_57643 = {}
        # Getting the type of 'os' (line 669)
        os_57639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 669)
        path_57640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 27), os_57639, 'path')
        # Obtaining the member 'isfile' of a type (line 669)
        isfile_57641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 27), path_57640, 'isfile')
        # Calling isfile(args, kwargs) (line 669)
        isfile_call_result_57644 = invoke(stypy.reporting.localization.Localization(__file__, 669, 27), isfile_57641, *[target_file_57642], **kwargs_57643)
        
        # Applying the 'not' unary operator (line 669)
        result_not__57645 = python_operator(stypy.reporting.localization.Localization(__file__, 669, 23), 'not', isfile_call_result_57644)
        
        # Testing the type of an if condition (line 669)
        if_condition_57646 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 669, 20), result_not__57645)
        # Assigning a type to the variable 'if_condition_57646' (line 669)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 20), 'if_condition_57646', if_condition_57646)
        # SSA begins for if statement (line 669)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 670)
        # Processing the call arguments (line 670)
        str_57649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 670, 33), 'str', '  target %s does not exist:\n   Assuming %s_wrap.{c,cpp} was generated with "build_src --inplace" command.')
        
        # Obtaining an instance of the builtin type 'tuple' (line 673)
        tuple_57650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 673)
        # Adding element type (line 673)
        # Getting the type of 'target_file' (line 673)
        target_file_57651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 36), 'target_file', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 673, 36), tuple_57650, target_file_57651)
        # Adding element type (line 673)
        # Getting the type of 'name' (line 673)
        name_57652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 49), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 673, 36), tuple_57650, name_57652)
        
        # Applying the binary operator '%' (line 670)
        result_mod_57653 = python_operator(stypy.reporting.localization.Localization(__file__, 670, 33), '%', str_57649, tuple_57650)
        
        # Processing the call keyword arguments (line 670)
        kwargs_57654 = {}
        # Getting the type of 'log' (line 670)
        log_57647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 24), 'log', False)
        # Obtaining the member 'warn' of a type (line 670)
        warn_57648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 24), log_57647, 'warn')
        # Calling warn(args, kwargs) (line 670)
        warn_call_result_57655 = invoke(stypy.reporting.localization.Localization(__file__, 670, 24), warn_57648, *[result_mod_57653], **kwargs_57654)
        
        
        # Assigning a Call to a Name (line 674):
        
        # Assigning a Call to a Name (line 674):
        
        # Call to dirname(...): (line 674)
        # Processing the call arguments (line 674)
        # Getting the type of 'base' (line 674)
        base_57659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 53), 'base', False)
        # Processing the call keyword arguments (line 674)
        kwargs_57660 = {}
        # Getting the type of 'os' (line 674)
        os_57656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 37), 'os', False)
        # Obtaining the member 'path' of a type (line 674)
        path_57657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 37), os_57656, 'path')
        # Obtaining the member 'dirname' of a type (line 674)
        dirname_57658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 37), path_57657, 'dirname')
        # Calling dirname(args, kwargs) (line 674)
        dirname_call_result_57661 = invoke(stypy.reporting.localization.Localization(__file__, 674, 37), dirname_57658, *[base_57659], **kwargs_57660)
        
        # Assigning a type to the variable 'target_dir' (line 674)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 24), 'target_dir', dirname_call_result_57661)
        
        # Assigning a Call to a Name (line 675):
        
        # Assigning a Call to a Name (line 675):
        
        # Call to _find_swig_target(...): (line 675)
        # Processing the call arguments (line 675)
        # Getting the type of 'target_dir' (line 675)
        target_dir_57663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 56), 'target_dir', False)
        # Getting the type of 'name' (line 675)
        name_57664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 68), 'name', False)
        # Processing the call keyword arguments (line 675)
        kwargs_57665 = {}
        # Getting the type of '_find_swig_target' (line 675)
        _find_swig_target_57662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 38), '_find_swig_target', False)
        # Calling _find_swig_target(args, kwargs) (line 675)
        _find_swig_target_call_result_57666 = invoke(stypy.reporting.localization.Localization(__file__, 675, 38), _find_swig_target_57662, *[target_dir_57663, name_57664], **kwargs_57665)
        
        # Assigning a type to the variable 'target_file' (line 675)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 24), 'target_file', _find_swig_target_call_result_57666)
        
        
        
        # Call to isfile(...): (line 676)
        # Processing the call arguments (line 676)
        # Getting the type of 'target_file' (line 676)
        target_file_57670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 46), 'target_file', False)
        # Processing the call keyword arguments (line 676)
        kwargs_57671 = {}
        # Getting the type of 'os' (line 676)
        os_57667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 31), 'os', False)
        # Obtaining the member 'path' of a type (line 676)
        path_57668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 31), os_57667, 'path')
        # Obtaining the member 'isfile' of a type (line 676)
        isfile_57669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 31), path_57668, 'isfile')
        # Calling isfile(args, kwargs) (line 676)
        isfile_call_result_57672 = invoke(stypy.reporting.localization.Localization(__file__, 676, 31), isfile_57669, *[target_file_57670], **kwargs_57671)
        
        # Applying the 'not' unary operator (line 676)
        result_not__57673 = python_operator(stypy.reporting.localization.Localization(__file__, 676, 27), 'not', isfile_call_result_57672)
        
        # Testing the type of an if condition (line 676)
        if_condition_57674 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 676, 24), result_not__57673)
        # Assigning a type to the variable 'if_condition_57674' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 24), 'if_condition_57674', if_condition_57674)
        # SSA begins for if statement (line 676)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to DistutilsSetupError(...): (line 677)
        # Processing the call arguments (line 677)
        str_57676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 54), 'str', '%r missing')
        
        # Obtaining an instance of the builtin type 'tuple' (line 677)
        tuple_57677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 70), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 677)
        # Adding element type (line 677)
        # Getting the type of 'target_file' (line 677)
        target_file_57678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 70), 'target_file', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 677, 70), tuple_57677, target_file_57678)
        
        # Applying the binary operator '%' (line 677)
        result_mod_57679 = python_operator(stypy.reporting.localization.Localization(__file__, 677, 54), '%', str_57676, tuple_57677)
        
        # Processing the call keyword arguments (line 677)
        kwargs_57680 = {}
        # Getting the type of 'DistutilsSetupError' (line 677)
        DistutilsSetupError_57675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 34), 'DistutilsSetupError', False)
        # Calling DistutilsSetupError(args, kwargs) (line 677)
        DistutilsSetupError_call_result_57681 = invoke(stypy.reporting.localization.Localization(__file__, 677, 34), DistutilsSetupError_57675, *[result_mod_57679], **kwargs_57680)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 677, 28), DistutilsSetupError_call_result_57681, 'raise parameter', BaseException)
        # SSA join for if statement (line 676)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to warn(...): (line 678)
        # Processing the call arguments (line 678)
        str_57684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 33), 'str', '   Yes! Using %r as up-to-date target.')
        # Getting the type of 'target_file' (line 679)
        target_file_57685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 36), 'target_file', False)
        # Applying the binary operator '%' (line 678)
        result_mod_57686 = python_operator(stypy.reporting.localization.Localization(__file__, 678, 33), '%', str_57684, target_file_57685)
        
        # Processing the call keyword arguments (line 678)
        kwargs_57687 = {}
        # Getting the type of 'log' (line 678)
        log_57682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 24), 'log', False)
        # Obtaining the member 'warn' of a type (line 678)
        warn_57683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 24), log_57682, 'warn')
        # Calling warn(args, kwargs) (line 678)
        warn_call_result_57688 = invoke(stypy.reporting.localization.Localization(__file__, 678, 24), warn_57683, *[result_mod_57686], **kwargs_57687)
        
        # SSA join for if statement (line 669)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 637)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 680)
        # Processing the call arguments (line 680)
        # Getting the type of 'target_dir' (line 680)
        target_dir_57691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 35), 'target_dir', False)
        # Processing the call keyword arguments (line 680)
        kwargs_57692 = {}
        # Getting the type of 'target_dirs' (line 680)
        target_dirs_57689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 16), 'target_dirs', False)
        # Obtaining the member 'append' of a type (line 680)
        append_57690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 16), target_dirs_57689, 'append')
        # Calling append(args, kwargs) (line 680)
        append_call_result_57693 = invoke(stypy.reporting.localization.Localization(__file__, 680, 16), append_57690, *[target_dir_57691], **kwargs_57692)
        
        
        # Call to append(...): (line 681)
        # Processing the call arguments (line 681)
        # Getting the type of 'target_file' (line 681)
        target_file_57696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 35), 'target_file', False)
        # Processing the call keyword arguments (line 681)
        kwargs_57697 = {}
        # Getting the type of 'new_sources' (line 681)
        new_sources_57694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 16), 'new_sources', False)
        # Obtaining the member 'append' of a type (line 681)
        append_57695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 16), new_sources_57694, 'append')
        # Calling append(args, kwargs) (line 681)
        append_call_result_57698 = invoke(stypy.reporting.localization.Localization(__file__, 681, 16), append_57695, *[target_file_57696], **kwargs_57697)
        
        
        # Call to append(...): (line 682)
        # Processing the call arguments (line 682)
        
        # Call to join(...): (line 682)
        # Processing the call arguments (line 682)
        # Getting the type of 'py_target_dir' (line 682)
        py_target_dir_57704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 45), 'py_target_dir', False)
        # Getting the type of 'name' (line 682)
        name_57705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 60), 'name', False)
        str_57706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 65), 'str', '.py')
        # Applying the binary operator '+' (line 682)
        result_add_57707 = python_operator(stypy.reporting.localization.Localization(__file__, 682, 60), '+', name_57705, str_57706)
        
        # Processing the call keyword arguments (line 682)
        kwargs_57708 = {}
        # Getting the type of 'os' (line 682)
        os_57701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 32), 'os', False)
        # Obtaining the member 'path' of a type (line 682)
        path_57702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 32), os_57701, 'path')
        # Obtaining the member 'join' of a type (line 682)
        join_57703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 32), path_57702, 'join')
        # Calling join(args, kwargs) (line 682)
        join_call_result_57709 = invoke(stypy.reporting.localization.Localization(__file__, 682, 32), join_57703, *[py_target_dir_57704, result_add_57707], **kwargs_57708)
        
        # Processing the call keyword arguments (line 682)
        kwargs_57710 = {}
        # Getting the type of 'py_files' (line 682)
        py_files_57699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 16), 'py_files', False)
        # Obtaining the member 'append' of a type (line 682)
        append_57700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 16), py_files_57699, 'append')
        # Calling append(args, kwargs) (line 682)
        append_call_result_57711 = invoke(stypy.reporting.localization.Localization(__file__, 682, 16), append_57700, *[join_call_result_57709], **kwargs_57710)
        
        
        # Call to append(...): (line 683)
        # Processing the call arguments (line 683)
        # Getting the type of 'source' (line 683)
        source_57714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 36), 'source', False)
        # Processing the call keyword arguments (line 683)
        kwargs_57715 = {}
        # Getting the type of 'swig_sources' (line 683)
        swig_sources_57712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 16), 'swig_sources', False)
        # Obtaining the member 'append' of a type (line 683)
        append_57713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 683, 16), swig_sources_57712, 'append')
        # Calling append(args, kwargs) (line 683)
        append_call_result_57716 = invoke(stypy.reporting.localization.Localization(__file__, 683, 16), append_57713, *[source_57714], **kwargs_57715)
        
        
        # Assigning a Subscript to a Subscript (line 684):
        
        # Assigning a Subscript to a Subscript (line 684):
        
        # Obtaining the type of the subscript
        int_57717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 51), 'int')
        # Getting the type of 'new_sources' (line 684)
        new_sources_57718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 39), 'new_sources')
        # Obtaining the member '__getitem__' of a type (line 684)
        getitem___57719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 684, 39), new_sources_57718, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 684)
        subscript_call_result_57720 = invoke(stypy.reporting.localization.Localization(__file__, 684, 39), getitem___57719, int_57717)
        
        # Getting the type of 'swig_targets' (line 684)
        swig_targets_57721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 16), 'swig_targets')
        # Getting the type of 'source' (line 684)
        source_57722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 29), 'source')
        # Storing an element on a container (line 684)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 684, 16), swig_targets_57721, (source_57722, subscript_call_result_57720))
        # SSA branch for the else part of an if statement (line 628)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 686)
        # Processing the call arguments (line 686)
        # Getting the type of 'source' (line 686)
        source_57725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 35), 'source', False)
        # Processing the call keyword arguments (line 686)
        kwargs_57726 = {}
        # Getting the type of 'new_sources' (line 686)
        new_sources_57723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 16), 'new_sources', False)
        # Obtaining the member 'append' of a type (line 686)
        append_57724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 16), new_sources_57723, 'append')
        # Calling append(args, kwargs) (line 686)
        append_call_result_57727 = invoke(stypy.reporting.localization.Localization(__file__, 686, 16), append_57724, *[source_57725], **kwargs_57726)
        
        # SSA join for if statement (line 628)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'swig_sources' (line 688)
        swig_sources_57728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 15), 'swig_sources')
        # Applying the 'not' unary operator (line 688)
        result_not__57729 = python_operator(stypy.reporting.localization.Localization(__file__, 688, 11), 'not', swig_sources_57728)
        
        # Testing the type of an if condition (line 688)
        if_condition_57730 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 688, 8), result_not__57729)
        # Assigning a type to the variable 'if_condition_57730' (line 688)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 688, 8), 'if_condition_57730', if_condition_57730)
        # SSA begins for if statement (line 688)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'new_sources' (line 689)
        new_sources_57731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 19), 'new_sources')
        # Assigning a type to the variable 'stypy_return_type' (line 689)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 12), 'stypy_return_type', new_sources_57731)
        # SSA join for if statement (line 688)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'skip_swig' (line 691)
        skip_swig_57732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 11), 'skip_swig')
        # Testing the type of an if condition (line 691)
        if_condition_57733 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 691, 8), skip_swig_57732)
        # Assigning a type to the variable 'if_condition_57733' (line 691)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 8), 'if_condition_57733', if_condition_57733)
        # SSA begins for if statement (line 691)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'new_sources' (line 692)
        new_sources_57734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 19), 'new_sources')
        # Getting the type of 'py_files' (line 692)
        py_files_57735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 33), 'py_files')
        # Applying the binary operator '+' (line 692)
        result_add_57736 = python_operator(stypy.reporting.localization.Localization(__file__, 692, 19), '+', new_sources_57734, py_files_57735)
        
        # Assigning a type to the variable 'stypy_return_type' (line 692)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 12), 'stypy_return_type', result_add_57736)
        # SSA join for if statement (line 691)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'target_dirs' (line 694)
        target_dirs_57737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 17), 'target_dirs')
        # Testing the type of a for loop iterable (line 694)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 694, 8), target_dirs_57737)
        # Getting the type of the for loop variable (line 694)
        for_loop_var_57738 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 694, 8), target_dirs_57737)
        # Assigning a type to the variable 'd' (line 694)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 8), 'd', for_loop_var_57738)
        # SSA begins for a for statement (line 694)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to mkpath(...): (line 695)
        # Processing the call arguments (line 695)
        # Getting the type of 'd' (line 695)
        d_57741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 24), 'd', False)
        # Processing the call keyword arguments (line 695)
        kwargs_57742 = {}
        # Getting the type of 'self' (line 695)
        self_57739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 12), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 695)
        mkpath_57740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 12), self_57739, 'mkpath')
        # Calling mkpath(args, kwargs) (line 695)
        mkpath_call_result_57743 = invoke(stypy.reporting.localization.Localization(__file__, 695, 12), mkpath_57740, *[d_57741], **kwargs_57742)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BoolOp to a Name (line 697):
        
        # Assigning a BoolOp to a Name (line 697):
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 697)
        self_57744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 15), 'self')
        # Obtaining the member 'swig' of a type (line 697)
        swig_57745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 15), self_57744, 'swig')
        
        # Call to find_swig(...): (line 697)
        # Processing the call keyword arguments (line 697)
        kwargs_57748 = {}
        # Getting the type of 'self' (line 697)
        self_57746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 28), 'self', False)
        # Obtaining the member 'find_swig' of a type (line 697)
        find_swig_57747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 28), self_57746, 'find_swig')
        # Calling find_swig(args, kwargs) (line 697)
        find_swig_call_result_57749 = invoke(stypy.reporting.localization.Localization(__file__, 697, 28), find_swig_57747, *[], **kwargs_57748)
        
        # Applying the binary operator 'or' (line 697)
        result_or_keyword_57750 = python_operator(stypy.reporting.localization.Localization(__file__, 697, 15), 'or', swig_57745, find_swig_call_result_57749)
        
        # Assigning a type to the variable 'swig' (line 697)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 697, 8), 'swig', result_or_keyword_57750)
        
        # Assigning a BinOp to a Name (line 698):
        
        # Assigning a BinOp to a Name (line 698):
        
        # Obtaining an instance of the builtin type 'list' (line 698)
        list_57751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 698)
        # Adding element type (line 698)
        # Getting the type of 'swig' (line 698)
        swig_57752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 20), 'swig')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 698, 19), list_57751, swig_57752)
        # Adding element type (line 698)
        str_57753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 26), 'str', '-python')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 698, 19), list_57751, str_57753)
        
        # Getting the type of 'extension' (line 698)
        extension_57754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 39), 'extension')
        # Obtaining the member 'swig_opts' of a type (line 698)
        swig_opts_57755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 39), extension_57754, 'swig_opts')
        # Applying the binary operator '+' (line 698)
        result_add_57756 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 19), '+', list_57751, swig_opts_57755)
        
        # Assigning a type to the variable 'swig_cmd' (line 698)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 8), 'swig_cmd', result_add_57756)
        
        # Getting the type of 'is_cpp' (line 699)
        is_cpp_57757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 11), 'is_cpp')
        # Testing the type of an if condition (line 699)
        if_condition_57758 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 699, 8), is_cpp_57757)
        # Assigning a type to the variable 'if_condition_57758' (line 699)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 8), 'if_condition_57758', if_condition_57758)
        # SSA begins for if statement (line 699)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 700)
        # Processing the call arguments (line 700)
        str_57761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 28), 'str', '-c++')
        # Processing the call keyword arguments (line 700)
        kwargs_57762 = {}
        # Getting the type of 'swig_cmd' (line 700)
        swig_cmd_57759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 12), 'swig_cmd', False)
        # Obtaining the member 'append' of a type (line 700)
        append_57760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 12), swig_cmd_57759, 'append')
        # Calling append(args, kwargs) (line 700)
        append_call_result_57763 = invoke(stypy.reporting.localization.Localization(__file__, 700, 12), append_57760, *[str_57761], **kwargs_57762)
        
        # SSA join for if statement (line 699)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'extension' (line 701)
        extension_57764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 17), 'extension')
        # Obtaining the member 'include_dirs' of a type (line 701)
        include_dirs_57765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 701, 17), extension_57764, 'include_dirs')
        # Testing the type of a for loop iterable (line 701)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 701, 8), include_dirs_57765)
        # Getting the type of the for loop variable (line 701)
        for_loop_var_57766 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 701, 8), include_dirs_57765)
        # Assigning a type to the variable 'd' (line 701)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 701, 8), 'd', for_loop_var_57766)
        # SSA begins for a for statement (line 701)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 702)
        # Processing the call arguments (line 702)
        str_57769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 28), 'str', '-I')
        # Getting the type of 'd' (line 702)
        d_57770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 33), 'd', False)
        # Applying the binary operator '+' (line 702)
        result_add_57771 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 28), '+', str_57769, d_57770)
        
        # Processing the call keyword arguments (line 702)
        kwargs_57772 = {}
        # Getting the type of 'swig_cmd' (line 702)
        swig_cmd_57767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 12), 'swig_cmd', False)
        # Obtaining the member 'append' of a type (line 702)
        append_57768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 12), swig_cmd_57767, 'append')
        # Calling append(args, kwargs) (line 702)
        append_call_result_57773 = invoke(stypy.reporting.localization.Localization(__file__, 702, 12), append_57768, *[result_add_57771], **kwargs_57772)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'swig_sources' (line 703)
        swig_sources_57774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 22), 'swig_sources')
        # Testing the type of a for loop iterable (line 703)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 703, 8), swig_sources_57774)
        # Getting the type of the for loop variable (line 703)
        for_loop_var_57775 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 703, 8), swig_sources_57774)
        # Assigning a type to the variable 'source' (line 703)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 8), 'source', for_loop_var_57775)
        # SSA begins for a for statement (line 703)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 704):
        
        # Assigning a Subscript to a Name (line 704):
        
        # Obtaining the type of the subscript
        # Getting the type of 'source' (line 704)
        source_57776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 34), 'source')
        # Getting the type of 'swig_targets' (line 704)
        swig_targets_57777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 21), 'swig_targets')
        # Obtaining the member '__getitem__' of a type (line 704)
        getitem___57778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 21), swig_targets_57777, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 704)
        subscript_call_result_57779 = invoke(stypy.reporting.localization.Localization(__file__, 704, 21), getitem___57778, source_57776)
        
        # Assigning a type to the variable 'target' (line 704)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 12), 'target', subscript_call_result_57779)
        
        # Assigning a BinOp to a Name (line 705):
        
        # Assigning a BinOp to a Name (line 705):
        
        # Obtaining an instance of the builtin type 'list' (line 705)
        list_57780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 705)
        # Adding element type (line 705)
        # Getting the type of 'source' (line 705)
        source_57781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 23), 'source')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 705, 22), list_57780, source_57781)
        
        # Getting the type of 'extension' (line 705)
        extension_57782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 33), 'extension')
        # Obtaining the member 'depends' of a type (line 705)
        depends_57783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 33), extension_57782, 'depends')
        # Applying the binary operator '+' (line 705)
        result_add_57784 = python_operator(stypy.reporting.localization.Localization(__file__, 705, 22), '+', list_57780, depends_57783)
        
        # Assigning a type to the variable 'depends' (line 705)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 12), 'depends', result_add_57784)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 706)
        self_57785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 15), 'self')
        # Obtaining the member 'force' of a type (line 706)
        force_57786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 706, 15), self_57785, 'force')
        
        # Call to newer_group(...): (line 706)
        # Processing the call arguments (line 706)
        # Getting the type of 'depends' (line 706)
        depends_57788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 41), 'depends', False)
        # Getting the type of 'target' (line 706)
        target_57789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 50), 'target', False)
        str_57790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 58), 'str', 'newer')
        # Processing the call keyword arguments (line 706)
        kwargs_57791 = {}
        # Getting the type of 'newer_group' (line 706)
        newer_group_57787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 29), 'newer_group', False)
        # Calling newer_group(args, kwargs) (line 706)
        newer_group_call_result_57792 = invoke(stypy.reporting.localization.Localization(__file__, 706, 29), newer_group_57787, *[depends_57788, target_57789, str_57790], **kwargs_57791)
        
        # Applying the binary operator 'or' (line 706)
        result_or_keyword_57793 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 15), 'or', force_57786, newer_group_call_result_57792)
        
        # Testing the type of an if condition (line 706)
        if_condition_57794 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 706, 12), result_or_keyword_57793)
        # Assigning a type to the variable 'if_condition_57794' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 12), 'if_condition_57794', if_condition_57794)
        # SSA begins for if statement (line 706)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 707)
        # Processing the call arguments (line 707)
        str_57797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 25), 'str', '%s: %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 707)
        tuple_57798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 707)
        # Adding element type (line 707)
        
        # Call to basename(...): (line 707)
        # Processing the call arguments (line 707)
        # Getting the type of 'swig' (line 707)
        swig_57802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 54), 'swig', False)
        # Processing the call keyword arguments (line 707)
        kwargs_57803 = {}
        # Getting the type of 'os' (line 707)
        os_57799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 37), 'os', False)
        # Obtaining the member 'path' of a type (line 707)
        path_57800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 37), os_57799, 'path')
        # Obtaining the member 'basename' of a type (line 707)
        basename_57801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 37), path_57800, 'basename')
        # Calling basename(args, kwargs) (line 707)
        basename_call_result_57804 = invoke(stypy.reporting.localization.Localization(__file__, 707, 37), basename_57801, *[swig_57802], **kwargs_57803)
        
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        # Getting the type of 'is_cpp' (line 708)
        is_cpp_57805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 40), 'is_cpp', False)
        str_57806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 51), 'str', '++')
        # Applying the binary operator 'and' (line 708)
        result_and_keyword_57807 = python_operator(stypy.reporting.localization.Localization(__file__, 708, 40), 'and', is_cpp_57805, str_57806)
        
        str_57808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 59), 'str', '')
        # Applying the binary operator 'or' (line 708)
        result_or_keyword_57809 = python_operator(stypy.reporting.localization.Localization(__file__, 708, 40), 'or', result_and_keyword_57807, str_57808)
        
        # Applying the binary operator '+' (line 707)
        result_add_57810 = python_operator(stypy.reporting.localization.Localization(__file__, 707, 37), '+', basename_call_result_57804, result_or_keyword_57809)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 707, 37), tuple_57798, result_add_57810)
        # Adding element type (line 707)
        # Getting the type of 'source' (line 708)
        source_57811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 64), 'source', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 707, 37), tuple_57798, source_57811)
        
        # Applying the binary operator '%' (line 707)
        result_mod_57812 = python_operator(stypy.reporting.localization.Localization(__file__, 707, 25), '%', str_57797, tuple_57798)
        
        # Processing the call keyword arguments (line 707)
        kwargs_57813 = {}
        # Getting the type of 'log' (line 707)
        log_57795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 16), 'log', False)
        # Obtaining the member 'info' of a type (line 707)
        info_57796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 16), log_57795, 'info')
        # Calling info(args, kwargs) (line 707)
        info_call_result_57814 = invoke(stypy.reporting.localization.Localization(__file__, 707, 16), info_57796, *[result_mod_57812], **kwargs_57813)
        
        
        # Call to spawn(...): (line 709)
        # Processing the call arguments (line 709)
        # Getting the type of 'swig_cmd' (line 709)
        swig_cmd_57817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 27), 'swig_cmd', False)
        # Getting the type of 'self' (line 709)
        self_57818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 38), 'self', False)
        # Obtaining the member 'swig_opts' of a type (line 709)
        swig_opts_57819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 38), self_57818, 'swig_opts')
        # Applying the binary operator '+' (line 709)
        result_add_57820 = python_operator(stypy.reporting.localization.Localization(__file__, 709, 27), '+', swig_cmd_57817, swig_opts_57819)
        
        
        # Obtaining an instance of the builtin type 'list' (line 710)
        list_57821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 710)
        # Adding element type (line 710)
        str_57822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 30), 'str', '-o')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 710, 29), list_57821, str_57822)
        # Adding element type (line 710)
        # Getting the type of 'target' (line 710)
        target_57823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 36), 'target', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 710, 29), list_57821, target_57823)
        # Adding element type (line 710)
        str_57824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 44), 'str', '-outdir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 710, 29), list_57821, str_57824)
        # Adding element type (line 710)
        # Getting the type of 'py_target_dir' (line 710)
        py_target_dir_57825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 55), 'py_target_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 710, 29), list_57821, py_target_dir_57825)
        # Adding element type (line 710)
        # Getting the type of 'source' (line 710)
        source_57826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 70), 'source', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 710, 29), list_57821, source_57826)
        
        # Applying the binary operator '+' (line 710)
        result_add_57827 = python_operator(stypy.reporting.localization.Localization(__file__, 710, 27), '+', result_add_57820, list_57821)
        
        # Processing the call keyword arguments (line 709)
        kwargs_57828 = {}
        # Getting the type of 'self' (line 709)
        self_57815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 16), 'self', False)
        # Obtaining the member 'spawn' of a type (line 709)
        spawn_57816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 16), self_57815, 'spawn')
        # Calling spawn(args, kwargs) (line 709)
        spawn_call_result_57829 = invoke(stypy.reporting.localization.Localization(__file__, 709, 16), spawn_57816, *[result_add_57827], **kwargs_57828)
        
        # SSA branch for the else part of an if statement (line 706)
        module_type_store.open_ssa_branch('else')
        
        # Call to debug(...): (line 712)
        # Processing the call arguments (line 712)
        str_57832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 26), 'str', "  skipping '%s' swig interface (up-to-date)")
        # Getting the type of 'source' (line 713)
        source_57833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 28), 'source', False)
        # Applying the binary operator '%' (line 712)
        result_mod_57834 = python_operator(stypy.reporting.localization.Localization(__file__, 712, 26), '%', str_57832, source_57833)
        
        # Processing the call keyword arguments (line 712)
        kwargs_57835 = {}
        # Getting the type of 'log' (line 712)
        log_57830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 16), 'log', False)
        # Obtaining the member 'debug' of a type (line 712)
        debug_57831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 16), log_57830, 'debug')
        # Calling debug(args, kwargs) (line 712)
        debug_call_result_57836 = invoke(stypy.reporting.localization.Localization(__file__, 712, 16), debug_57831, *[result_mod_57834], **kwargs_57835)
        
        # SSA join for if statement (line 706)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'new_sources' (line 715)
        new_sources_57837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 15), 'new_sources')
        # Getting the type of 'py_files' (line 715)
        py_files_57838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 29), 'py_files')
        # Applying the binary operator '+' (line 715)
        result_add_57839 = python_operator(stypy.reporting.localization.Localization(__file__, 715, 15), '+', new_sources_57837, py_files_57838)
        
        # Assigning a type to the variable 'stypy_return_type' (line 715)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 8), 'stypy_return_type', result_add_57839)
        
        # ################# End of 'swig_sources(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'swig_sources' in the type store
        # Getting the type of 'stypy_return_type' (line 603)
        stypy_return_type_57840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_57840)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'swig_sources'
        return stypy_return_type_57840


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 45, 0, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_src.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'build_src' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'build_src', build_src)

# Assigning a Str to a Name (line 47):
str_57841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 18), 'str', 'build sources from SWIG, F2PY files or a function')
# Getting the type of 'build_src'
build_src_57842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_src')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_src_57842, 'description', str_57841)

# Assigning a List to a Name (line 49):

# Obtaining an instance of the builtin type 'list' (line 49)
list_57843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 49)
# Adding element type (line 49)

# Obtaining an instance of the builtin type 'tuple' (line 50)
tuple_57844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 50)
# Adding element type (line 50)
str_57845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 9), 'str', 'build-src=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 9), tuple_57844, str_57845)
# Adding element type (line 50)
str_57846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 23), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 9), tuple_57844, str_57846)
# Adding element type (line 50)
str_57847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 28), 'str', 'directory to "build" sources to')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 9), tuple_57844, str_57847)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 19), list_57843, tuple_57844)
# Adding element type (line 49)

# Obtaining an instance of the builtin type 'tuple' (line 51)
tuple_57848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 51)
# Adding element type (line 51)
str_57849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 9), 'str', 'f2py-opts=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 9), tuple_57848, str_57849)
# Adding element type (line 51)
# Getting the type of 'None' (line 51)
None_57850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 23), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 9), tuple_57848, None_57850)
# Adding element type (line 51)
str_57851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 29), 'str', 'list of f2py command line options')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 9), tuple_57848, str_57851)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 19), list_57843, tuple_57848)
# Adding element type (line 49)

# Obtaining an instance of the builtin type 'tuple' (line 52)
tuple_57852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 52)
# Adding element type (line 52)
str_57853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 9), 'str', 'swig=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 9), tuple_57852, str_57853)
# Adding element type (line 52)
# Getting the type of 'None' (line 52)
None_57854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 18), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 9), tuple_57852, None_57854)
# Adding element type (line 52)
str_57855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 24), 'str', 'path to the SWIG executable')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 9), tuple_57852, str_57855)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 19), list_57843, tuple_57852)
# Adding element type (line 49)

# Obtaining an instance of the builtin type 'tuple' (line 53)
tuple_57856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 53)
# Adding element type (line 53)
str_57857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 9), 'str', 'swig-opts=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 9), tuple_57856, str_57857)
# Adding element type (line 53)
# Getting the type of 'None' (line 53)
None_57858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 23), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 9), tuple_57856, None_57858)
# Adding element type (line 53)
str_57859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 29), 'str', 'list of SWIG command line options')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 9), tuple_57856, str_57859)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 19), list_57843, tuple_57856)
# Adding element type (line 49)

# Obtaining an instance of the builtin type 'tuple' (line 54)
tuple_57860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 54)
# Adding element type (line 54)
str_57861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 9), 'str', 'swig-cpp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 9), tuple_57860, str_57861)
# Adding element type (line 54)
# Getting the type of 'None' (line 54)
None_57862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 21), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 9), tuple_57860, None_57862)
# Adding element type (line 54)
str_57863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 27), 'str', 'make SWIG create C++ files (default is autodetected from sources)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 9), tuple_57860, str_57863)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 19), list_57843, tuple_57860)
# Adding element type (line 49)

# Obtaining an instance of the builtin type 'tuple' (line 55)
tuple_57864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 55)
# Adding element type (line 55)
str_57865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 9), 'str', 'f2pyflags=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 9), tuple_57864, str_57865)
# Adding element type (line 55)
# Getting the type of 'None' (line 55)
None_57866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 23), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 9), tuple_57864, None_57866)
# Adding element type (line 55)
str_57867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 29), 'str', 'additional flags to f2py (use --f2py-opts= instead)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 9), tuple_57864, str_57867)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 19), list_57843, tuple_57864)
# Adding element type (line 49)

# Obtaining an instance of the builtin type 'tuple' (line 56)
tuple_57868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 56)
# Adding element type (line 56)
str_57869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 9), 'str', 'swigflags=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 9), tuple_57868, str_57869)
# Adding element type (line 56)
# Getting the type of 'None' (line 56)
None_57870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 23), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 9), tuple_57868, None_57870)
# Adding element type (line 56)
str_57871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 29), 'str', 'additional flags to swig (use --swig-opts= instead)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 9), tuple_57868, str_57871)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 19), list_57843, tuple_57868)
# Adding element type (line 49)

# Obtaining an instance of the builtin type 'tuple' (line 57)
tuple_57872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 57)
# Adding element type (line 57)
str_57873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 9), 'str', 'force')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 9), tuple_57872, str_57873)
# Adding element type (line 57)
str_57874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 18), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 9), tuple_57872, str_57874)
# Adding element type (line 57)
str_57875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 23), 'str', 'forcibly build everything (ignore file timestamps)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 9), tuple_57872, str_57875)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 19), list_57843, tuple_57872)
# Adding element type (line 49)

# Obtaining an instance of the builtin type 'tuple' (line 58)
tuple_57876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 58)
# Adding element type (line 58)
str_57877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 9), 'str', 'inplace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 9), tuple_57876, str_57877)
# Adding element type (line 58)
str_57878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 20), 'str', 'i')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 9), tuple_57876, str_57878)
# Adding element type (line 58)
str_57879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 9), 'str', 'ignore build-lib and put compiled extensions into the source ')
str_57880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 9), 'str', 'directory alongside your pure Python modules')
# Applying the binary operator '+' (line 59)
result_add_57881 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 9), '+', str_57879, str_57880)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 9), tuple_57876, result_add_57881)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 19), list_57843, tuple_57876)

# Getting the type of 'build_src'
build_src_57882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_src')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_src_57882, 'user_options', list_57843)

# Assigning a List to a Name (line 63):

# Obtaining an instance of the builtin type 'list' (line 63)
list_57883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 63)
# Adding element type (line 63)
str_57884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 23), 'str', 'force')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 22), list_57883, str_57884)
# Adding element type (line 63)
str_57885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 32), 'str', 'inplace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 22), list_57883, str_57885)

# Getting the type of 'build_src'
build_src_57886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_src')
# Setting the type of the member 'boolean_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_src_57886, 'boolean_options', list_57883)

# Assigning a List to a Name (line 65):

# Obtaining an instance of the builtin type 'list' (line 65)
list_57887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 65)

# Getting the type of 'build_src'
build_src_57888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_src')
# Setting the type of the member 'help_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_src_57888, 'help_options', list_57887)

# Assigning a Attribute to a Name (line 717):

# Assigning a Attribute to a Name (line 717):

# Call to compile(...): (line 717)
# Processing the call arguments (line 717)
str_57891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 717, 30), 'str', '.*[.](f90|f95|f77|for|ftn|f|pyf)\\Z')
# Getting the type of 're' (line 717)
re_57892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 69), 're', False)
# Obtaining the member 'I' of a type (line 717)
I_57893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 69), re_57892, 'I')
# Processing the call keyword arguments (line 717)
kwargs_57894 = {}
# Getting the type of 're' (line 717)
re_57889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 19), 're', False)
# Obtaining the member 'compile' of a type (line 717)
compile_57890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 19), re_57889, 'compile')
# Calling compile(args, kwargs) (line 717)
compile_call_result_57895 = invoke(stypy.reporting.localization.Localization(__file__, 717, 19), compile_57890, *[str_57891, I_57893], **kwargs_57894)

# Obtaining the member 'match' of a type (line 717)
match_57896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 19), compile_call_result_57895, 'match')
# Assigning a type to the variable '_f_pyf_ext_match' (line 717)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 0), '_f_pyf_ext_match', match_57896)

# Assigning a Attribute to a Name (line 718):

# Assigning a Attribute to a Name (line 718):

# Call to compile(...): (line 718)
# Processing the call arguments (line 718)
str_57899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 31), 'str', '.*[.](inc|h|hpp)\\Z')
# Getting the type of 're' (line 718)
re_57900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 54), 're', False)
# Obtaining the member 'I' of a type (line 718)
I_57901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 54), re_57900, 'I')
# Processing the call keyword arguments (line 718)
kwargs_57902 = {}
# Getting the type of 're' (line 718)
re_57897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 20), 're', False)
# Obtaining the member 'compile' of a type (line 718)
compile_57898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 20), re_57897, 'compile')
# Calling compile(args, kwargs) (line 718)
compile_call_result_57903 = invoke(stypy.reporting.localization.Localization(__file__, 718, 20), compile_57898, *[str_57899, I_57901], **kwargs_57902)

# Obtaining the member 'match' of a type (line 718)
match_57904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 20), compile_call_result_57903, 'match')
# Assigning a type to the variable '_header_ext_match' (line 718)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 0), '_header_ext_match', match_57904)

# Assigning a Attribute to a Name (line 721):

# Assigning a Attribute to a Name (line 721):

# Call to compile(...): (line 721)
# Processing the call arguments (line 721)
str_57907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, 37), 'str', '\\s*%module\\s*(.*\\(\\s*package\\s*=\\s*"(?P<package>[\\w_]+)".*\\)|)\\s*(?P<name>[\\w_]+)')
# Getting the type of 're' (line 722)
re_57908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 37), 're', False)
# Obtaining the member 'I' of a type (line 722)
I_57909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 37), re_57908, 'I')
# Processing the call keyword arguments (line 721)
kwargs_57910 = {}
# Getting the type of 're' (line 721)
re_57905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 26), 're', False)
# Obtaining the member 'compile' of a type (line 721)
compile_57906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 721, 26), re_57905, 'compile')
# Calling compile(args, kwargs) (line 721)
compile_call_result_57911 = invoke(stypy.reporting.localization.Localization(__file__, 721, 26), compile_57906, *[str_57907, I_57909], **kwargs_57910)

# Obtaining the member 'match' of a type (line 721)
match_57912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 721, 26), compile_call_result_57911, 'match')
# Assigning a type to the variable '_swig_module_name_match' (line 721)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 721, 0), '_swig_module_name_match', match_57912)

# Assigning a Attribute to a Name (line 723):

# Assigning a Attribute to a Name (line 723):

# Call to compile(...): (line 723)
# Processing the call arguments (line 723)
str_57915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 27), 'str', '-[*]-\\s*c\\s*-[*]-')
# Getting the type of 're' (line 723)
re_57916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 49), 're', False)
# Obtaining the member 'I' of a type (line 723)
I_57917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 49), re_57916, 'I')
# Processing the call keyword arguments (line 723)
kwargs_57918 = {}
# Getting the type of 're' (line 723)
re_57913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 16), 're', False)
# Obtaining the member 'compile' of a type (line 723)
compile_57914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 16), re_57913, 'compile')
# Calling compile(args, kwargs) (line 723)
compile_call_result_57919 = invoke(stypy.reporting.localization.Localization(__file__, 723, 16), compile_57914, *[str_57915, I_57917], **kwargs_57918)

# Obtaining the member 'search' of a type (line 723)
search_57920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 16), compile_call_result_57919, 'search')
# Assigning a type to the variable '_has_c_header' (line 723)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 0), '_has_c_header', search_57920)

# Assigning a Attribute to a Name (line 724):

# Assigning a Attribute to a Name (line 724):

# Call to compile(...): (line 724)
# Processing the call arguments (line 724)
str_57923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 29), 'str', '-[*]-\\s*c[+][+]\\s*-[*]-')
# Getting the type of 're' (line 724)
re_57924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 57), 're', False)
# Obtaining the member 'I' of a type (line 724)
I_57925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 724, 57), re_57924, 'I')
# Processing the call keyword arguments (line 724)
kwargs_57926 = {}
# Getting the type of 're' (line 724)
re_57921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 18), 're', False)
# Obtaining the member 'compile' of a type (line 724)
compile_57922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 724, 18), re_57921, 'compile')
# Calling compile(args, kwargs) (line 724)
compile_call_result_57927 = invoke(stypy.reporting.localization.Localization(__file__, 724, 18), compile_57922, *[str_57923, I_57925], **kwargs_57926)

# Obtaining the member 'search' of a type (line 724)
search_57928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 724, 18), compile_call_result_57927, 'search')
# Assigning a type to the variable '_has_cpp_header' (line 724)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 0), '_has_cpp_header', search_57928)

@norecursion
def get_swig_target(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_swig_target'
    module_type_store = module_type_store.open_function_context('get_swig_target', 726, 0, False)
    
    # Passed parameters checking function
    get_swig_target.stypy_localization = localization
    get_swig_target.stypy_type_of_self = None
    get_swig_target.stypy_type_store = module_type_store
    get_swig_target.stypy_function_name = 'get_swig_target'
    get_swig_target.stypy_param_names_list = ['source']
    get_swig_target.stypy_varargs_param_name = None
    get_swig_target.stypy_kwargs_param_name = None
    get_swig_target.stypy_call_defaults = defaults
    get_swig_target.stypy_call_varargs = varargs
    get_swig_target.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_swig_target', ['source'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_swig_target', localization, ['source'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_swig_target(...)' code ##################

    
    # Assigning a Call to a Name (line 727):
    
    # Assigning a Call to a Name (line 727):
    
    # Call to open(...): (line 727)
    # Processing the call arguments (line 727)
    # Getting the type of 'source' (line 727)
    source_57930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 13), 'source', False)
    str_57931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 21), 'str', 'r')
    # Processing the call keyword arguments (line 727)
    kwargs_57932 = {}
    # Getting the type of 'open' (line 727)
    open_57929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 8), 'open', False)
    # Calling open(args, kwargs) (line 727)
    open_call_result_57933 = invoke(stypy.reporting.localization.Localization(__file__, 727, 8), open_57929, *[source_57930, str_57931], **kwargs_57932)
    
    # Assigning a type to the variable 'f' (line 727)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 727, 4), 'f', open_call_result_57933)
    
    # Assigning a Name to a Name (line 728):
    
    # Assigning a Name to a Name (line 728):
    # Getting the type of 'None' (line 728)
    None_57934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 13), 'None')
    # Assigning a type to the variable 'result' (line 728)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 4), 'result', None_57934)
    
    # Assigning a Call to a Name (line 729):
    
    # Assigning a Call to a Name (line 729):
    
    # Call to readline(...): (line 729)
    # Processing the call keyword arguments (line 729)
    kwargs_57937 = {}
    # Getting the type of 'f' (line 729)
    f_57935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 11), 'f', False)
    # Obtaining the member 'readline' of a type (line 729)
    readline_57936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 11), f_57935, 'readline')
    # Calling readline(args, kwargs) (line 729)
    readline_call_result_57938 = invoke(stypy.reporting.localization.Localization(__file__, 729, 11), readline_57936, *[], **kwargs_57937)
    
    # Assigning a type to the variable 'line' (line 729)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 4), 'line', readline_call_result_57938)
    
    
    # Call to _has_cpp_header(...): (line 730)
    # Processing the call arguments (line 730)
    # Getting the type of 'line' (line 730)
    line_57940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 23), 'line', False)
    # Processing the call keyword arguments (line 730)
    kwargs_57941 = {}
    # Getting the type of '_has_cpp_header' (line 730)
    _has_cpp_header_57939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 7), '_has_cpp_header', False)
    # Calling _has_cpp_header(args, kwargs) (line 730)
    _has_cpp_header_call_result_57942 = invoke(stypy.reporting.localization.Localization(__file__, 730, 7), _has_cpp_header_57939, *[line_57940], **kwargs_57941)
    
    # Testing the type of an if condition (line 730)
    if_condition_57943 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 730, 4), _has_cpp_header_call_result_57942)
    # Assigning a type to the variable 'if_condition_57943' (line 730)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 4), 'if_condition_57943', if_condition_57943)
    # SSA begins for if statement (line 730)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 731):
    
    # Assigning a Str to a Name (line 731):
    str_57944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 731, 17), 'str', 'c++')
    # Assigning a type to the variable 'result' (line 731)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 8), 'result', str_57944)
    # SSA join for if statement (line 730)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to _has_c_header(...): (line 732)
    # Processing the call arguments (line 732)
    # Getting the type of 'line' (line 732)
    line_57946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 21), 'line', False)
    # Processing the call keyword arguments (line 732)
    kwargs_57947 = {}
    # Getting the type of '_has_c_header' (line 732)
    _has_c_header_57945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 7), '_has_c_header', False)
    # Calling _has_c_header(args, kwargs) (line 732)
    _has_c_header_call_result_57948 = invoke(stypy.reporting.localization.Localization(__file__, 732, 7), _has_c_header_57945, *[line_57946], **kwargs_57947)
    
    # Testing the type of an if condition (line 732)
    if_condition_57949 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 732, 4), _has_c_header_call_result_57948)
    # Assigning a type to the variable 'if_condition_57949' (line 732)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 732, 4), 'if_condition_57949', if_condition_57949)
    # SSA begins for if statement (line 732)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 733):
    
    # Assigning a Str to a Name (line 733):
    str_57950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 17), 'str', 'c')
    # Assigning a type to the variable 'result' (line 733)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 8), 'result', str_57950)
    # SSA join for if statement (line 732)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to close(...): (line 734)
    # Processing the call keyword arguments (line 734)
    kwargs_57953 = {}
    # Getting the type of 'f' (line 734)
    f_57951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 4), 'f', False)
    # Obtaining the member 'close' of a type (line 734)
    close_57952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 4), f_57951, 'close')
    # Calling close(args, kwargs) (line 734)
    close_call_result_57954 = invoke(stypy.reporting.localization.Localization(__file__, 734, 4), close_57952, *[], **kwargs_57953)
    
    # Getting the type of 'result' (line 735)
    result_57955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 735)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 4), 'stypy_return_type', result_57955)
    
    # ################# End of 'get_swig_target(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_swig_target' in the type store
    # Getting the type of 'stypy_return_type' (line 726)
    stypy_return_type_57956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_57956)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_swig_target'
    return stypy_return_type_57956

# Assigning a type to the variable 'get_swig_target' (line 726)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 0), 'get_swig_target', get_swig_target)

@norecursion
def get_swig_modulename(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_swig_modulename'
    module_type_store = module_type_store.open_function_context('get_swig_modulename', 737, 0, False)
    
    # Passed parameters checking function
    get_swig_modulename.stypy_localization = localization
    get_swig_modulename.stypy_type_of_self = None
    get_swig_modulename.stypy_type_store = module_type_store
    get_swig_modulename.stypy_function_name = 'get_swig_modulename'
    get_swig_modulename.stypy_param_names_list = ['source']
    get_swig_modulename.stypy_varargs_param_name = None
    get_swig_modulename.stypy_kwargs_param_name = None
    get_swig_modulename.stypy_call_defaults = defaults
    get_swig_modulename.stypy_call_varargs = varargs
    get_swig_modulename.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_swig_modulename', ['source'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_swig_modulename', localization, ['source'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_swig_modulename(...)' code ##################

    
    # Assigning a Call to a Name (line 738):
    
    # Assigning a Call to a Name (line 738):
    
    # Call to open(...): (line 738)
    # Processing the call arguments (line 738)
    # Getting the type of 'source' (line 738)
    source_57958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 13), 'source', False)
    str_57959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 21), 'str', 'r')
    # Processing the call keyword arguments (line 738)
    kwargs_57960 = {}
    # Getting the type of 'open' (line 738)
    open_57957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 8), 'open', False)
    # Calling open(args, kwargs) (line 738)
    open_call_result_57961 = invoke(stypy.reporting.localization.Localization(__file__, 738, 8), open_57957, *[source_57958, str_57959], **kwargs_57960)
    
    # Assigning a type to the variable 'f' (line 738)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 4), 'f', open_call_result_57961)
    
    # Assigning a Name to a Name (line 739):
    
    # Assigning a Name to a Name (line 739):
    # Getting the type of 'None' (line 739)
    None_57962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 11), 'None')
    # Assigning a type to the variable 'name' (line 739)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 4), 'name', None_57962)
    
    # Getting the type of 'f' (line 740)
    f_57963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 16), 'f')
    # Testing the type of a for loop iterable (line 740)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 740, 4), f_57963)
    # Getting the type of the for loop variable (line 740)
    for_loop_var_57964 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 740, 4), f_57963)
    # Assigning a type to the variable 'line' (line 740)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 740, 4), 'line', for_loop_var_57964)
    # SSA begins for a for statement (line 740)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 741):
    
    # Assigning a Call to a Name (line 741):
    
    # Call to _swig_module_name_match(...): (line 741)
    # Processing the call arguments (line 741)
    # Getting the type of 'line' (line 741)
    line_57966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 36), 'line', False)
    # Processing the call keyword arguments (line 741)
    kwargs_57967 = {}
    # Getting the type of '_swig_module_name_match' (line 741)
    _swig_module_name_match_57965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 12), '_swig_module_name_match', False)
    # Calling _swig_module_name_match(args, kwargs) (line 741)
    _swig_module_name_match_call_result_57968 = invoke(stypy.reporting.localization.Localization(__file__, 741, 12), _swig_module_name_match_57965, *[line_57966], **kwargs_57967)
    
    # Assigning a type to the variable 'm' (line 741)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 8), 'm', _swig_module_name_match_call_result_57968)
    
    # Getting the type of 'm' (line 742)
    m_57969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 11), 'm')
    # Testing the type of an if condition (line 742)
    if_condition_57970 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 742, 8), m_57969)
    # Assigning a type to the variable 'if_condition_57970' (line 742)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'if_condition_57970', if_condition_57970)
    # SSA begins for if statement (line 742)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 743):
    
    # Assigning a Call to a Name (line 743):
    
    # Call to group(...): (line 743)
    # Processing the call arguments (line 743)
    str_57973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 27), 'str', 'name')
    # Processing the call keyword arguments (line 743)
    kwargs_57974 = {}
    # Getting the type of 'm' (line 743)
    m_57971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 19), 'm', False)
    # Obtaining the member 'group' of a type (line 743)
    group_57972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 743, 19), m_57971, 'group')
    # Calling group(args, kwargs) (line 743)
    group_call_result_57975 = invoke(stypy.reporting.localization.Localization(__file__, 743, 19), group_57972, *[str_57973], **kwargs_57974)
    
    # Assigning a type to the variable 'name' (line 743)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 743, 12), 'name', group_call_result_57975)
    # SSA join for if statement (line 742)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to close(...): (line 745)
    # Processing the call keyword arguments (line 745)
    kwargs_57978 = {}
    # Getting the type of 'f' (line 745)
    f_57976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 4), 'f', False)
    # Obtaining the member 'close' of a type (line 745)
    close_57977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 745, 4), f_57976, 'close')
    # Calling close(args, kwargs) (line 745)
    close_call_result_57979 = invoke(stypy.reporting.localization.Localization(__file__, 745, 4), close_57977, *[], **kwargs_57978)
    
    # Getting the type of 'name' (line 746)
    name_57980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 11), 'name')
    # Assigning a type to the variable 'stypy_return_type' (line 746)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 746, 4), 'stypy_return_type', name_57980)
    
    # ################# End of 'get_swig_modulename(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_swig_modulename' in the type store
    # Getting the type of 'stypy_return_type' (line 737)
    stypy_return_type_57981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_57981)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_swig_modulename'
    return stypy_return_type_57981

# Assigning a type to the variable 'get_swig_modulename' (line 737)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 737, 0), 'get_swig_modulename', get_swig_modulename)

@norecursion
def _find_swig_target(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_find_swig_target'
    module_type_store = module_type_store.open_function_context('_find_swig_target', 748, 0, False)
    
    # Passed parameters checking function
    _find_swig_target.stypy_localization = localization
    _find_swig_target.stypy_type_of_self = None
    _find_swig_target.stypy_type_store = module_type_store
    _find_swig_target.stypy_function_name = '_find_swig_target'
    _find_swig_target.stypy_param_names_list = ['target_dir', 'name']
    _find_swig_target.stypy_varargs_param_name = None
    _find_swig_target.stypy_kwargs_param_name = None
    _find_swig_target.stypy_call_defaults = defaults
    _find_swig_target.stypy_call_varargs = varargs
    _find_swig_target.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_find_swig_target', ['target_dir', 'name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_find_swig_target', localization, ['target_dir', 'name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_find_swig_target(...)' code ##################

    
    
    # Obtaining an instance of the builtin type 'list' (line 749)
    list_57982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 749)
    # Adding element type (line 749)
    str_57983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 16), 'str', '.cpp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 749, 15), list_57982, str_57983)
    # Adding element type (line 749)
    str_57984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 24), 'str', '.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 749, 15), list_57982, str_57984)
    
    # Testing the type of a for loop iterable (line 749)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 749, 4), list_57982)
    # Getting the type of the for loop variable (line 749)
    for_loop_var_57985 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 749, 4), list_57982)
    # Assigning a type to the variable 'ext' (line 749)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 749, 4), 'ext', for_loop_var_57985)
    # SSA begins for a for statement (line 749)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 750):
    
    # Assigning a Call to a Name (line 750):
    
    # Call to join(...): (line 750)
    # Processing the call arguments (line 750)
    # Getting the type of 'target_dir' (line 750)
    target_dir_57989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 30), 'target_dir', False)
    str_57990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 42), 'str', '%s_wrap%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 750)
    tuple_57991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 57), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 750)
    # Adding element type (line 750)
    # Getting the type of 'name' (line 750)
    name_57992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 57), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 57), tuple_57991, name_57992)
    # Adding element type (line 750)
    # Getting the type of 'ext' (line 750)
    ext_57993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 63), 'ext', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 57), tuple_57991, ext_57993)
    
    # Applying the binary operator '%' (line 750)
    result_mod_57994 = python_operator(stypy.reporting.localization.Localization(__file__, 750, 42), '%', str_57990, tuple_57991)
    
    # Processing the call keyword arguments (line 750)
    kwargs_57995 = {}
    # Getting the type of 'os' (line 750)
    os_57986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 17), 'os', False)
    # Obtaining the member 'path' of a type (line 750)
    path_57987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 17), os_57986, 'path')
    # Obtaining the member 'join' of a type (line 750)
    join_57988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 17), path_57987, 'join')
    # Calling join(args, kwargs) (line 750)
    join_call_result_57996 = invoke(stypy.reporting.localization.Localization(__file__, 750, 17), join_57988, *[target_dir_57989, result_mod_57994], **kwargs_57995)
    
    # Assigning a type to the variable 'target' (line 750)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 8), 'target', join_call_result_57996)
    
    
    # Call to isfile(...): (line 751)
    # Processing the call arguments (line 751)
    # Getting the type of 'target' (line 751)
    target_58000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 26), 'target', False)
    # Processing the call keyword arguments (line 751)
    kwargs_58001 = {}
    # Getting the type of 'os' (line 751)
    os_57997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 751)
    path_57998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 11), os_57997, 'path')
    # Obtaining the member 'isfile' of a type (line 751)
    isfile_57999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 11), path_57998, 'isfile')
    # Calling isfile(args, kwargs) (line 751)
    isfile_call_result_58002 = invoke(stypy.reporting.localization.Localization(__file__, 751, 11), isfile_57999, *[target_58000], **kwargs_58001)
    
    # Testing the type of an if condition (line 751)
    if_condition_58003 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 751, 8), isfile_call_result_58002)
    # Assigning a type to the variable 'if_condition_58003' (line 751)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 8), 'if_condition_58003', if_condition_58003)
    # SSA begins for if statement (line 751)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 751)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'target' (line 753)
    target_58004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 11), 'target')
    # Assigning a type to the variable 'stypy_return_type' (line 753)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 753, 4), 'stypy_return_type', target_58004)
    
    # ################# End of '_find_swig_target(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_find_swig_target' in the type store
    # Getting the type of 'stypy_return_type' (line 748)
    stypy_return_type_58005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_58005)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_find_swig_target'
    return stypy_return_type_58005

# Assigning a type to the variable '_find_swig_target' (line 748)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 748, 0), '_find_swig_target', _find_swig_target)

# Assigning a Attribute to a Name (line 757):

# Assigning a Attribute to a Name (line 757):

# Call to compile(...): (line 757)
# Processing the call arguments (line 757)
str_58008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 37), 'str', '\\s*python\\s*module\\s*(?P<name>[\\w_]+)')
# Getting the type of 're' (line 758)
re_58009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 32), 're', False)
# Obtaining the member 'I' of a type (line 758)
I_58010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 32), re_58009, 'I')
# Processing the call keyword arguments (line 757)
kwargs_58011 = {}
# Getting the type of 're' (line 757)
re_58006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 26), 're', False)
# Obtaining the member 'compile' of a type (line 757)
compile_58007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 757, 26), re_58006, 'compile')
# Calling compile(args, kwargs) (line 757)
compile_call_result_58012 = invoke(stypy.reporting.localization.Localization(__file__, 757, 26), compile_58007, *[str_58008, I_58010], **kwargs_58011)

# Obtaining the member 'match' of a type (line 757)
match_58013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 757, 26), compile_call_result_58012, 'match')
# Assigning a type to the variable '_f2py_module_name_match' (line 757)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 757, 0), '_f2py_module_name_match', match_58013)

# Assigning a Attribute to a Name (line 759):

# Assigning a Attribute to a Name (line 759):

# Call to compile(...): (line 759)
# Processing the call arguments (line 759)
str_58016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 42), 'str', '\\s*python\\s*module\\s*(?P<name>[\\w_]*?__user__[\\w_]*)')
# Getting the type of 're' (line 760)
re_58017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 56), 're', False)
# Obtaining the member 'I' of a type (line 760)
I_58018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 56), re_58017, 'I')
# Processing the call keyword arguments (line 759)
kwargs_58019 = {}
# Getting the type of 're' (line 759)
re_58014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 31), 're', False)
# Obtaining the member 'compile' of a type (line 759)
compile_58015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 31), re_58014, 'compile')
# Calling compile(args, kwargs) (line 759)
compile_call_result_58020 = invoke(stypy.reporting.localization.Localization(__file__, 759, 31), compile_58015, *[str_58016, I_58018], **kwargs_58019)

# Obtaining the member 'match' of a type (line 759)
match_58021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 31), compile_call_result_58020, 'match')
# Assigning a type to the variable '_f2py_user_module_name_match' (line 759)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 759, 0), '_f2py_user_module_name_match', match_58021)

@norecursion
def get_f2py_modulename(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_f2py_modulename'
    module_type_store = module_type_store.open_function_context('get_f2py_modulename', 762, 0, False)
    
    # Passed parameters checking function
    get_f2py_modulename.stypy_localization = localization
    get_f2py_modulename.stypy_type_of_self = None
    get_f2py_modulename.stypy_type_store = module_type_store
    get_f2py_modulename.stypy_function_name = 'get_f2py_modulename'
    get_f2py_modulename.stypy_param_names_list = ['source']
    get_f2py_modulename.stypy_varargs_param_name = None
    get_f2py_modulename.stypy_kwargs_param_name = None
    get_f2py_modulename.stypy_call_defaults = defaults
    get_f2py_modulename.stypy_call_varargs = varargs
    get_f2py_modulename.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_f2py_modulename', ['source'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_f2py_modulename', localization, ['source'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_f2py_modulename(...)' code ##################

    
    # Assigning a Name to a Name (line 763):
    
    # Assigning a Name to a Name (line 763):
    # Getting the type of 'None' (line 763)
    None_58022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 11), 'None')
    # Assigning a type to the variable 'name' (line 763)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 4), 'name', None_58022)
    
    # Assigning a Call to a Name (line 764):
    
    # Assigning a Call to a Name (line 764):
    
    # Call to open(...): (line 764)
    # Processing the call arguments (line 764)
    # Getting the type of 'source' (line 764)
    source_58024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 13), 'source', False)
    # Processing the call keyword arguments (line 764)
    kwargs_58025 = {}
    # Getting the type of 'open' (line 764)
    open_58023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 8), 'open', False)
    # Calling open(args, kwargs) (line 764)
    open_call_result_58026 = invoke(stypy.reporting.localization.Localization(__file__, 764, 8), open_58023, *[source_58024], **kwargs_58025)
    
    # Assigning a type to the variable 'f' (line 764)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 4), 'f', open_call_result_58026)
    
    # Getting the type of 'f' (line 765)
    f_58027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 16), 'f')
    # Testing the type of a for loop iterable (line 765)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 765, 4), f_58027)
    # Getting the type of the for loop variable (line 765)
    for_loop_var_58028 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 765, 4), f_58027)
    # Assigning a type to the variable 'line' (line 765)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 4), 'line', for_loop_var_58028)
    # SSA begins for a for statement (line 765)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 766):
    
    # Assigning a Call to a Name (line 766):
    
    # Call to _f2py_module_name_match(...): (line 766)
    # Processing the call arguments (line 766)
    # Getting the type of 'line' (line 766)
    line_58030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 36), 'line', False)
    # Processing the call keyword arguments (line 766)
    kwargs_58031 = {}
    # Getting the type of '_f2py_module_name_match' (line 766)
    _f2py_module_name_match_58029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 12), '_f2py_module_name_match', False)
    # Calling _f2py_module_name_match(args, kwargs) (line 766)
    _f2py_module_name_match_call_result_58032 = invoke(stypy.reporting.localization.Localization(__file__, 766, 12), _f2py_module_name_match_58029, *[line_58030], **kwargs_58031)
    
    # Assigning a type to the variable 'm' (line 766)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 8), 'm', _f2py_module_name_match_call_result_58032)
    
    # Getting the type of 'm' (line 767)
    m_58033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 11), 'm')
    # Testing the type of an if condition (line 767)
    if_condition_58034 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 767, 8), m_58033)
    # Assigning a type to the variable 'if_condition_58034' (line 767)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 8), 'if_condition_58034', if_condition_58034)
    # SSA begins for if statement (line 767)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to _f2py_user_module_name_match(...): (line 768)
    # Processing the call arguments (line 768)
    # Getting the type of 'line' (line 768)
    line_58036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 44), 'line', False)
    # Processing the call keyword arguments (line 768)
    kwargs_58037 = {}
    # Getting the type of '_f2py_user_module_name_match' (line 768)
    _f2py_user_module_name_match_58035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 15), '_f2py_user_module_name_match', False)
    # Calling _f2py_user_module_name_match(args, kwargs) (line 768)
    _f2py_user_module_name_match_call_result_58038 = invoke(stypy.reporting.localization.Localization(__file__, 768, 15), _f2py_user_module_name_match_58035, *[line_58036], **kwargs_58037)
    
    # Testing the type of an if condition (line 768)
    if_condition_58039 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 768, 12), _f2py_user_module_name_match_call_result_58038)
    # Assigning a type to the variable 'if_condition_58039' (line 768)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 768, 12), 'if_condition_58039', if_condition_58039)
    # SSA begins for if statement (line 768)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 768)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 770):
    
    # Assigning a Call to a Name (line 770):
    
    # Call to group(...): (line 770)
    # Processing the call arguments (line 770)
    str_58042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 770, 27), 'str', 'name')
    # Processing the call keyword arguments (line 770)
    kwargs_58043 = {}
    # Getting the type of 'm' (line 770)
    m_58040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 19), 'm', False)
    # Obtaining the member 'group' of a type (line 770)
    group_58041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 770, 19), m_58040, 'group')
    # Calling group(args, kwargs) (line 770)
    group_call_result_58044 = invoke(stypy.reporting.localization.Localization(__file__, 770, 19), group_58041, *[str_58042], **kwargs_58043)
    
    # Assigning a type to the variable 'name' (line 770)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 770, 12), 'name', group_call_result_58044)
    # SSA join for if statement (line 767)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to close(...): (line 772)
    # Processing the call keyword arguments (line 772)
    kwargs_58047 = {}
    # Getting the type of 'f' (line 772)
    f_58045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 4), 'f', False)
    # Obtaining the member 'close' of a type (line 772)
    close_58046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 772, 4), f_58045, 'close')
    # Calling close(args, kwargs) (line 772)
    close_call_result_58048 = invoke(stypy.reporting.localization.Localization(__file__, 772, 4), close_58046, *[], **kwargs_58047)
    
    # Getting the type of 'name' (line 773)
    name_58049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 11), 'name')
    # Assigning a type to the variable 'stypy_return_type' (line 773)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 4), 'stypy_return_type', name_58049)
    
    # ################# End of 'get_f2py_modulename(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_f2py_modulename' in the type store
    # Getting the type of 'stypy_return_type' (line 762)
    stypy_return_type_58050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_58050)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_f2py_modulename'
    return stypy_return_type_58050

# Assigning a type to the variable 'get_f2py_modulename' (line 762)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 0), 'get_f2py_modulename', get_f2py_modulename)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

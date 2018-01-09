
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: '''
3: 
4: f2py2e - Fortran to Python C/API generator. 2nd Edition.
5:          See __usage__ below.
6: 
7: Copyright 1999--2011 Pearu Peterson all rights reserved,
8: Pearu Peterson <pearu@cens.ioc.ee>
9: Permission to use, modify, and distribute this software is given under the
10: terms of the NumPy License.
11: 
12: NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
13: $Date: 2005/05/06 08:31:19 $
14: Pearu Peterson
15: 
16: '''
17: from __future__ import division, absolute_import, print_function
18: 
19: import sys
20: import os
21: import pprint
22: import re
23: 
24: from . import crackfortran
25: from . import rules
26: from . import cb_rules
27: from . import auxfuncs
28: from . import cfuncs
29: from . import f90mod_rules
30: from . import __version__
31: 
32: f2py_version = __version__.version
33: errmess = sys.stderr.write
34: # outmess=sys.stdout.write
35: show = pprint.pprint
36: outmess = auxfuncs.outmess
37: 
38: try:
39:     from numpy import __version__ as numpy_version
40: except ImportError:
41:     numpy_version = 'N/A'
42: 
43: __usage__ = '''\
44: Usage:
45: 
46: 1) To construct extension module sources:
47: 
48:       f2py [<options>] <fortran files> [[[only:]||[skip:]] \\
49:                                         <fortran functions> ] \\
50:                                        [: <fortran files> ...]
51: 
52: 2) To compile fortran files and build extension modules:
53: 
54:       f2py -c [<options>, <build_flib options>, <extra options>] <fortran files>
55: 
56: 3) To generate signature files:
57: 
58:       f2py -h <filename.pyf> ...< same options as in (1) >
59: 
60: Description: This program generates a Python C/API file (<modulename>module.c)
61:              that contains wrappers for given fortran functions so that they
62:              can be called from Python. With the -c option the corresponding
63:              extension modules are built.
64: 
65: Options:
66: 
67:   --2d-numpy       Use numpy.f2py tool with NumPy support. [DEFAULT]
68:   --2d-numeric     Use f2py2e tool with Numeric support.
69:   --2d-numarray    Use f2py2e tool with Numarray support.
70:   --g3-numpy       Use 3rd generation f2py from the separate f2py package.
71:                    [NOT AVAILABLE YET]
72: 
73:   -h <filename>    Write signatures of the fortran routines to file <filename>
74:                    and exit. You can then edit <filename> and use it instead
75:                    of <fortran files>. If <filename>==stdout then the
76:                    signatures are printed to stdout.
77:   <fortran functions>  Names of fortran routines for which Python C/API
78:                    functions will be generated. Default is all that are found
79:                    in <fortran files>.
80:   <fortran files>  Paths to fortran/signature files that will be scanned for
81:                    <fortran functions> in order to determine their signatures.
82:   skip:            Ignore fortran functions that follow until `:'.
83:   only:            Use only fortran functions that follow until `:'.
84:   :                Get back to <fortran files> mode.
85: 
86:   -m <modulename>  Name of the module; f2py generates a Python/C API
87:                    file <modulename>module.c or extension module <modulename>.
88:                    Default is 'untitled'.
89: 
90:   --[no-]lower     Do [not] lower the cases in <fortran files>. By default,
91:                    --lower is assumed with -h key, and --no-lower without -h key.
92: 
93:   --build-dir <dirname>  All f2py generated files are created in <dirname>.
94:                    Default is tempfile.mkdtemp().
95: 
96:   --overwrite-signature  Overwrite existing signature file.
97: 
98:   --[no-]latex-doc Create (or not) <modulename>module.tex.
99:                    Default is --no-latex-doc.
100:   --short-latex    Create 'incomplete' LaTeX document (without commands
101:                    \\documentclass, \\tableofcontents, and \\begin{document},
102:                    \\end{document}).
103: 
104:   --[no-]rest-doc Create (or not) <modulename>module.rst.
105:                    Default is --no-rest-doc.
106: 
107:   --debug-capi     Create C/API code that reports the state of the wrappers
108:                    during runtime. Useful for debugging.
109: 
110:   --[no-]wrap-functions    Create Fortran subroutine wrappers to Fortran 77
111:                    functions. --wrap-functions is default because it ensures
112:                    maximum portability/compiler independence.
113: 
114:   --include-paths <path1>:<path2>:...   Search include files from the given
115:                    directories.
116: 
117:   --help-link [..] List system resources found by system_info.py. See also
118:                    --link-<resource> switch below. [..] is optional list
119:                    of resources names. E.g. try 'f2py --help-link lapack_opt'.
120: 
121:   --quiet          Run quietly.
122:   --verbose        Run with extra verbosity.
123:   -v               Print f2py version ID and exit.
124: 
125: 
126: numpy.distutils options (only effective with -c):
127: 
128:   --fcompiler=         Specify Fortran compiler type by vendor
129:   --compiler=          Specify C compiler type (as defined by distutils)
130: 
131:   --help-fcompiler     List available Fortran compilers and exit
132:   --f77exec=           Specify the path to F77 compiler
133:   --f90exec=           Specify the path to F90 compiler
134:   --f77flags=          Specify F77 compiler flags
135:   --f90flags=          Specify F90 compiler flags
136:   --opt=               Specify optimization flags
137:   --arch=              Specify architecture specific optimization flags
138:   --noopt              Compile without optimization
139:   --noarch             Compile without arch-dependent optimization
140:   --debug              Compile with debugging information
141: 
142: Extra options (only effective with -c):
143: 
144:   --link-<resource>    Link extension module with <resource> as defined
145:                        by numpy.distutils/system_info.py. E.g. to link
146:                        with optimized LAPACK libraries (vecLib on MacOSX,
147:                        ATLAS elsewhere), use --link-lapack_opt.
148:                        See also --help-link switch.
149: 
150:   -L/path/to/lib/ -l<libname>
151:   -D<define> -U<name>
152:   -I/path/to/include/
153:   <filename>.o <filename>.so <filename>.a
154: 
155:   Using the following macros may be required with non-gcc Fortran
156:   compilers:
157:     -DPREPEND_FORTRAN -DNO_APPEND_FORTRAN -DUPPERCASE_FORTRAN
158:     -DUNDERSCORE_G77
159: 
160:   When using -DF2PY_REPORT_ATEXIT, a performance report of F2PY
161:   interface is printed out at exit (platforms: Linux).
162: 
163:   When using -DF2PY_REPORT_ON_ARRAY_COPY=<int>, a message is
164:   sent to stderr whenever F2PY interface makes a copy of an
165:   array. Integer <int> sets the threshold for array sizes when
166:   a message should be shown.
167: 
168: Version:     %s
169: numpy Version: %s
170: Requires:    Python 2.3 or higher.
171: License:     NumPy license (see LICENSE.txt in the NumPy source code)
172: Copyright 1999 - 2011 Pearu Peterson all rights reserved.
173: http://cens.ioc.ee/projects/f2py2e/''' % (f2py_version, numpy_version)
174: 
175: 
176: def scaninputline(inputline):
177:     files, skipfuncs, onlyfuncs, debug = [], [], [], []
178:     f, f2, f3, f5, f6, f7, f8, f9 = 1, 0, 0, 0, 0, 0, 0, 0
179:     verbose = 1
180:     dolc = -1
181:     dolatexdoc = 0
182:     dorestdoc = 0
183:     wrapfuncs = 1
184:     buildpath = '.'
185:     include_paths = []
186:     signsfile, modulename = None, None
187:     options = {'buildpath': buildpath,
188:                'coutput': None,
189:                'f2py_wrapper_output': None}
190:     for l in inputline:
191:         if l == '':
192:             pass
193:         elif l == 'only:':
194:             f = 0
195:         elif l == 'skip:':
196:             f = -1
197:         elif l == ':':
198:             f = 1
199:         elif l[:8] == '--debug-':
200:             debug.append(l[8:])
201:         elif l == '--lower':
202:             dolc = 1
203:         elif l == '--build-dir':
204:             f6 = 1
205:         elif l == '--no-lower':
206:             dolc = 0
207:         elif l == '--quiet':
208:             verbose = 0
209:         elif l == '--verbose':
210:             verbose += 1
211:         elif l == '--latex-doc':
212:             dolatexdoc = 1
213:         elif l == '--no-latex-doc':
214:             dolatexdoc = 0
215:         elif l == '--rest-doc':
216:             dorestdoc = 1
217:         elif l == '--no-rest-doc':
218:             dorestdoc = 0
219:         elif l == '--wrap-functions':
220:             wrapfuncs = 1
221:         elif l == '--no-wrap-functions':
222:             wrapfuncs = 0
223:         elif l == '--short-latex':
224:             options['shortlatex'] = 1
225:         elif l == '--coutput':
226:             f8 = 1
227:         elif l == '--f2py-wrapper-output':
228:             f9 = 1
229:         elif l == '--overwrite-signature':
230:             options['h-overwrite'] = 1
231:         elif l == '-h':
232:             f2 = 1
233:         elif l == '-m':
234:             f3 = 1
235:         elif l[:2] == '-v':
236:             print(f2py_version)
237:             sys.exit()
238:         elif l == '--show-compilers':
239:             f5 = 1
240:         elif l[:8] == '-include':
241:             cfuncs.outneeds['userincludes'].append(l[9:-1])
242:             cfuncs.userincludes[l[9:-1]] = '#include ' + l[8:]
243:         elif l[:15] in '--include_paths':
244:             outmess(
245:                 'f2py option --include_paths is deprecated, use --include-paths instead.\n')
246:             f7 = 1
247:         elif l[:15] in '--include-paths':
248:             f7 = 1
249:         elif l[0] == '-':
250:             errmess('Unknown option %s\n' % repr(l))
251:             sys.exit()
252:         elif f2:
253:             f2 = 0
254:             signsfile = l
255:         elif f3:
256:             f3 = 0
257:             modulename = l
258:         elif f6:
259:             f6 = 0
260:             buildpath = l
261:         elif f7:
262:             f7 = 0
263:             include_paths.extend(l.split(os.pathsep))
264:         elif f8:
265:             f8 = 0
266:             options["coutput"] = l
267:         elif f9:
268:             f9 = 0
269:             options["f2py_wrapper_output"] = l
270:         elif f == 1:
271:             try:
272:                 open(l).close()
273:                 files.append(l)
274:             except IOError as detail:
275:                 errmess('IOError: %s. Skipping file "%s".\n' %
276:                         (str(detail), l))
277:         elif f == -1:
278:             skipfuncs.append(l)
279:         elif f == 0:
280:             onlyfuncs.append(l)
281:     if not f5 and not files and not modulename:
282:         print(__usage__)
283:         sys.exit()
284:     if not os.path.isdir(buildpath):
285:         if not verbose:
286:             outmess('Creating build directory %s' % (buildpath))
287:         os.mkdir(buildpath)
288:     if signsfile:
289:         signsfile = os.path.join(buildpath, signsfile)
290:     if signsfile and os.path.isfile(signsfile) and 'h-overwrite' not in options:
291:         errmess(
292:             'Signature file "%s" exists!!! Use --overwrite-signature to overwrite.\n' % (signsfile))
293:         sys.exit()
294: 
295:     options['debug'] = debug
296:     options['verbose'] = verbose
297:     if dolc == -1 and not signsfile:
298:         options['do-lower'] = 0
299:     else:
300:         options['do-lower'] = dolc
301:     if modulename:
302:         options['module'] = modulename
303:     if signsfile:
304:         options['signsfile'] = signsfile
305:     if onlyfuncs:
306:         options['onlyfuncs'] = onlyfuncs
307:     if skipfuncs:
308:         options['skipfuncs'] = skipfuncs
309:     options['dolatexdoc'] = dolatexdoc
310:     options['dorestdoc'] = dorestdoc
311:     options['wrapfuncs'] = wrapfuncs
312:     options['buildpath'] = buildpath
313:     options['include_paths'] = include_paths
314:     return files, options
315: 
316: 
317: def callcrackfortran(files, options):
318:     rules.options = options
319:     crackfortran.debug = options['debug']
320:     crackfortran.verbose = options['verbose']
321:     if 'module' in options:
322:         crackfortran.f77modulename = options['module']
323:     if 'skipfuncs' in options:
324:         crackfortran.skipfuncs = options['skipfuncs']
325:     if 'onlyfuncs' in options:
326:         crackfortran.onlyfuncs = options['onlyfuncs']
327:     crackfortran.include_paths[:] = options['include_paths']
328:     crackfortran.dolowercase = options['do-lower']
329:     postlist = crackfortran.crackfortran(files)
330:     if 'signsfile' in options:
331:         outmess('Saving signatures to file "%s"\n' % (options['signsfile']))
332:         pyf = crackfortran.crack2fortran(postlist)
333:         if options['signsfile'][-6:] == 'stdout':
334:             sys.stdout.write(pyf)
335:         else:
336:             f = open(options['signsfile'], 'w')
337:             f.write(pyf)
338:             f.close()
339:     if options["coutput"] is None:
340:         for mod in postlist:
341:             mod["coutput"] = "%smodule.c" % mod["name"]
342:     else:
343:         for mod in postlist:
344:             mod["coutput"] = options["coutput"]
345:     if options["f2py_wrapper_output"] is None:
346:         for mod in postlist:
347:             mod["f2py_wrapper_output"] = "%s-f2pywrappers.f" % mod["name"]
348:     else:
349:         for mod in postlist:
350:             mod["f2py_wrapper_output"] = options["f2py_wrapper_output"]
351:     return postlist
352: 
353: 
354: def buildmodules(lst):
355:     cfuncs.buildcfuncs()
356:     outmess('Building modules...\n')
357:     modules, mnames, isusedby = [], [], {}
358:     for i in range(len(lst)):
359:         if '__user__' in lst[i]['name']:
360:             cb_rules.buildcallbacks(lst[i])
361:         else:
362:             if 'use' in lst[i]:
363:                 for u in lst[i]['use'].keys():
364:                     if u not in isusedby:
365:                         isusedby[u] = []
366:                     isusedby[u].append(lst[i]['name'])
367:             modules.append(lst[i])
368:             mnames.append(lst[i]['name'])
369:     ret = {}
370:     for i in range(len(mnames)):
371:         if mnames[i] in isusedby:
372:             outmess('\tSkipping module "%s" which is used by %s.\n' % (
373:                 mnames[i], ','.join(['"%s"' % s for s in isusedby[mnames[i]]])))
374:         else:
375:             um = []
376:             if 'use' in modules[i]:
377:                 for u in modules[i]['use'].keys():
378:                     if u in isusedby and u in mnames:
379:                         um.append(modules[mnames.index(u)])
380:                     else:
381:                         outmess(
382:                             '\tModule "%s" uses nonexisting "%s" which will be ignored.\n' % (mnames[i], u))
383:             ret[mnames[i]] = {}
384:             dict_append(ret[mnames[i]], rules.buildmodule(modules[i], um))
385:     return ret
386: 
387: 
388: def dict_append(d_out, d_in):
389:     for (k, v) in d_in.items():
390:         if k not in d_out:
391:             d_out[k] = []
392:         if isinstance(v, list):
393:             d_out[k] = d_out[k] + v
394:         else:
395:             d_out[k].append(v)
396: 
397: 
398: def run_main(comline_list):
399:     '''Run f2py as if string.join(comline_list,' ') is used as a command line.
400:     In case of using -h flag, return None.
401:     '''
402:     crackfortran.reset_global_f2py_vars()
403:     f2pydir = os.path.dirname(os.path.abspath(cfuncs.__file__))
404:     fobjhsrc = os.path.join(f2pydir, 'src', 'fortranobject.h')
405:     fobjcsrc = os.path.join(f2pydir, 'src', 'fortranobject.c')
406:     files, options = scaninputline(comline_list)
407:     auxfuncs.options = options
408:     postlist = callcrackfortran(files, options)
409:     isusedby = {}
410:     for i in range(len(postlist)):
411:         if 'use' in postlist[i]:
412:             for u in postlist[i]['use'].keys():
413:                 if u not in isusedby:
414:                     isusedby[u] = []
415:                 isusedby[u].append(postlist[i]['name'])
416:     for i in range(len(postlist)):
417:         if postlist[i]['block'] == 'python module' and '__user__' in postlist[i]['name']:
418:             if postlist[i]['name'] in isusedby:
419:                 # if not quiet:
420:                 outmess('Skipping Makefile build for module "%s" which is used by %s\n' % (
421:                     postlist[i]['name'], ','.join(['"%s"' % s for s in isusedby[postlist[i]['name']]])))
422:     if 'signsfile' in options:
423:         if options['verbose'] > 1:
424:             outmess(
425:                 'Stopping. Edit the signature file and then run f2py on the signature file: ')
426:             outmess('%s %s\n' %
427:                     (os.path.basename(sys.argv[0]), options['signsfile']))
428:         return
429:     for i in range(len(postlist)):
430:         if postlist[i]['block'] != 'python module':
431:             if 'python module' not in options:
432:                 errmess(
433:                     'Tip: If your original code is Fortran source then you must use -m option.\n')
434:             raise TypeError('All blocks must be python module blocks but got %s' % (
435:                 repr(postlist[i]['block'])))
436:     auxfuncs.debugoptions = options['debug']
437:     f90mod_rules.options = options
438:     auxfuncs.wrapfuncs = options['wrapfuncs']
439: 
440:     ret = buildmodules(postlist)
441: 
442:     for mn in ret.keys():
443:         dict_append(ret[mn], {'csrc': fobjcsrc, 'h': fobjhsrc})
444:     return ret
445: 
446: 
447: def filter_files(prefix, suffix, files, remove_prefix=None):
448:     '''
449:     Filter files by prefix and suffix.
450:     '''
451:     filtered, rest = [], []
452:     match = re.compile(prefix + r'.*' + suffix + r'\Z').match
453:     if remove_prefix:
454:         ind = len(prefix)
455:     else:
456:         ind = 0
457:     for file in [x.strip() for x in files]:
458:         if match(file):
459:             filtered.append(file[ind:])
460:         else:
461:             rest.append(file)
462:     return filtered, rest
463: 
464: 
465: def get_prefix(module):
466:     p = os.path.dirname(os.path.dirname(module.__file__))
467:     return p
468: 
469: 
470: def run_compile():
471:     '''
472:     Do it all in one call!
473:     '''
474:     import tempfile
475: 
476:     i = sys.argv.index('-c')
477:     del sys.argv[i]
478: 
479:     remove_build_dir = 0
480:     try:
481:         i = sys.argv.index('--build-dir')
482:     except ValueError:
483:         i = None
484:     if i is not None:
485:         build_dir = sys.argv[i + 1]
486:         del sys.argv[i + 1]
487:         del sys.argv[i]
488:     else:
489:         remove_build_dir = 1
490:         build_dir = tempfile.mkdtemp()
491: 
492:     _reg1 = re.compile(r'[-][-]link[-]')
493:     sysinfo_flags = [_m for _m in sys.argv[1:] if _reg1.match(_m)]
494:     sys.argv = [_m for _m in sys.argv if _m not in sysinfo_flags]
495:     if sysinfo_flags:
496:         sysinfo_flags = [f[7:] for f in sysinfo_flags]
497: 
498:     _reg2 = re.compile(
499:         r'[-][-]((no[-]|)(wrap[-]functions|lower)|debug[-]capi|quiet)|[-]include')
500:     f2py_flags = [_m for _m in sys.argv[1:] if _reg2.match(_m)]
501:     sys.argv = [_m for _m in sys.argv if _m not in f2py_flags]
502:     f2py_flags2 = []
503:     fl = 0
504:     for a in sys.argv[1:]:
505:         if a in ['only:', 'skip:']:
506:             fl = 1
507:         elif a == ':':
508:             fl = 0
509:         if fl or a == ':':
510:             f2py_flags2.append(a)
511:     if f2py_flags2 and f2py_flags2[-1] != ':':
512:         f2py_flags2.append(':')
513:     f2py_flags.extend(f2py_flags2)
514: 
515:     sys.argv = [_m for _m in sys.argv if _m not in f2py_flags2]
516:     _reg3 = re.compile(
517:         r'[-][-]((f(90)?compiler([-]exec|)|compiler)=|help[-]compiler)')
518:     flib_flags = [_m for _m in sys.argv[1:] if _reg3.match(_m)]
519:     sys.argv = [_m for _m in sys.argv if _m not in flib_flags]
520:     _reg4 = re.compile(
521:         r'[-][-]((f(77|90)(flags|exec)|opt|arch)=|(debug|noopt|noarch|help[-]fcompiler))')
522:     fc_flags = [_m for _m in sys.argv[1:] if _reg4.match(_m)]
523:     sys.argv = [_m for _m in sys.argv if _m not in fc_flags]
524: 
525:     if 1:
526:         del_list = []
527:         for s in flib_flags:
528:             v = '--fcompiler='
529:             if s[:len(v)] == v:
530:                 from numpy.distutils import fcompiler
531:                 fcompiler.load_all_fcompiler_classes()
532:                 allowed_keys = list(fcompiler.fcompiler_class.keys())
533:                 nv = ov = s[len(v):].lower()
534:                 if ov not in allowed_keys:
535:                     vmap = {}  # XXX
536:                     try:
537:                         nv = vmap[ov]
538:                     except KeyError:
539:                         if ov not in vmap.values():
540:                             print('Unknown vendor: "%s"' % (s[len(v):]))
541:                     nv = ov
542:                 i = flib_flags.index(s)
543:                 flib_flags[i] = '--fcompiler=' + nv
544:                 continue
545:         for s in del_list:
546:             i = flib_flags.index(s)
547:             del flib_flags[i]
548:         assert len(flib_flags) <= 2, repr(flib_flags)
549: 
550:     _reg5 = re.compile(r'[-][-](verbose)')
551:     setup_flags = [_m for _m in sys.argv[1:] if _reg5.match(_m)]
552:     sys.argv = [_m for _m in sys.argv if _m not in setup_flags]
553: 
554:     if '--quiet' in f2py_flags:
555:         setup_flags.append('--quiet')
556: 
557:     modulename = 'untitled'
558:     sources = sys.argv[1:]
559: 
560:     for optname in ['--include_paths', '--include-paths']:
561:         if optname in sys.argv:
562:             i = sys.argv.index(optname)
563:             f2py_flags.extend(sys.argv[i:i + 2])
564:             del sys.argv[i + 1], sys.argv[i]
565:             sources = sys.argv[1:]
566: 
567:     if '-m' in sys.argv:
568:         i = sys.argv.index('-m')
569:         modulename = sys.argv[i + 1]
570:         del sys.argv[i + 1], sys.argv[i]
571:         sources = sys.argv[1:]
572:     else:
573:         from numpy.distutils.command.build_src import get_f2py_modulename
574:         pyf_files, sources = filter_files('', '[.]pyf([.]src|)', sources)
575:         sources = pyf_files + sources
576:         for f in pyf_files:
577:             modulename = get_f2py_modulename(f)
578:             if modulename:
579:                 break
580: 
581:     extra_objects, sources = filter_files('', '[.](o|a|so)', sources)
582:     include_dirs, sources = filter_files('-I', '', sources, remove_prefix=1)
583:     library_dirs, sources = filter_files('-L', '', sources, remove_prefix=1)
584:     libraries, sources = filter_files('-l', '', sources, remove_prefix=1)
585:     undef_macros, sources = filter_files('-U', '', sources, remove_prefix=1)
586:     define_macros, sources = filter_files('-D', '', sources, remove_prefix=1)
587:     for i in range(len(define_macros)):
588:         name_value = define_macros[i].split('=', 1)
589:         if len(name_value) == 1:
590:             name_value.append(None)
591:         if len(name_value) == 2:
592:             define_macros[i] = tuple(name_value)
593:         else:
594:             print('Invalid use of -D:', name_value)
595: 
596:     from numpy.distutils.system_info import get_info
597: 
598:     num_info = {}
599:     if num_info:
600:         include_dirs.extend(num_info.get('include_dirs', []))
601: 
602:     from numpy.distutils.core import setup, Extension
603:     ext_args = {'name': modulename, 'sources': sources,
604:                 'include_dirs': include_dirs,
605:                 'library_dirs': library_dirs,
606:                 'libraries': libraries,
607:                 'define_macros': define_macros,
608:                 'undef_macros': undef_macros,
609:                 'extra_objects': extra_objects,
610:                 'f2py_options': f2py_flags,
611:                 }
612: 
613:     if sysinfo_flags:
614:         from numpy.distutils.misc_util import dict_append
615:         for n in sysinfo_flags:
616:             i = get_info(n)
617:             if not i:
618:                 outmess('No %s resources found in system'
619:                         ' (try `f2py --help-link`)\n' % (repr(n)))
620:             dict_append(ext_args, **i)
621: 
622:     ext = Extension(**ext_args)
623:     sys.argv = [sys.argv[0]] + setup_flags
624:     sys.argv.extend(['build',
625:                      '--build-temp', build_dir,
626:                      '--build-base', build_dir,
627:                      '--build-platlib', '.'])
628:     if fc_flags:
629:         sys.argv.extend(['config_fc'] + fc_flags)
630:     if flib_flags:
631:         sys.argv.extend(['build_ext'] + flib_flags)
632: 
633:     setup(ext_modules=[ext])
634: 
635:     if remove_build_dir and os.path.exists(build_dir):
636:         import shutil
637:         outmess('Removing build directory %s\n' % (build_dir))
638:         shutil.rmtree(build_dir)
639: 
640: 
641: def main():
642:     if '--help-link' in sys.argv[1:]:
643:         sys.argv.remove('--help-link')
644:         from numpy.distutils.system_info import show_all
645:         show_all()
646:         return
647:     if '-c' in sys.argv[1:]:
648:         run_compile()
649:     else:
650:         run_main(sys.argv[1:])
651: 
652: # if __name__ == "__main__":
653: #    main()
654: 
655: 
656: # EOF
657: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_90881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, (-1)), 'str', '\n\nf2py2e - Fortran to Python C/API generator. 2nd Edition.\n         See __usage__ below.\n\nCopyright 1999--2011 Pearu Peterson all rights reserved,\nPearu Peterson <pearu@cens.ioc.ee>\nPermission to use, modify, and distribute this software is given under the\nterms of the NumPy License.\n\nNO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.\n$Date: 2005/05/06 08:31:19 $\nPearu Peterson\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'import sys' statement (line 19)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import os' statement (line 20)
import os

import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'import pprint' statement (line 21)
import pprint

import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'pprint', pprint, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'import re' statement (line 22)
import re

import_module(stypy.reporting.localization.Localization(__file__, 22, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from numpy.f2py import crackfortran' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_90882 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.f2py')

if (type(import_90882) is not StypyTypeError):

    if (import_90882 != 'pyd_module'):
        __import__(import_90882)
        sys_modules_90883 = sys.modules[import_90882]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.f2py', sys_modules_90883.module_type_store, module_type_store, ['crackfortran'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_90883, sys_modules_90883.module_type_store, module_type_store)
    else:
        from numpy.f2py import crackfortran

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.f2py', None, module_type_store, ['crackfortran'], [crackfortran])

else:
    # Assigning a type to the variable 'numpy.f2py' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.f2py', import_90882)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'from numpy.f2py import rules' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_90884 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.f2py')

if (type(import_90884) is not StypyTypeError):

    if (import_90884 != 'pyd_module'):
        __import__(import_90884)
        sys_modules_90885 = sys.modules[import_90884]
        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.f2py', sys_modules_90885.module_type_store, module_type_store, ['rules'])
        nest_module(stypy.reporting.localization.Localization(__file__, 25, 0), __file__, sys_modules_90885, sys_modules_90885.module_type_store, module_type_store)
    else:
        from numpy.f2py import rules

        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.f2py', None, module_type_store, ['rules'], [rules])

else:
    # Assigning a type to the variable 'numpy.f2py' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.f2py', import_90884)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from numpy.f2py import cb_rules' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_90886 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.f2py')

if (type(import_90886) is not StypyTypeError):

    if (import_90886 != 'pyd_module'):
        __import__(import_90886)
        sys_modules_90887 = sys.modules[import_90886]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.f2py', sys_modules_90887.module_type_store, module_type_store, ['cb_rules'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_90887, sys_modules_90887.module_type_store, module_type_store)
    else:
        from numpy.f2py import cb_rules

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.f2py', None, module_type_store, ['cb_rules'], [cb_rules])

else:
    # Assigning a type to the variable 'numpy.f2py' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.f2py', import_90886)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'from numpy.f2py import auxfuncs' statement (line 27)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_90888 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.f2py')

if (type(import_90888) is not StypyTypeError):

    if (import_90888 != 'pyd_module'):
        __import__(import_90888)
        sys_modules_90889 = sys.modules[import_90888]
        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.f2py', sys_modules_90889.module_type_store, module_type_store, ['auxfuncs'])
        nest_module(stypy.reporting.localization.Localization(__file__, 27, 0), __file__, sys_modules_90889, sys_modules_90889.module_type_store, module_type_store)
    else:
        from numpy.f2py import auxfuncs

        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.f2py', None, module_type_store, ['auxfuncs'], [auxfuncs])

else:
    # Assigning a type to the variable 'numpy.f2py' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.f2py', import_90888)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'from numpy.f2py import cfuncs' statement (line 28)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_90890 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'numpy.f2py')

if (type(import_90890) is not StypyTypeError):

    if (import_90890 != 'pyd_module'):
        __import__(import_90890)
        sys_modules_90891 = sys.modules[import_90890]
        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'numpy.f2py', sys_modules_90891.module_type_store, module_type_store, ['cfuncs'])
        nest_module(stypy.reporting.localization.Localization(__file__, 28, 0), __file__, sys_modules_90891, sys_modules_90891.module_type_store, module_type_store)
    else:
        from numpy.f2py import cfuncs

        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'numpy.f2py', None, module_type_store, ['cfuncs'], [cfuncs])

else:
    # Assigning a type to the variable 'numpy.f2py' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'numpy.f2py', import_90890)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 0))

# 'from numpy.f2py import f90mod_rules' statement (line 29)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_90892 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'numpy.f2py')

if (type(import_90892) is not StypyTypeError):

    if (import_90892 != 'pyd_module'):
        __import__(import_90892)
        sys_modules_90893 = sys.modules[import_90892]
        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'numpy.f2py', sys_modules_90893.module_type_store, module_type_store, ['f90mod_rules'])
        nest_module(stypy.reporting.localization.Localization(__file__, 29, 0), __file__, sys_modules_90893, sys_modules_90893.module_type_store, module_type_store)
    else:
        from numpy.f2py import f90mod_rules

        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'numpy.f2py', None, module_type_store, ['f90mod_rules'], [f90mod_rules])

else:
    # Assigning a type to the variable 'numpy.f2py' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'numpy.f2py', import_90892)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 0))

# 'from numpy.f2py import __version__' statement (line 30)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_90894 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'numpy.f2py')

if (type(import_90894) is not StypyTypeError):

    if (import_90894 != 'pyd_module'):
        __import__(import_90894)
        sys_modules_90895 = sys.modules[import_90894]
        import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'numpy.f2py', sys_modules_90895.module_type_store, module_type_store, ['__version__'])
        nest_module(stypy.reporting.localization.Localization(__file__, 30, 0), __file__, sys_modules_90895, sys_modules_90895.module_type_store, module_type_store)
    else:
        from numpy.f2py import __version__

        import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'numpy.f2py', None, module_type_store, ['__version__'], [__version__])

else:
    # Assigning a type to the variable 'numpy.f2py' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'numpy.f2py', import_90894)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')


# Assigning a Attribute to a Name (line 32):

# Assigning a Attribute to a Name (line 32):
# Getting the type of '__version__' (line 32)
version___90896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), '__version__')
# Obtaining the member 'version' of a type (line 32)
version_90897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 15), version___90896, 'version')
# Assigning a type to the variable 'f2py_version' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'f2py_version', version_90897)

# Assigning a Attribute to a Name (line 33):

# Assigning a Attribute to a Name (line 33):
# Getting the type of 'sys' (line 33)
sys_90898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 10), 'sys')
# Obtaining the member 'stderr' of a type (line 33)
stderr_90899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 10), sys_90898, 'stderr')
# Obtaining the member 'write' of a type (line 33)
write_90900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 10), stderr_90899, 'write')
# Assigning a type to the variable 'errmess' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'errmess', write_90900)

# Assigning a Attribute to a Name (line 35):

# Assigning a Attribute to a Name (line 35):
# Getting the type of 'pprint' (line 35)
pprint_90901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 7), 'pprint')
# Obtaining the member 'pprint' of a type (line 35)
pprint_90902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 7), pprint_90901, 'pprint')
# Assigning a type to the variable 'show' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'show', pprint_90902)

# Assigning a Attribute to a Name (line 36):

# Assigning a Attribute to a Name (line 36):
# Getting the type of 'auxfuncs' (line 36)
auxfuncs_90903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 10), 'auxfuncs')
# Obtaining the member 'outmess' of a type (line 36)
outmess_90904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 10), auxfuncs_90903, 'outmess')
# Assigning a type to the variable 'outmess' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'outmess', outmess_90904)


# SSA begins for try-except statement (line 38)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 39, 4))

# 'from numpy import numpy_version' statement (line 39)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_90905 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 39, 4), 'numpy')

if (type(import_90905) is not StypyTypeError):

    if (import_90905 != 'pyd_module'):
        __import__(import_90905)
        sys_modules_90906 = sys.modules[import_90905]
        import_from_module(stypy.reporting.localization.Localization(__file__, 39, 4), 'numpy', sys_modules_90906.module_type_store, module_type_store, ['__version__'])
        nest_module(stypy.reporting.localization.Localization(__file__, 39, 4), __file__, sys_modules_90906, sys_modules_90906.module_type_store, module_type_store)
    else:
        from numpy import __version__ as numpy_version

        import_from_module(stypy.reporting.localization.Localization(__file__, 39, 4), 'numpy', None, module_type_store, ['__version__'], [numpy_version])

else:
    # Assigning a type to the variable 'numpy' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'numpy', import_90905)

# Adding an alias
module_type_store.add_alias('numpy_version', '__version__')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

# SSA branch for the except part of a try statement (line 38)
# SSA branch for the except 'ImportError' branch of a try statement (line 38)
module_type_store.open_ssa_branch('except')

# Assigning a Str to a Name (line 41):

# Assigning a Str to a Name (line 41):
str_90907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 20), 'str', 'N/A')
# Assigning a type to the variable 'numpy_version' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'numpy_version', str_90907)
# SSA join for try-except statement (line 38)
module_type_store = module_type_store.join_ssa_context()


# Assigning a BinOp to a Name (line 43):

# Assigning a BinOp to a Name (line 43):
str_90908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, (-1)), 'str', "Usage:\n\n1) To construct extension module sources:\n\n      f2py [<options>] <fortran files> [[[only:]||[skip:]] \\\n                                        <fortran functions> ] \\\n                                       [: <fortran files> ...]\n\n2) To compile fortran files and build extension modules:\n\n      f2py -c [<options>, <build_flib options>, <extra options>] <fortran files>\n\n3) To generate signature files:\n\n      f2py -h <filename.pyf> ...< same options as in (1) >\n\nDescription: This program generates a Python C/API file (<modulename>module.c)\n             that contains wrappers for given fortran functions so that they\n             can be called from Python. With the -c option the corresponding\n             extension modules are built.\n\nOptions:\n\n  --2d-numpy       Use numpy.f2py tool with NumPy support. [DEFAULT]\n  --2d-numeric     Use f2py2e tool with Numeric support.\n  --2d-numarray    Use f2py2e tool with Numarray support.\n  --g3-numpy       Use 3rd generation f2py from the separate f2py package.\n                   [NOT AVAILABLE YET]\n\n  -h <filename>    Write signatures of the fortran routines to file <filename>\n                   and exit. You can then edit <filename> and use it instead\n                   of <fortran files>. If <filename>==stdout then the\n                   signatures are printed to stdout.\n  <fortran functions>  Names of fortran routines for which Python C/API\n                   functions will be generated. Default is all that are found\n                   in <fortran files>.\n  <fortran files>  Paths to fortran/signature files that will be scanned for\n                   <fortran functions> in order to determine their signatures.\n  skip:            Ignore fortran functions that follow until `:'.\n  only:            Use only fortran functions that follow until `:'.\n  :                Get back to <fortran files> mode.\n\n  -m <modulename>  Name of the module; f2py generates a Python/C API\n                   file <modulename>module.c or extension module <modulename>.\n                   Default is 'untitled'.\n\n  --[no-]lower     Do [not] lower the cases in <fortran files>. By default,\n                   --lower is assumed with -h key, and --no-lower without -h key.\n\n  --build-dir <dirname>  All f2py generated files are created in <dirname>.\n                   Default is tempfile.mkdtemp().\n\n  --overwrite-signature  Overwrite existing signature file.\n\n  --[no-]latex-doc Create (or not) <modulename>module.tex.\n                   Default is --no-latex-doc.\n  --short-latex    Create 'incomplete' LaTeX document (without commands\n                   \\documentclass, \\tableofcontents, and \\begin{document},\n                   \\end{document}).\n\n  --[no-]rest-doc Create (or not) <modulename>module.rst.\n                   Default is --no-rest-doc.\n\n  --debug-capi     Create C/API code that reports the state of the wrappers\n                   during runtime. Useful for debugging.\n\n  --[no-]wrap-functions    Create Fortran subroutine wrappers to Fortran 77\n                   functions. --wrap-functions is default because it ensures\n                   maximum portability/compiler independence.\n\n  --include-paths <path1>:<path2>:...   Search include files from the given\n                   directories.\n\n  --help-link [..] List system resources found by system_info.py. See also\n                   --link-<resource> switch below. [..] is optional list\n                   of resources names. E.g. try 'f2py --help-link lapack_opt'.\n\n  --quiet          Run quietly.\n  --verbose        Run with extra verbosity.\n  -v               Print f2py version ID and exit.\n\n\nnumpy.distutils options (only effective with -c):\n\n  --fcompiler=         Specify Fortran compiler type by vendor\n  --compiler=          Specify C compiler type (as defined by distutils)\n\n  --help-fcompiler     List available Fortran compilers and exit\n  --f77exec=           Specify the path to F77 compiler\n  --f90exec=           Specify the path to F90 compiler\n  --f77flags=          Specify F77 compiler flags\n  --f90flags=          Specify F90 compiler flags\n  --opt=               Specify optimization flags\n  --arch=              Specify architecture specific optimization flags\n  --noopt              Compile without optimization\n  --noarch             Compile without arch-dependent optimization\n  --debug              Compile with debugging information\n\nExtra options (only effective with -c):\n\n  --link-<resource>    Link extension module with <resource> as defined\n                       by numpy.distutils/system_info.py. E.g. to link\n                       with optimized LAPACK libraries (vecLib on MacOSX,\n                       ATLAS elsewhere), use --link-lapack_opt.\n                       See also --help-link switch.\n\n  -L/path/to/lib/ -l<libname>\n  -D<define> -U<name>\n  -I/path/to/include/\n  <filename>.o <filename>.so <filename>.a\n\n  Using the following macros may be required with non-gcc Fortran\n  compilers:\n    -DPREPEND_FORTRAN -DNO_APPEND_FORTRAN -DUPPERCASE_FORTRAN\n    -DUNDERSCORE_G77\n\n  When using -DF2PY_REPORT_ATEXIT, a performance report of F2PY\n  interface is printed out at exit (platforms: Linux).\n\n  When using -DF2PY_REPORT_ON_ARRAY_COPY=<int>, a message is\n  sent to stderr whenever F2PY interface makes a copy of an\n  array. Integer <int> sets the threshold for array sizes when\n  a message should be shown.\n\nVersion:     %s\nnumpy Version: %s\nRequires:    Python 2.3 or higher.\nLicense:     NumPy license (see LICENSE.txt in the NumPy source code)\nCopyright 1999 - 2011 Pearu Peterson all rights reserved.\nhttp://cens.ioc.ee/projects/f2py2e/")

# Obtaining an instance of the builtin type 'tuple' (line 173)
tuple_90909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 42), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 173)
# Adding element type (line 173)
# Getting the type of 'f2py_version' (line 173)
f2py_version_90910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 42), 'f2py_version')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 42), tuple_90909, f2py_version_90910)
# Adding element type (line 173)
# Getting the type of 'numpy_version' (line 173)
numpy_version_90911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 56), 'numpy_version')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 42), tuple_90909, numpy_version_90911)

# Applying the binary operator '%' (line 173)
result_mod_90912 = python_operator(stypy.reporting.localization.Localization(__file__, 173, (-1)), '%', str_90908, tuple_90909)

# Assigning a type to the variable '__usage__' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), '__usage__', result_mod_90912)

@norecursion
def scaninputline(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'scaninputline'
    module_type_store = module_type_store.open_function_context('scaninputline', 176, 0, False)
    
    # Passed parameters checking function
    scaninputline.stypy_localization = localization
    scaninputline.stypy_type_of_self = None
    scaninputline.stypy_type_store = module_type_store
    scaninputline.stypy_function_name = 'scaninputline'
    scaninputline.stypy_param_names_list = ['inputline']
    scaninputline.stypy_varargs_param_name = None
    scaninputline.stypy_kwargs_param_name = None
    scaninputline.stypy_call_defaults = defaults
    scaninputline.stypy_call_varargs = varargs
    scaninputline.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'scaninputline', ['inputline'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'scaninputline', localization, ['inputline'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'scaninputline(...)' code ##################

    
    # Assigning a Tuple to a Tuple (line 177):
    
    # Assigning a List to a Name (line 177):
    
    # Obtaining an instance of the builtin type 'list' (line 177)
    list_90913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 41), 'list')
    # Adding type elements to the builtin type 'list' instance (line 177)
    
    # Assigning a type to the variable 'tuple_assignment_90838' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'tuple_assignment_90838', list_90913)
    
    # Assigning a List to a Name (line 177):
    
    # Obtaining an instance of the builtin type 'list' (line 177)
    list_90914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 45), 'list')
    # Adding type elements to the builtin type 'list' instance (line 177)
    
    # Assigning a type to the variable 'tuple_assignment_90839' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'tuple_assignment_90839', list_90914)
    
    # Assigning a List to a Name (line 177):
    
    # Obtaining an instance of the builtin type 'list' (line 177)
    list_90915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 49), 'list')
    # Adding type elements to the builtin type 'list' instance (line 177)
    
    # Assigning a type to the variable 'tuple_assignment_90840' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'tuple_assignment_90840', list_90915)
    
    # Assigning a List to a Name (line 177):
    
    # Obtaining an instance of the builtin type 'list' (line 177)
    list_90916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 53), 'list')
    # Adding type elements to the builtin type 'list' instance (line 177)
    
    # Assigning a type to the variable 'tuple_assignment_90841' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'tuple_assignment_90841', list_90916)
    
    # Assigning a Name to a Name (line 177):
    # Getting the type of 'tuple_assignment_90838' (line 177)
    tuple_assignment_90838_90917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'tuple_assignment_90838')
    # Assigning a type to the variable 'files' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'files', tuple_assignment_90838_90917)
    
    # Assigning a Name to a Name (line 177):
    # Getting the type of 'tuple_assignment_90839' (line 177)
    tuple_assignment_90839_90918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'tuple_assignment_90839')
    # Assigning a type to the variable 'skipfuncs' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 11), 'skipfuncs', tuple_assignment_90839_90918)
    
    # Assigning a Name to a Name (line 177):
    # Getting the type of 'tuple_assignment_90840' (line 177)
    tuple_assignment_90840_90919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'tuple_assignment_90840')
    # Assigning a type to the variable 'onlyfuncs' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 22), 'onlyfuncs', tuple_assignment_90840_90919)
    
    # Assigning a Name to a Name (line 177):
    # Getting the type of 'tuple_assignment_90841' (line 177)
    tuple_assignment_90841_90920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'tuple_assignment_90841')
    # Assigning a type to the variable 'debug' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 33), 'debug', tuple_assignment_90841_90920)
    
    # Assigning a Tuple to a Tuple (line 178):
    
    # Assigning a Num to a Name (line 178):
    int_90921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 36), 'int')
    # Assigning a type to the variable 'tuple_assignment_90842' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'tuple_assignment_90842', int_90921)
    
    # Assigning a Num to a Name (line 178):
    int_90922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 39), 'int')
    # Assigning a type to the variable 'tuple_assignment_90843' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'tuple_assignment_90843', int_90922)
    
    # Assigning a Num to a Name (line 178):
    int_90923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 42), 'int')
    # Assigning a type to the variable 'tuple_assignment_90844' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'tuple_assignment_90844', int_90923)
    
    # Assigning a Num to a Name (line 178):
    int_90924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 45), 'int')
    # Assigning a type to the variable 'tuple_assignment_90845' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'tuple_assignment_90845', int_90924)
    
    # Assigning a Num to a Name (line 178):
    int_90925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 48), 'int')
    # Assigning a type to the variable 'tuple_assignment_90846' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'tuple_assignment_90846', int_90925)
    
    # Assigning a Num to a Name (line 178):
    int_90926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 51), 'int')
    # Assigning a type to the variable 'tuple_assignment_90847' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'tuple_assignment_90847', int_90926)
    
    # Assigning a Num to a Name (line 178):
    int_90927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 54), 'int')
    # Assigning a type to the variable 'tuple_assignment_90848' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'tuple_assignment_90848', int_90927)
    
    # Assigning a Num to a Name (line 178):
    int_90928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 57), 'int')
    # Assigning a type to the variable 'tuple_assignment_90849' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'tuple_assignment_90849', int_90928)
    
    # Assigning a Name to a Name (line 178):
    # Getting the type of 'tuple_assignment_90842' (line 178)
    tuple_assignment_90842_90929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'tuple_assignment_90842')
    # Assigning a type to the variable 'f' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'f', tuple_assignment_90842_90929)
    
    # Assigning a Name to a Name (line 178):
    # Getting the type of 'tuple_assignment_90843' (line 178)
    tuple_assignment_90843_90930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'tuple_assignment_90843')
    # Assigning a type to the variable 'f2' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 7), 'f2', tuple_assignment_90843_90930)
    
    # Assigning a Name to a Name (line 178):
    # Getting the type of 'tuple_assignment_90844' (line 178)
    tuple_assignment_90844_90931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'tuple_assignment_90844')
    # Assigning a type to the variable 'f3' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 11), 'f3', tuple_assignment_90844_90931)
    
    # Assigning a Name to a Name (line 178):
    # Getting the type of 'tuple_assignment_90845' (line 178)
    tuple_assignment_90845_90932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'tuple_assignment_90845')
    # Assigning a type to the variable 'f5' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 15), 'f5', tuple_assignment_90845_90932)
    
    # Assigning a Name to a Name (line 178):
    # Getting the type of 'tuple_assignment_90846' (line 178)
    tuple_assignment_90846_90933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'tuple_assignment_90846')
    # Assigning a type to the variable 'f6' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 19), 'f6', tuple_assignment_90846_90933)
    
    # Assigning a Name to a Name (line 178):
    # Getting the type of 'tuple_assignment_90847' (line 178)
    tuple_assignment_90847_90934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'tuple_assignment_90847')
    # Assigning a type to the variable 'f7' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 23), 'f7', tuple_assignment_90847_90934)
    
    # Assigning a Name to a Name (line 178):
    # Getting the type of 'tuple_assignment_90848' (line 178)
    tuple_assignment_90848_90935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'tuple_assignment_90848')
    # Assigning a type to the variable 'f8' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 27), 'f8', tuple_assignment_90848_90935)
    
    # Assigning a Name to a Name (line 178):
    # Getting the type of 'tuple_assignment_90849' (line 178)
    tuple_assignment_90849_90936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'tuple_assignment_90849')
    # Assigning a type to the variable 'f9' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 31), 'f9', tuple_assignment_90849_90936)
    
    # Assigning a Num to a Name (line 179):
    
    # Assigning a Num to a Name (line 179):
    int_90937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 14), 'int')
    # Assigning a type to the variable 'verbose' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'verbose', int_90937)
    
    # Assigning a Num to a Name (line 180):
    
    # Assigning a Num to a Name (line 180):
    int_90938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 11), 'int')
    # Assigning a type to the variable 'dolc' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'dolc', int_90938)
    
    # Assigning a Num to a Name (line 181):
    
    # Assigning a Num to a Name (line 181):
    int_90939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 17), 'int')
    # Assigning a type to the variable 'dolatexdoc' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'dolatexdoc', int_90939)
    
    # Assigning a Num to a Name (line 182):
    
    # Assigning a Num to a Name (line 182):
    int_90940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 16), 'int')
    # Assigning a type to the variable 'dorestdoc' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'dorestdoc', int_90940)
    
    # Assigning a Num to a Name (line 183):
    
    # Assigning a Num to a Name (line 183):
    int_90941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 16), 'int')
    # Assigning a type to the variable 'wrapfuncs' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'wrapfuncs', int_90941)
    
    # Assigning a Str to a Name (line 184):
    
    # Assigning a Str to a Name (line 184):
    str_90942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 16), 'str', '.')
    # Assigning a type to the variable 'buildpath' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'buildpath', str_90942)
    
    # Assigning a List to a Name (line 185):
    
    # Assigning a List to a Name (line 185):
    
    # Obtaining an instance of the builtin type 'list' (line 185)
    list_90943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 185)
    
    # Assigning a type to the variable 'include_paths' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'include_paths', list_90943)
    
    # Assigning a Tuple to a Tuple (line 186):
    
    # Assigning a Name to a Name (line 186):
    # Getting the type of 'None' (line 186)
    None_90944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 28), 'None')
    # Assigning a type to the variable 'tuple_assignment_90850' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'tuple_assignment_90850', None_90944)
    
    # Assigning a Name to a Name (line 186):
    # Getting the type of 'None' (line 186)
    None_90945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 34), 'None')
    # Assigning a type to the variable 'tuple_assignment_90851' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'tuple_assignment_90851', None_90945)
    
    # Assigning a Name to a Name (line 186):
    # Getting the type of 'tuple_assignment_90850' (line 186)
    tuple_assignment_90850_90946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'tuple_assignment_90850')
    # Assigning a type to the variable 'signsfile' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'signsfile', tuple_assignment_90850_90946)
    
    # Assigning a Name to a Name (line 186):
    # Getting the type of 'tuple_assignment_90851' (line 186)
    tuple_assignment_90851_90947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'tuple_assignment_90851')
    # Assigning a type to the variable 'modulename' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 15), 'modulename', tuple_assignment_90851_90947)
    
    # Assigning a Dict to a Name (line 187):
    
    # Assigning a Dict to a Name (line 187):
    
    # Obtaining an instance of the builtin type 'dict' (line 187)
    dict_90948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 14), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 187)
    # Adding element type (key, value) (line 187)
    str_90949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 15), 'str', 'buildpath')
    # Getting the type of 'buildpath' (line 187)
    buildpath_90950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 28), 'buildpath')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 14), dict_90948, (str_90949, buildpath_90950))
    # Adding element type (key, value) (line 187)
    str_90951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 15), 'str', 'coutput')
    # Getting the type of 'None' (line 188)
    None_90952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 26), 'None')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 14), dict_90948, (str_90951, None_90952))
    # Adding element type (key, value) (line 187)
    str_90953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 15), 'str', 'f2py_wrapper_output')
    # Getting the type of 'None' (line 189)
    None_90954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 38), 'None')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 14), dict_90948, (str_90953, None_90954))
    
    # Assigning a type to the variable 'options' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'options', dict_90948)
    
    # Getting the type of 'inputline' (line 190)
    inputline_90955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 13), 'inputline')
    # Testing the type of a for loop iterable (line 190)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 190, 4), inputline_90955)
    # Getting the type of the for loop variable (line 190)
    for_loop_var_90956 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 190, 4), inputline_90955)
    # Assigning a type to the variable 'l' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'l', for_loop_var_90956)
    # SSA begins for a for statement (line 190)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'l' (line 191)
    l_90957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 11), 'l')
    str_90958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 16), 'str', '')
    # Applying the binary operator '==' (line 191)
    result_eq_90959 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 11), '==', l_90957, str_90958)
    
    # Testing the type of an if condition (line 191)
    if_condition_90960 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 191, 8), result_eq_90959)
    # Assigning a type to the variable 'if_condition_90960' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'if_condition_90960', if_condition_90960)
    # SSA begins for if statement (line 191)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    pass
    # SSA branch for the else part of an if statement (line 191)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'l' (line 193)
    l_90961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 13), 'l')
    str_90962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 18), 'str', 'only:')
    # Applying the binary operator '==' (line 193)
    result_eq_90963 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 13), '==', l_90961, str_90962)
    
    # Testing the type of an if condition (line 193)
    if_condition_90964 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 193, 13), result_eq_90963)
    # Assigning a type to the variable 'if_condition_90964' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 13), 'if_condition_90964', if_condition_90964)
    # SSA begins for if statement (line 193)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 194):
    
    # Assigning a Num to a Name (line 194):
    int_90965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 16), 'int')
    # Assigning a type to the variable 'f' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'f', int_90965)
    # SSA branch for the else part of an if statement (line 193)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'l' (line 195)
    l_90966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 13), 'l')
    str_90967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 18), 'str', 'skip:')
    # Applying the binary operator '==' (line 195)
    result_eq_90968 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 13), '==', l_90966, str_90967)
    
    # Testing the type of an if condition (line 195)
    if_condition_90969 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 195, 13), result_eq_90968)
    # Assigning a type to the variable 'if_condition_90969' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 13), 'if_condition_90969', if_condition_90969)
    # SSA begins for if statement (line 195)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 196):
    
    # Assigning a Num to a Name (line 196):
    int_90970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 16), 'int')
    # Assigning a type to the variable 'f' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'f', int_90970)
    # SSA branch for the else part of an if statement (line 195)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'l' (line 197)
    l_90971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 13), 'l')
    str_90972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 18), 'str', ':')
    # Applying the binary operator '==' (line 197)
    result_eq_90973 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 13), '==', l_90971, str_90972)
    
    # Testing the type of an if condition (line 197)
    if_condition_90974 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 197, 13), result_eq_90973)
    # Assigning a type to the variable 'if_condition_90974' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 13), 'if_condition_90974', if_condition_90974)
    # SSA begins for if statement (line 197)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 198):
    
    # Assigning a Num to a Name (line 198):
    int_90975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 16), 'int')
    # Assigning a type to the variable 'f' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'f', int_90975)
    # SSA branch for the else part of an if statement (line 197)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_90976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 16), 'int')
    slice_90977 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 199, 13), None, int_90976, None)
    # Getting the type of 'l' (line 199)
    l_90978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 13), 'l')
    # Obtaining the member '__getitem__' of a type (line 199)
    getitem___90979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 13), l_90978, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 199)
    subscript_call_result_90980 = invoke(stypy.reporting.localization.Localization(__file__, 199, 13), getitem___90979, slice_90977)
    
    str_90981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 22), 'str', '--debug-')
    # Applying the binary operator '==' (line 199)
    result_eq_90982 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 13), '==', subscript_call_result_90980, str_90981)
    
    # Testing the type of an if condition (line 199)
    if_condition_90983 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 199, 13), result_eq_90982)
    # Assigning a type to the variable 'if_condition_90983' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 13), 'if_condition_90983', if_condition_90983)
    # SSA begins for if statement (line 199)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 200)
    # Processing the call arguments (line 200)
    
    # Obtaining the type of the subscript
    int_90986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 27), 'int')
    slice_90987 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 200, 25), int_90986, None, None)
    # Getting the type of 'l' (line 200)
    l_90988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 25), 'l', False)
    # Obtaining the member '__getitem__' of a type (line 200)
    getitem___90989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 25), l_90988, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 200)
    subscript_call_result_90990 = invoke(stypy.reporting.localization.Localization(__file__, 200, 25), getitem___90989, slice_90987)
    
    # Processing the call keyword arguments (line 200)
    kwargs_90991 = {}
    # Getting the type of 'debug' (line 200)
    debug_90984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'debug', False)
    # Obtaining the member 'append' of a type (line 200)
    append_90985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 12), debug_90984, 'append')
    # Calling append(args, kwargs) (line 200)
    append_call_result_90992 = invoke(stypy.reporting.localization.Localization(__file__, 200, 12), append_90985, *[subscript_call_result_90990], **kwargs_90991)
    
    # SSA branch for the else part of an if statement (line 199)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'l' (line 201)
    l_90993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 13), 'l')
    str_90994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 18), 'str', '--lower')
    # Applying the binary operator '==' (line 201)
    result_eq_90995 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 13), '==', l_90993, str_90994)
    
    # Testing the type of an if condition (line 201)
    if_condition_90996 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 13), result_eq_90995)
    # Assigning a type to the variable 'if_condition_90996' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 13), 'if_condition_90996', if_condition_90996)
    # SSA begins for if statement (line 201)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 202):
    
    # Assigning a Num to a Name (line 202):
    int_90997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 19), 'int')
    # Assigning a type to the variable 'dolc' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'dolc', int_90997)
    # SSA branch for the else part of an if statement (line 201)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'l' (line 203)
    l_90998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 13), 'l')
    str_90999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 18), 'str', '--build-dir')
    # Applying the binary operator '==' (line 203)
    result_eq_91000 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 13), '==', l_90998, str_90999)
    
    # Testing the type of an if condition (line 203)
    if_condition_91001 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 13), result_eq_91000)
    # Assigning a type to the variable 'if_condition_91001' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 13), 'if_condition_91001', if_condition_91001)
    # SSA begins for if statement (line 203)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 204):
    
    # Assigning a Num to a Name (line 204):
    int_91002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 17), 'int')
    # Assigning a type to the variable 'f6' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'f6', int_91002)
    # SSA branch for the else part of an if statement (line 203)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'l' (line 205)
    l_91003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 13), 'l')
    str_91004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 18), 'str', '--no-lower')
    # Applying the binary operator '==' (line 205)
    result_eq_91005 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 13), '==', l_91003, str_91004)
    
    # Testing the type of an if condition (line 205)
    if_condition_91006 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 13), result_eq_91005)
    # Assigning a type to the variable 'if_condition_91006' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 13), 'if_condition_91006', if_condition_91006)
    # SSA begins for if statement (line 205)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 206):
    
    # Assigning a Num to a Name (line 206):
    int_91007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 19), 'int')
    # Assigning a type to the variable 'dolc' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'dolc', int_91007)
    # SSA branch for the else part of an if statement (line 205)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'l' (line 207)
    l_91008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 13), 'l')
    str_91009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 18), 'str', '--quiet')
    # Applying the binary operator '==' (line 207)
    result_eq_91010 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 13), '==', l_91008, str_91009)
    
    # Testing the type of an if condition (line 207)
    if_condition_91011 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 207, 13), result_eq_91010)
    # Assigning a type to the variable 'if_condition_91011' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 13), 'if_condition_91011', if_condition_91011)
    # SSA begins for if statement (line 207)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 208):
    
    # Assigning a Num to a Name (line 208):
    int_91012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 22), 'int')
    # Assigning a type to the variable 'verbose' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'verbose', int_91012)
    # SSA branch for the else part of an if statement (line 207)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'l' (line 209)
    l_91013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 13), 'l')
    str_91014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 18), 'str', '--verbose')
    # Applying the binary operator '==' (line 209)
    result_eq_91015 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 13), '==', l_91013, str_91014)
    
    # Testing the type of an if condition (line 209)
    if_condition_91016 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 209, 13), result_eq_91015)
    # Assigning a type to the variable 'if_condition_91016' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 13), 'if_condition_91016', if_condition_91016)
    # SSA begins for if statement (line 209)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'verbose' (line 210)
    verbose_91017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'verbose')
    int_91018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 23), 'int')
    # Applying the binary operator '+=' (line 210)
    result_iadd_91019 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 12), '+=', verbose_91017, int_91018)
    # Assigning a type to the variable 'verbose' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'verbose', result_iadd_91019)
    
    # SSA branch for the else part of an if statement (line 209)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'l' (line 211)
    l_91020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 13), 'l')
    str_91021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 18), 'str', '--latex-doc')
    # Applying the binary operator '==' (line 211)
    result_eq_91022 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 13), '==', l_91020, str_91021)
    
    # Testing the type of an if condition (line 211)
    if_condition_91023 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 211, 13), result_eq_91022)
    # Assigning a type to the variable 'if_condition_91023' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 13), 'if_condition_91023', if_condition_91023)
    # SSA begins for if statement (line 211)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 212):
    
    # Assigning a Num to a Name (line 212):
    int_91024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 25), 'int')
    # Assigning a type to the variable 'dolatexdoc' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'dolatexdoc', int_91024)
    # SSA branch for the else part of an if statement (line 211)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'l' (line 213)
    l_91025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 13), 'l')
    str_91026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 18), 'str', '--no-latex-doc')
    # Applying the binary operator '==' (line 213)
    result_eq_91027 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 13), '==', l_91025, str_91026)
    
    # Testing the type of an if condition (line 213)
    if_condition_91028 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 213, 13), result_eq_91027)
    # Assigning a type to the variable 'if_condition_91028' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 13), 'if_condition_91028', if_condition_91028)
    # SSA begins for if statement (line 213)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 214):
    
    # Assigning a Num to a Name (line 214):
    int_91029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 25), 'int')
    # Assigning a type to the variable 'dolatexdoc' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'dolatexdoc', int_91029)
    # SSA branch for the else part of an if statement (line 213)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'l' (line 215)
    l_91030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 13), 'l')
    str_91031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 18), 'str', '--rest-doc')
    # Applying the binary operator '==' (line 215)
    result_eq_91032 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 13), '==', l_91030, str_91031)
    
    # Testing the type of an if condition (line 215)
    if_condition_91033 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 13), result_eq_91032)
    # Assigning a type to the variable 'if_condition_91033' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 13), 'if_condition_91033', if_condition_91033)
    # SSA begins for if statement (line 215)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 216):
    
    # Assigning a Num to a Name (line 216):
    int_91034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 24), 'int')
    # Assigning a type to the variable 'dorestdoc' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'dorestdoc', int_91034)
    # SSA branch for the else part of an if statement (line 215)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'l' (line 217)
    l_91035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 13), 'l')
    str_91036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 18), 'str', '--no-rest-doc')
    # Applying the binary operator '==' (line 217)
    result_eq_91037 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 13), '==', l_91035, str_91036)
    
    # Testing the type of an if condition (line 217)
    if_condition_91038 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 217, 13), result_eq_91037)
    # Assigning a type to the variable 'if_condition_91038' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 13), 'if_condition_91038', if_condition_91038)
    # SSA begins for if statement (line 217)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 218):
    
    # Assigning a Num to a Name (line 218):
    int_91039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 24), 'int')
    # Assigning a type to the variable 'dorestdoc' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'dorestdoc', int_91039)
    # SSA branch for the else part of an if statement (line 217)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'l' (line 219)
    l_91040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 13), 'l')
    str_91041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 18), 'str', '--wrap-functions')
    # Applying the binary operator '==' (line 219)
    result_eq_91042 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 13), '==', l_91040, str_91041)
    
    # Testing the type of an if condition (line 219)
    if_condition_91043 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 219, 13), result_eq_91042)
    # Assigning a type to the variable 'if_condition_91043' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 13), 'if_condition_91043', if_condition_91043)
    # SSA begins for if statement (line 219)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 220):
    
    # Assigning a Num to a Name (line 220):
    int_91044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 24), 'int')
    # Assigning a type to the variable 'wrapfuncs' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'wrapfuncs', int_91044)
    # SSA branch for the else part of an if statement (line 219)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'l' (line 221)
    l_91045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 13), 'l')
    str_91046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 18), 'str', '--no-wrap-functions')
    # Applying the binary operator '==' (line 221)
    result_eq_91047 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 13), '==', l_91045, str_91046)
    
    # Testing the type of an if condition (line 221)
    if_condition_91048 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 13), result_eq_91047)
    # Assigning a type to the variable 'if_condition_91048' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 13), 'if_condition_91048', if_condition_91048)
    # SSA begins for if statement (line 221)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 222):
    
    # Assigning a Num to a Name (line 222):
    int_91049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 24), 'int')
    # Assigning a type to the variable 'wrapfuncs' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'wrapfuncs', int_91049)
    # SSA branch for the else part of an if statement (line 221)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'l' (line 223)
    l_91050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 13), 'l')
    str_91051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 18), 'str', '--short-latex')
    # Applying the binary operator '==' (line 223)
    result_eq_91052 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 13), '==', l_91050, str_91051)
    
    # Testing the type of an if condition (line 223)
    if_condition_91053 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 13), result_eq_91052)
    # Assigning a type to the variable 'if_condition_91053' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 13), 'if_condition_91053', if_condition_91053)
    # SSA begins for if statement (line 223)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Subscript (line 224):
    
    # Assigning a Num to a Subscript (line 224):
    int_91054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 36), 'int')
    # Getting the type of 'options' (line 224)
    options_91055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'options')
    str_91056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 20), 'str', 'shortlatex')
    # Storing an element on a container (line 224)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 12), options_91055, (str_91056, int_91054))
    # SSA branch for the else part of an if statement (line 223)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'l' (line 225)
    l_91057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 13), 'l')
    str_91058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 18), 'str', '--coutput')
    # Applying the binary operator '==' (line 225)
    result_eq_91059 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 13), '==', l_91057, str_91058)
    
    # Testing the type of an if condition (line 225)
    if_condition_91060 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 225, 13), result_eq_91059)
    # Assigning a type to the variable 'if_condition_91060' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 13), 'if_condition_91060', if_condition_91060)
    # SSA begins for if statement (line 225)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 226):
    
    # Assigning a Num to a Name (line 226):
    int_91061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 17), 'int')
    # Assigning a type to the variable 'f8' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'f8', int_91061)
    # SSA branch for the else part of an if statement (line 225)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'l' (line 227)
    l_91062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 13), 'l')
    str_91063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 18), 'str', '--f2py-wrapper-output')
    # Applying the binary operator '==' (line 227)
    result_eq_91064 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 13), '==', l_91062, str_91063)
    
    # Testing the type of an if condition (line 227)
    if_condition_91065 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 13), result_eq_91064)
    # Assigning a type to the variable 'if_condition_91065' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 13), 'if_condition_91065', if_condition_91065)
    # SSA begins for if statement (line 227)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 228):
    
    # Assigning a Num to a Name (line 228):
    int_91066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 17), 'int')
    # Assigning a type to the variable 'f9' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'f9', int_91066)
    # SSA branch for the else part of an if statement (line 227)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'l' (line 229)
    l_91067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 13), 'l')
    str_91068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 18), 'str', '--overwrite-signature')
    # Applying the binary operator '==' (line 229)
    result_eq_91069 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 13), '==', l_91067, str_91068)
    
    # Testing the type of an if condition (line 229)
    if_condition_91070 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 229, 13), result_eq_91069)
    # Assigning a type to the variable 'if_condition_91070' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 13), 'if_condition_91070', if_condition_91070)
    # SSA begins for if statement (line 229)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Subscript (line 230):
    
    # Assigning a Num to a Subscript (line 230):
    int_91071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 37), 'int')
    # Getting the type of 'options' (line 230)
    options_91072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'options')
    str_91073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 20), 'str', 'h-overwrite')
    # Storing an element on a container (line 230)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 12), options_91072, (str_91073, int_91071))
    # SSA branch for the else part of an if statement (line 229)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'l' (line 231)
    l_91074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 13), 'l')
    str_91075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 18), 'str', '-h')
    # Applying the binary operator '==' (line 231)
    result_eq_91076 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 13), '==', l_91074, str_91075)
    
    # Testing the type of an if condition (line 231)
    if_condition_91077 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 231, 13), result_eq_91076)
    # Assigning a type to the variable 'if_condition_91077' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 13), 'if_condition_91077', if_condition_91077)
    # SSA begins for if statement (line 231)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 232):
    
    # Assigning a Num to a Name (line 232):
    int_91078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 17), 'int')
    # Assigning a type to the variable 'f2' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'f2', int_91078)
    # SSA branch for the else part of an if statement (line 231)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'l' (line 233)
    l_91079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 13), 'l')
    str_91080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 18), 'str', '-m')
    # Applying the binary operator '==' (line 233)
    result_eq_91081 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 13), '==', l_91079, str_91080)
    
    # Testing the type of an if condition (line 233)
    if_condition_91082 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 233, 13), result_eq_91081)
    # Assigning a type to the variable 'if_condition_91082' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 13), 'if_condition_91082', if_condition_91082)
    # SSA begins for if statement (line 233)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 234):
    
    # Assigning a Num to a Name (line 234):
    int_91083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 17), 'int')
    # Assigning a type to the variable 'f3' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'f3', int_91083)
    # SSA branch for the else part of an if statement (line 233)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_91084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 16), 'int')
    slice_91085 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 235, 13), None, int_91084, None)
    # Getting the type of 'l' (line 235)
    l_91086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 13), 'l')
    # Obtaining the member '__getitem__' of a type (line 235)
    getitem___91087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 13), l_91086, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 235)
    subscript_call_result_91088 = invoke(stypy.reporting.localization.Localization(__file__, 235, 13), getitem___91087, slice_91085)
    
    str_91089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 22), 'str', '-v')
    # Applying the binary operator '==' (line 235)
    result_eq_91090 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 13), '==', subscript_call_result_91088, str_91089)
    
    # Testing the type of an if condition (line 235)
    if_condition_91091 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 13), result_eq_91090)
    # Assigning a type to the variable 'if_condition_91091' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 13), 'if_condition_91091', if_condition_91091)
    # SSA begins for if statement (line 235)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'f2py_version' (line 236)
    f2py_version_91093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 18), 'f2py_version', False)
    # Processing the call keyword arguments (line 236)
    kwargs_91094 = {}
    # Getting the type of 'print' (line 236)
    print_91092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'print', False)
    # Calling print(args, kwargs) (line 236)
    print_call_result_91095 = invoke(stypy.reporting.localization.Localization(__file__, 236, 12), print_91092, *[f2py_version_91093], **kwargs_91094)
    
    
    # Call to exit(...): (line 237)
    # Processing the call keyword arguments (line 237)
    kwargs_91098 = {}
    # Getting the type of 'sys' (line 237)
    sys_91096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'sys', False)
    # Obtaining the member 'exit' of a type (line 237)
    exit_91097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 12), sys_91096, 'exit')
    # Calling exit(args, kwargs) (line 237)
    exit_call_result_91099 = invoke(stypy.reporting.localization.Localization(__file__, 237, 12), exit_91097, *[], **kwargs_91098)
    
    # SSA branch for the else part of an if statement (line 235)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'l' (line 238)
    l_91100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 13), 'l')
    str_91101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 18), 'str', '--show-compilers')
    # Applying the binary operator '==' (line 238)
    result_eq_91102 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 13), '==', l_91100, str_91101)
    
    # Testing the type of an if condition (line 238)
    if_condition_91103 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 238, 13), result_eq_91102)
    # Assigning a type to the variable 'if_condition_91103' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 13), 'if_condition_91103', if_condition_91103)
    # SSA begins for if statement (line 238)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 239):
    
    # Assigning a Num to a Name (line 239):
    int_91104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 17), 'int')
    # Assigning a type to the variable 'f5' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'f5', int_91104)
    # SSA branch for the else part of an if statement (line 238)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_91105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 16), 'int')
    slice_91106 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 240, 13), None, int_91105, None)
    # Getting the type of 'l' (line 240)
    l_91107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 13), 'l')
    # Obtaining the member '__getitem__' of a type (line 240)
    getitem___91108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 13), l_91107, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 240)
    subscript_call_result_91109 = invoke(stypy.reporting.localization.Localization(__file__, 240, 13), getitem___91108, slice_91106)
    
    str_91110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 22), 'str', '-include')
    # Applying the binary operator '==' (line 240)
    result_eq_91111 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 13), '==', subscript_call_result_91109, str_91110)
    
    # Testing the type of an if condition (line 240)
    if_condition_91112 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 13), result_eq_91111)
    # Assigning a type to the variable 'if_condition_91112' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 13), 'if_condition_91112', if_condition_91112)
    # SSA begins for if statement (line 240)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 241)
    # Processing the call arguments (line 241)
    
    # Obtaining the type of the subscript
    int_91119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 53), 'int')
    int_91120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 55), 'int')
    slice_91121 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 241, 51), int_91119, int_91120, None)
    # Getting the type of 'l' (line 241)
    l_91122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 51), 'l', False)
    # Obtaining the member '__getitem__' of a type (line 241)
    getitem___91123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 51), l_91122, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 241)
    subscript_call_result_91124 = invoke(stypy.reporting.localization.Localization(__file__, 241, 51), getitem___91123, slice_91121)
    
    # Processing the call keyword arguments (line 241)
    kwargs_91125 = {}
    
    # Obtaining the type of the subscript
    str_91113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 28), 'str', 'userincludes')
    # Getting the type of 'cfuncs' (line 241)
    cfuncs_91114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'cfuncs', False)
    # Obtaining the member 'outneeds' of a type (line 241)
    outneeds_91115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 12), cfuncs_91114, 'outneeds')
    # Obtaining the member '__getitem__' of a type (line 241)
    getitem___91116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 12), outneeds_91115, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 241)
    subscript_call_result_91117 = invoke(stypy.reporting.localization.Localization(__file__, 241, 12), getitem___91116, str_91113)
    
    # Obtaining the member 'append' of a type (line 241)
    append_91118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 12), subscript_call_result_91117, 'append')
    # Calling append(args, kwargs) (line 241)
    append_call_result_91126 = invoke(stypy.reporting.localization.Localization(__file__, 241, 12), append_91118, *[subscript_call_result_91124], **kwargs_91125)
    
    
    # Assigning a BinOp to a Subscript (line 242):
    
    # Assigning a BinOp to a Subscript (line 242):
    str_91127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 43), 'str', '#include ')
    
    # Obtaining the type of the subscript
    int_91128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 59), 'int')
    slice_91129 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 242, 57), int_91128, None, None)
    # Getting the type of 'l' (line 242)
    l_91130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 57), 'l')
    # Obtaining the member '__getitem__' of a type (line 242)
    getitem___91131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 57), l_91130, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 242)
    subscript_call_result_91132 = invoke(stypy.reporting.localization.Localization(__file__, 242, 57), getitem___91131, slice_91129)
    
    # Applying the binary operator '+' (line 242)
    result_add_91133 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 43), '+', str_91127, subscript_call_result_91132)
    
    # Getting the type of 'cfuncs' (line 242)
    cfuncs_91134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'cfuncs')
    # Obtaining the member 'userincludes' of a type (line 242)
    userincludes_91135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 12), cfuncs_91134, 'userincludes')
    
    # Obtaining the type of the subscript
    int_91136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 34), 'int')
    int_91137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 36), 'int')
    slice_91138 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 242, 32), int_91136, int_91137, None)
    # Getting the type of 'l' (line 242)
    l_91139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 32), 'l')
    # Obtaining the member '__getitem__' of a type (line 242)
    getitem___91140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 32), l_91139, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 242)
    subscript_call_result_91141 = invoke(stypy.reporting.localization.Localization(__file__, 242, 32), getitem___91140, slice_91138)
    
    # Storing an element on a container (line 242)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 12), userincludes_91135, (subscript_call_result_91141, result_add_91133))
    # SSA branch for the else part of an if statement (line 240)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_91142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 16), 'int')
    slice_91143 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 243, 13), None, int_91142, None)
    # Getting the type of 'l' (line 243)
    l_91144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 13), 'l')
    # Obtaining the member '__getitem__' of a type (line 243)
    getitem___91145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 13), l_91144, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 243)
    subscript_call_result_91146 = invoke(stypy.reporting.localization.Localization(__file__, 243, 13), getitem___91145, slice_91143)
    
    str_91147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 23), 'str', '--include_paths')
    # Applying the binary operator 'in' (line 243)
    result_contains_91148 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 13), 'in', subscript_call_result_91146, str_91147)
    
    # Testing the type of an if condition (line 243)
    if_condition_91149 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 243, 13), result_contains_91148)
    # Assigning a type to the variable 'if_condition_91149' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 13), 'if_condition_91149', if_condition_91149)
    # SSA begins for if statement (line 243)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to outmess(...): (line 244)
    # Processing the call arguments (line 244)
    str_91151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 16), 'str', 'f2py option --include_paths is deprecated, use --include-paths instead.\n')
    # Processing the call keyword arguments (line 244)
    kwargs_91152 = {}
    # Getting the type of 'outmess' (line 244)
    outmess_91150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'outmess', False)
    # Calling outmess(args, kwargs) (line 244)
    outmess_call_result_91153 = invoke(stypy.reporting.localization.Localization(__file__, 244, 12), outmess_91150, *[str_91151], **kwargs_91152)
    
    
    # Assigning a Num to a Name (line 246):
    
    # Assigning a Num to a Name (line 246):
    int_91154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 17), 'int')
    # Assigning a type to the variable 'f7' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'f7', int_91154)
    # SSA branch for the else part of an if statement (line 243)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_91155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 16), 'int')
    slice_91156 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 247, 13), None, int_91155, None)
    # Getting the type of 'l' (line 247)
    l_91157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 13), 'l')
    # Obtaining the member '__getitem__' of a type (line 247)
    getitem___91158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 13), l_91157, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 247)
    subscript_call_result_91159 = invoke(stypy.reporting.localization.Localization(__file__, 247, 13), getitem___91158, slice_91156)
    
    str_91160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 23), 'str', '--include-paths')
    # Applying the binary operator 'in' (line 247)
    result_contains_91161 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 13), 'in', subscript_call_result_91159, str_91160)
    
    # Testing the type of an if condition (line 247)
    if_condition_91162 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 247, 13), result_contains_91161)
    # Assigning a type to the variable 'if_condition_91162' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 13), 'if_condition_91162', if_condition_91162)
    # SSA begins for if statement (line 247)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 248):
    
    # Assigning a Num to a Name (line 248):
    int_91163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 17), 'int')
    # Assigning a type to the variable 'f7' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'f7', int_91163)
    # SSA branch for the else part of an if statement (line 247)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_91164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 15), 'int')
    # Getting the type of 'l' (line 249)
    l_91165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 13), 'l')
    # Obtaining the member '__getitem__' of a type (line 249)
    getitem___91166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 13), l_91165, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 249)
    subscript_call_result_91167 = invoke(stypy.reporting.localization.Localization(__file__, 249, 13), getitem___91166, int_91164)
    
    str_91168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 21), 'str', '-')
    # Applying the binary operator '==' (line 249)
    result_eq_91169 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 13), '==', subscript_call_result_91167, str_91168)
    
    # Testing the type of an if condition (line 249)
    if_condition_91170 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 249, 13), result_eq_91169)
    # Assigning a type to the variable 'if_condition_91170' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 13), 'if_condition_91170', if_condition_91170)
    # SSA begins for if statement (line 249)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to errmess(...): (line 250)
    # Processing the call arguments (line 250)
    str_91172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 20), 'str', 'Unknown option %s\n')
    
    # Call to repr(...): (line 250)
    # Processing the call arguments (line 250)
    # Getting the type of 'l' (line 250)
    l_91174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 49), 'l', False)
    # Processing the call keyword arguments (line 250)
    kwargs_91175 = {}
    # Getting the type of 'repr' (line 250)
    repr_91173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 44), 'repr', False)
    # Calling repr(args, kwargs) (line 250)
    repr_call_result_91176 = invoke(stypy.reporting.localization.Localization(__file__, 250, 44), repr_91173, *[l_91174], **kwargs_91175)
    
    # Applying the binary operator '%' (line 250)
    result_mod_91177 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 20), '%', str_91172, repr_call_result_91176)
    
    # Processing the call keyword arguments (line 250)
    kwargs_91178 = {}
    # Getting the type of 'errmess' (line 250)
    errmess_91171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'errmess', False)
    # Calling errmess(args, kwargs) (line 250)
    errmess_call_result_91179 = invoke(stypy.reporting.localization.Localization(__file__, 250, 12), errmess_91171, *[result_mod_91177], **kwargs_91178)
    
    
    # Call to exit(...): (line 251)
    # Processing the call keyword arguments (line 251)
    kwargs_91182 = {}
    # Getting the type of 'sys' (line 251)
    sys_91180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'sys', False)
    # Obtaining the member 'exit' of a type (line 251)
    exit_91181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 12), sys_91180, 'exit')
    # Calling exit(args, kwargs) (line 251)
    exit_call_result_91183 = invoke(stypy.reporting.localization.Localization(__file__, 251, 12), exit_91181, *[], **kwargs_91182)
    
    # SSA branch for the else part of an if statement (line 249)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'f2' (line 252)
    f2_91184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 13), 'f2')
    # Testing the type of an if condition (line 252)
    if_condition_91185 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 252, 13), f2_91184)
    # Assigning a type to the variable 'if_condition_91185' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 13), 'if_condition_91185', if_condition_91185)
    # SSA begins for if statement (line 252)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 253):
    
    # Assigning a Num to a Name (line 253):
    int_91186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 17), 'int')
    # Assigning a type to the variable 'f2' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'f2', int_91186)
    
    # Assigning a Name to a Name (line 254):
    
    # Assigning a Name to a Name (line 254):
    # Getting the type of 'l' (line 254)
    l_91187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 24), 'l')
    # Assigning a type to the variable 'signsfile' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'signsfile', l_91187)
    # SSA branch for the else part of an if statement (line 252)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'f3' (line 255)
    f3_91188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 13), 'f3')
    # Testing the type of an if condition (line 255)
    if_condition_91189 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 13), f3_91188)
    # Assigning a type to the variable 'if_condition_91189' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 13), 'if_condition_91189', if_condition_91189)
    # SSA begins for if statement (line 255)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 256):
    
    # Assigning a Num to a Name (line 256):
    int_91190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 17), 'int')
    # Assigning a type to the variable 'f3' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'f3', int_91190)
    
    # Assigning a Name to a Name (line 257):
    
    # Assigning a Name to a Name (line 257):
    # Getting the type of 'l' (line 257)
    l_91191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 25), 'l')
    # Assigning a type to the variable 'modulename' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'modulename', l_91191)
    # SSA branch for the else part of an if statement (line 255)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'f6' (line 258)
    f6_91192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 13), 'f6')
    # Testing the type of an if condition (line 258)
    if_condition_91193 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 258, 13), f6_91192)
    # Assigning a type to the variable 'if_condition_91193' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 13), 'if_condition_91193', if_condition_91193)
    # SSA begins for if statement (line 258)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 259):
    
    # Assigning a Num to a Name (line 259):
    int_91194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 17), 'int')
    # Assigning a type to the variable 'f6' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'f6', int_91194)
    
    # Assigning a Name to a Name (line 260):
    
    # Assigning a Name to a Name (line 260):
    # Getting the type of 'l' (line 260)
    l_91195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 24), 'l')
    # Assigning a type to the variable 'buildpath' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'buildpath', l_91195)
    # SSA branch for the else part of an if statement (line 258)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'f7' (line 261)
    f7_91196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 13), 'f7')
    # Testing the type of an if condition (line 261)
    if_condition_91197 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 261, 13), f7_91196)
    # Assigning a type to the variable 'if_condition_91197' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 13), 'if_condition_91197', if_condition_91197)
    # SSA begins for if statement (line 261)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 262):
    
    # Assigning a Num to a Name (line 262):
    int_91198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 17), 'int')
    # Assigning a type to the variable 'f7' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'f7', int_91198)
    
    # Call to extend(...): (line 263)
    # Processing the call arguments (line 263)
    
    # Call to split(...): (line 263)
    # Processing the call arguments (line 263)
    # Getting the type of 'os' (line 263)
    os_91203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 41), 'os', False)
    # Obtaining the member 'pathsep' of a type (line 263)
    pathsep_91204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 41), os_91203, 'pathsep')
    # Processing the call keyword arguments (line 263)
    kwargs_91205 = {}
    # Getting the type of 'l' (line 263)
    l_91201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 33), 'l', False)
    # Obtaining the member 'split' of a type (line 263)
    split_91202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 33), l_91201, 'split')
    # Calling split(args, kwargs) (line 263)
    split_call_result_91206 = invoke(stypy.reporting.localization.Localization(__file__, 263, 33), split_91202, *[pathsep_91204], **kwargs_91205)
    
    # Processing the call keyword arguments (line 263)
    kwargs_91207 = {}
    # Getting the type of 'include_paths' (line 263)
    include_paths_91199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'include_paths', False)
    # Obtaining the member 'extend' of a type (line 263)
    extend_91200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 12), include_paths_91199, 'extend')
    # Calling extend(args, kwargs) (line 263)
    extend_call_result_91208 = invoke(stypy.reporting.localization.Localization(__file__, 263, 12), extend_91200, *[split_call_result_91206], **kwargs_91207)
    
    # SSA branch for the else part of an if statement (line 261)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'f8' (line 264)
    f8_91209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 13), 'f8')
    # Testing the type of an if condition (line 264)
    if_condition_91210 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 264, 13), f8_91209)
    # Assigning a type to the variable 'if_condition_91210' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 13), 'if_condition_91210', if_condition_91210)
    # SSA begins for if statement (line 264)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 265):
    
    # Assigning a Num to a Name (line 265):
    int_91211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 17), 'int')
    # Assigning a type to the variable 'f8' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'f8', int_91211)
    
    # Assigning a Name to a Subscript (line 266):
    
    # Assigning a Name to a Subscript (line 266):
    # Getting the type of 'l' (line 266)
    l_91212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 33), 'l')
    # Getting the type of 'options' (line 266)
    options_91213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'options')
    str_91214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 20), 'str', 'coutput')
    # Storing an element on a container (line 266)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 12), options_91213, (str_91214, l_91212))
    # SSA branch for the else part of an if statement (line 264)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'f9' (line 267)
    f9_91215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 13), 'f9')
    # Testing the type of an if condition (line 267)
    if_condition_91216 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 267, 13), f9_91215)
    # Assigning a type to the variable 'if_condition_91216' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 13), 'if_condition_91216', if_condition_91216)
    # SSA begins for if statement (line 267)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 268):
    
    # Assigning a Num to a Name (line 268):
    int_91217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 17), 'int')
    # Assigning a type to the variable 'f9' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'f9', int_91217)
    
    # Assigning a Name to a Subscript (line 269):
    
    # Assigning a Name to a Subscript (line 269):
    # Getting the type of 'l' (line 269)
    l_91218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 45), 'l')
    # Getting the type of 'options' (line 269)
    options_91219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'options')
    str_91220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 20), 'str', 'f2py_wrapper_output')
    # Storing an element on a container (line 269)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 12), options_91219, (str_91220, l_91218))
    # SSA branch for the else part of an if statement (line 267)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'f' (line 270)
    f_91221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 13), 'f')
    int_91222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 18), 'int')
    # Applying the binary operator '==' (line 270)
    result_eq_91223 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 13), '==', f_91221, int_91222)
    
    # Testing the type of an if condition (line 270)
    if_condition_91224 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 270, 13), result_eq_91223)
    # Assigning a type to the variable 'if_condition_91224' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 13), 'if_condition_91224', if_condition_91224)
    # SSA begins for if statement (line 270)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 271)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to close(...): (line 272)
    # Processing the call keyword arguments (line 272)
    kwargs_91230 = {}
    
    # Call to open(...): (line 272)
    # Processing the call arguments (line 272)
    # Getting the type of 'l' (line 272)
    l_91226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 21), 'l', False)
    # Processing the call keyword arguments (line 272)
    kwargs_91227 = {}
    # Getting the type of 'open' (line 272)
    open_91225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 16), 'open', False)
    # Calling open(args, kwargs) (line 272)
    open_call_result_91228 = invoke(stypy.reporting.localization.Localization(__file__, 272, 16), open_91225, *[l_91226], **kwargs_91227)
    
    # Obtaining the member 'close' of a type (line 272)
    close_91229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 16), open_call_result_91228, 'close')
    # Calling close(args, kwargs) (line 272)
    close_call_result_91231 = invoke(stypy.reporting.localization.Localization(__file__, 272, 16), close_91229, *[], **kwargs_91230)
    
    
    # Call to append(...): (line 273)
    # Processing the call arguments (line 273)
    # Getting the type of 'l' (line 273)
    l_91234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 29), 'l', False)
    # Processing the call keyword arguments (line 273)
    kwargs_91235 = {}
    # Getting the type of 'files' (line 273)
    files_91232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 16), 'files', False)
    # Obtaining the member 'append' of a type (line 273)
    append_91233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 16), files_91232, 'append')
    # Calling append(args, kwargs) (line 273)
    append_call_result_91236 = invoke(stypy.reporting.localization.Localization(__file__, 273, 16), append_91233, *[l_91234], **kwargs_91235)
    
    # SSA branch for the except part of a try statement (line 271)
    # SSA branch for the except 'IOError' branch of a try statement (line 271)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'IOError' (line 274)
    IOError_91237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 19), 'IOError')
    # Assigning a type to the variable 'detail' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'detail', IOError_91237)
    
    # Call to errmess(...): (line 275)
    # Processing the call arguments (line 275)
    str_91239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 24), 'str', 'IOError: %s. Skipping file "%s".\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 276)
    tuple_91240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 276)
    # Adding element type (line 276)
    
    # Call to str(...): (line 276)
    # Processing the call arguments (line 276)
    # Getting the type of 'detail' (line 276)
    detail_91242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 29), 'detail', False)
    # Processing the call keyword arguments (line 276)
    kwargs_91243 = {}
    # Getting the type of 'str' (line 276)
    str_91241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 25), 'str', False)
    # Calling str(args, kwargs) (line 276)
    str_call_result_91244 = invoke(stypy.reporting.localization.Localization(__file__, 276, 25), str_91241, *[detail_91242], **kwargs_91243)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 25), tuple_91240, str_call_result_91244)
    # Adding element type (line 276)
    # Getting the type of 'l' (line 276)
    l_91245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 38), 'l', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 25), tuple_91240, l_91245)
    
    # Applying the binary operator '%' (line 275)
    result_mod_91246 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 24), '%', str_91239, tuple_91240)
    
    # Processing the call keyword arguments (line 275)
    kwargs_91247 = {}
    # Getting the type of 'errmess' (line 275)
    errmess_91238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'errmess', False)
    # Calling errmess(args, kwargs) (line 275)
    errmess_call_result_91248 = invoke(stypy.reporting.localization.Localization(__file__, 275, 16), errmess_91238, *[result_mod_91246], **kwargs_91247)
    
    # SSA join for try-except statement (line 271)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 270)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'f' (line 277)
    f_91249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 13), 'f')
    int_91250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 18), 'int')
    # Applying the binary operator '==' (line 277)
    result_eq_91251 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 13), '==', f_91249, int_91250)
    
    # Testing the type of an if condition (line 277)
    if_condition_91252 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 277, 13), result_eq_91251)
    # Assigning a type to the variable 'if_condition_91252' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 13), 'if_condition_91252', if_condition_91252)
    # SSA begins for if statement (line 277)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 278)
    # Processing the call arguments (line 278)
    # Getting the type of 'l' (line 278)
    l_91255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 29), 'l', False)
    # Processing the call keyword arguments (line 278)
    kwargs_91256 = {}
    # Getting the type of 'skipfuncs' (line 278)
    skipfuncs_91253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'skipfuncs', False)
    # Obtaining the member 'append' of a type (line 278)
    append_91254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 12), skipfuncs_91253, 'append')
    # Calling append(args, kwargs) (line 278)
    append_call_result_91257 = invoke(stypy.reporting.localization.Localization(__file__, 278, 12), append_91254, *[l_91255], **kwargs_91256)
    
    # SSA branch for the else part of an if statement (line 277)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'f' (line 279)
    f_91258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 13), 'f')
    int_91259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 18), 'int')
    # Applying the binary operator '==' (line 279)
    result_eq_91260 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 13), '==', f_91258, int_91259)
    
    # Testing the type of an if condition (line 279)
    if_condition_91261 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 279, 13), result_eq_91260)
    # Assigning a type to the variable 'if_condition_91261' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 13), 'if_condition_91261', if_condition_91261)
    # SSA begins for if statement (line 279)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 280)
    # Processing the call arguments (line 280)
    # Getting the type of 'l' (line 280)
    l_91264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 29), 'l', False)
    # Processing the call keyword arguments (line 280)
    kwargs_91265 = {}
    # Getting the type of 'onlyfuncs' (line 280)
    onlyfuncs_91262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'onlyfuncs', False)
    # Obtaining the member 'append' of a type (line 280)
    append_91263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 12), onlyfuncs_91262, 'append')
    # Calling append(args, kwargs) (line 280)
    append_call_result_91266 = invoke(stypy.reporting.localization.Localization(__file__, 280, 12), append_91263, *[l_91264], **kwargs_91265)
    
    # SSA join for if statement (line 279)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 277)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 270)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 267)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 264)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 261)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 258)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 255)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 252)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 249)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 247)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 243)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 240)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 238)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 235)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 233)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 231)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 229)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 227)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 225)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 223)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 221)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 219)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 217)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 215)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 213)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 211)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 209)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 207)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 205)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 203)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 201)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 199)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 197)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 195)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 193)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 191)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'f5' (line 281)
    f5_91267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 11), 'f5')
    # Applying the 'not' unary operator (line 281)
    result_not__91268 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 7), 'not', f5_91267)
    
    
    # Getting the type of 'files' (line 281)
    files_91269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 22), 'files')
    # Applying the 'not' unary operator (line 281)
    result_not__91270 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 18), 'not', files_91269)
    
    # Applying the binary operator 'and' (line 281)
    result_and_keyword_91271 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 7), 'and', result_not__91268, result_not__91270)
    
    # Getting the type of 'modulename' (line 281)
    modulename_91272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 36), 'modulename')
    # Applying the 'not' unary operator (line 281)
    result_not__91273 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 32), 'not', modulename_91272)
    
    # Applying the binary operator 'and' (line 281)
    result_and_keyword_91274 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 7), 'and', result_and_keyword_91271, result_not__91273)
    
    # Testing the type of an if condition (line 281)
    if_condition_91275 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 281, 4), result_and_keyword_91274)
    # Assigning a type to the variable 'if_condition_91275' (line 281)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'if_condition_91275', if_condition_91275)
    # SSA begins for if statement (line 281)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 282)
    # Processing the call arguments (line 282)
    # Getting the type of '__usage__' (line 282)
    usage___91277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 14), '__usage__', False)
    # Processing the call keyword arguments (line 282)
    kwargs_91278 = {}
    # Getting the type of 'print' (line 282)
    print_91276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'print', False)
    # Calling print(args, kwargs) (line 282)
    print_call_result_91279 = invoke(stypy.reporting.localization.Localization(__file__, 282, 8), print_91276, *[usage___91277], **kwargs_91278)
    
    
    # Call to exit(...): (line 283)
    # Processing the call keyword arguments (line 283)
    kwargs_91282 = {}
    # Getting the type of 'sys' (line 283)
    sys_91280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'sys', False)
    # Obtaining the member 'exit' of a type (line 283)
    exit_91281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 8), sys_91280, 'exit')
    # Calling exit(args, kwargs) (line 283)
    exit_call_result_91283 = invoke(stypy.reporting.localization.Localization(__file__, 283, 8), exit_91281, *[], **kwargs_91282)
    
    # SSA join for if statement (line 281)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to isdir(...): (line 284)
    # Processing the call arguments (line 284)
    # Getting the type of 'buildpath' (line 284)
    buildpath_91287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 25), 'buildpath', False)
    # Processing the call keyword arguments (line 284)
    kwargs_91288 = {}
    # Getting the type of 'os' (line 284)
    os_91284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 284)
    path_91285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 11), os_91284, 'path')
    # Obtaining the member 'isdir' of a type (line 284)
    isdir_91286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 11), path_91285, 'isdir')
    # Calling isdir(args, kwargs) (line 284)
    isdir_call_result_91289 = invoke(stypy.reporting.localization.Localization(__file__, 284, 11), isdir_91286, *[buildpath_91287], **kwargs_91288)
    
    # Applying the 'not' unary operator (line 284)
    result_not__91290 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 7), 'not', isdir_call_result_91289)
    
    # Testing the type of an if condition (line 284)
    if_condition_91291 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 284, 4), result_not__91290)
    # Assigning a type to the variable 'if_condition_91291' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'if_condition_91291', if_condition_91291)
    # SSA begins for if statement (line 284)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'verbose' (line 285)
    verbose_91292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 15), 'verbose')
    # Applying the 'not' unary operator (line 285)
    result_not__91293 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 11), 'not', verbose_91292)
    
    # Testing the type of an if condition (line 285)
    if_condition_91294 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 285, 8), result_not__91293)
    # Assigning a type to the variable 'if_condition_91294' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'if_condition_91294', if_condition_91294)
    # SSA begins for if statement (line 285)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to outmess(...): (line 286)
    # Processing the call arguments (line 286)
    str_91296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 20), 'str', 'Creating build directory %s')
    # Getting the type of 'buildpath' (line 286)
    buildpath_91297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 53), 'buildpath', False)
    # Applying the binary operator '%' (line 286)
    result_mod_91298 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 20), '%', str_91296, buildpath_91297)
    
    # Processing the call keyword arguments (line 286)
    kwargs_91299 = {}
    # Getting the type of 'outmess' (line 286)
    outmess_91295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'outmess', False)
    # Calling outmess(args, kwargs) (line 286)
    outmess_call_result_91300 = invoke(stypy.reporting.localization.Localization(__file__, 286, 12), outmess_91295, *[result_mod_91298], **kwargs_91299)
    
    # SSA join for if statement (line 285)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to mkdir(...): (line 287)
    # Processing the call arguments (line 287)
    # Getting the type of 'buildpath' (line 287)
    buildpath_91303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 17), 'buildpath', False)
    # Processing the call keyword arguments (line 287)
    kwargs_91304 = {}
    # Getting the type of 'os' (line 287)
    os_91301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'os', False)
    # Obtaining the member 'mkdir' of a type (line 287)
    mkdir_91302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), os_91301, 'mkdir')
    # Calling mkdir(args, kwargs) (line 287)
    mkdir_call_result_91305 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), mkdir_91302, *[buildpath_91303], **kwargs_91304)
    
    # SSA join for if statement (line 284)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'signsfile' (line 288)
    signsfile_91306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 7), 'signsfile')
    # Testing the type of an if condition (line 288)
    if_condition_91307 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 288, 4), signsfile_91306)
    # Assigning a type to the variable 'if_condition_91307' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'if_condition_91307', if_condition_91307)
    # SSA begins for if statement (line 288)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 289):
    
    # Assigning a Call to a Name (line 289):
    
    # Call to join(...): (line 289)
    # Processing the call arguments (line 289)
    # Getting the type of 'buildpath' (line 289)
    buildpath_91311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 33), 'buildpath', False)
    # Getting the type of 'signsfile' (line 289)
    signsfile_91312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 44), 'signsfile', False)
    # Processing the call keyword arguments (line 289)
    kwargs_91313 = {}
    # Getting the type of 'os' (line 289)
    os_91308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 20), 'os', False)
    # Obtaining the member 'path' of a type (line 289)
    path_91309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 20), os_91308, 'path')
    # Obtaining the member 'join' of a type (line 289)
    join_91310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 20), path_91309, 'join')
    # Calling join(args, kwargs) (line 289)
    join_call_result_91314 = invoke(stypy.reporting.localization.Localization(__file__, 289, 20), join_91310, *[buildpath_91311, signsfile_91312], **kwargs_91313)
    
    # Assigning a type to the variable 'signsfile' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'signsfile', join_call_result_91314)
    # SSA join for if statement (line 288)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'signsfile' (line 290)
    signsfile_91315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 7), 'signsfile')
    
    # Call to isfile(...): (line 290)
    # Processing the call arguments (line 290)
    # Getting the type of 'signsfile' (line 290)
    signsfile_91319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 36), 'signsfile', False)
    # Processing the call keyword arguments (line 290)
    kwargs_91320 = {}
    # Getting the type of 'os' (line 290)
    os_91316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 21), 'os', False)
    # Obtaining the member 'path' of a type (line 290)
    path_91317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 21), os_91316, 'path')
    # Obtaining the member 'isfile' of a type (line 290)
    isfile_91318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 21), path_91317, 'isfile')
    # Calling isfile(args, kwargs) (line 290)
    isfile_call_result_91321 = invoke(stypy.reporting.localization.Localization(__file__, 290, 21), isfile_91318, *[signsfile_91319], **kwargs_91320)
    
    # Applying the binary operator 'and' (line 290)
    result_and_keyword_91322 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 7), 'and', signsfile_91315, isfile_call_result_91321)
    
    str_91323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 51), 'str', 'h-overwrite')
    # Getting the type of 'options' (line 290)
    options_91324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 72), 'options')
    # Applying the binary operator 'notin' (line 290)
    result_contains_91325 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 51), 'notin', str_91323, options_91324)
    
    # Applying the binary operator 'and' (line 290)
    result_and_keyword_91326 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 7), 'and', result_and_keyword_91322, result_contains_91325)
    
    # Testing the type of an if condition (line 290)
    if_condition_91327 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 290, 4), result_and_keyword_91326)
    # Assigning a type to the variable 'if_condition_91327' (line 290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'if_condition_91327', if_condition_91327)
    # SSA begins for if statement (line 290)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to errmess(...): (line 291)
    # Processing the call arguments (line 291)
    str_91329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 12), 'str', 'Signature file "%s" exists!!! Use --overwrite-signature to overwrite.\n')
    # Getting the type of 'signsfile' (line 292)
    signsfile_91330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 89), 'signsfile', False)
    # Applying the binary operator '%' (line 292)
    result_mod_91331 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 12), '%', str_91329, signsfile_91330)
    
    # Processing the call keyword arguments (line 291)
    kwargs_91332 = {}
    # Getting the type of 'errmess' (line 291)
    errmess_91328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'errmess', False)
    # Calling errmess(args, kwargs) (line 291)
    errmess_call_result_91333 = invoke(stypy.reporting.localization.Localization(__file__, 291, 8), errmess_91328, *[result_mod_91331], **kwargs_91332)
    
    
    # Call to exit(...): (line 293)
    # Processing the call keyword arguments (line 293)
    kwargs_91336 = {}
    # Getting the type of 'sys' (line 293)
    sys_91334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'sys', False)
    # Obtaining the member 'exit' of a type (line 293)
    exit_91335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 8), sys_91334, 'exit')
    # Calling exit(args, kwargs) (line 293)
    exit_call_result_91337 = invoke(stypy.reporting.localization.Localization(__file__, 293, 8), exit_91335, *[], **kwargs_91336)
    
    # SSA join for if statement (line 290)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 295):
    
    # Assigning a Name to a Subscript (line 295):
    # Getting the type of 'debug' (line 295)
    debug_91338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 23), 'debug')
    # Getting the type of 'options' (line 295)
    options_91339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'options')
    str_91340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 12), 'str', 'debug')
    # Storing an element on a container (line 295)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 4), options_91339, (str_91340, debug_91338))
    
    # Assigning a Name to a Subscript (line 296):
    
    # Assigning a Name to a Subscript (line 296):
    # Getting the type of 'verbose' (line 296)
    verbose_91341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 25), 'verbose')
    # Getting the type of 'options' (line 296)
    options_91342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 'options')
    str_91343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 12), 'str', 'verbose')
    # Storing an element on a container (line 296)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 4), options_91342, (str_91343, verbose_91341))
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'dolc' (line 297)
    dolc_91344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 7), 'dolc')
    int_91345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 15), 'int')
    # Applying the binary operator '==' (line 297)
    result_eq_91346 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 7), '==', dolc_91344, int_91345)
    
    
    # Getting the type of 'signsfile' (line 297)
    signsfile_91347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 26), 'signsfile')
    # Applying the 'not' unary operator (line 297)
    result_not__91348 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 22), 'not', signsfile_91347)
    
    # Applying the binary operator 'and' (line 297)
    result_and_keyword_91349 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 7), 'and', result_eq_91346, result_not__91348)
    
    # Testing the type of an if condition (line 297)
    if_condition_91350 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 297, 4), result_and_keyword_91349)
    # Assigning a type to the variable 'if_condition_91350' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'if_condition_91350', if_condition_91350)
    # SSA begins for if statement (line 297)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Subscript (line 298):
    
    # Assigning a Num to a Subscript (line 298):
    int_91351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 30), 'int')
    # Getting the type of 'options' (line 298)
    options_91352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'options')
    str_91353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 16), 'str', 'do-lower')
    # Storing an element on a container (line 298)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 8), options_91352, (str_91353, int_91351))
    # SSA branch for the else part of an if statement (line 297)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Subscript (line 300):
    
    # Assigning a Name to a Subscript (line 300):
    # Getting the type of 'dolc' (line 300)
    dolc_91354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 30), 'dolc')
    # Getting the type of 'options' (line 300)
    options_91355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'options')
    str_91356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 16), 'str', 'do-lower')
    # Storing an element on a container (line 300)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 8), options_91355, (str_91356, dolc_91354))
    # SSA join for if statement (line 297)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'modulename' (line 301)
    modulename_91357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 7), 'modulename')
    # Testing the type of an if condition (line 301)
    if_condition_91358 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 301, 4), modulename_91357)
    # Assigning a type to the variable 'if_condition_91358' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'if_condition_91358', if_condition_91358)
    # SSA begins for if statement (line 301)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 302):
    
    # Assigning a Name to a Subscript (line 302):
    # Getting the type of 'modulename' (line 302)
    modulename_91359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 28), 'modulename')
    # Getting the type of 'options' (line 302)
    options_91360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'options')
    str_91361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 16), 'str', 'module')
    # Storing an element on a container (line 302)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 8), options_91360, (str_91361, modulename_91359))
    # SSA join for if statement (line 301)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'signsfile' (line 303)
    signsfile_91362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 7), 'signsfile')
    # Testing the type of an if condition (line 303)
    if_condition_91363 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 303, 4), signsfile_91362)
    # Assigning a type to the variable 'if_condition_91363' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'if_condition_91363', if_condition_91363)
    # SSA begins for if statement (line 303)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 304):
    
    # Assigning a Name to a Subscript (line 304):
    # Getting the type of 'signsfile' (line 304)
    signsfile_91364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 31), 'signsfile')
    # Getting the type of 'options' (line 304)
    options_91365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'options')
    str_91366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 16), 'str', 'signsfile')
    # Storing an element on a container (line 304)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 8), options_91365, (str_91366, signsfile_91364))
    # SSA join for if statement (line 303)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'onlyfuncs' (line 305)
    onlyfuncs_91367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 7), 'onlyfuncs')
    # Testing the type of an if condition (line 305)
    if_condition_91368 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 305, 4), onlyfuncs_91367)
    # Assigning a type to the variable 'if_condition_91368' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'if_condition_91368', if_condition_91368)
    # SSA begins for if statement (line 305)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 306):
    
    # Assigning a Name to a Subscript (line 306):
    # Getting the type of 'onlyfuncs' (line 306)
    onlyfuncs_91369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 31), 'onlyfuncs')
    # Getting the type of 'options' (line 306)
    options_91370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'options')
    str_91371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 16), 'str', 'onlyfuncs')
    # Storing an element on a container (line 306)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 8), options_91370, (str_91371, onlyfuncs_91369))
    # SSA join for if statement (line 305)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'skipfuncs' (line 307)
    skipfuncs_91372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 7), 'skipfuncs')
    # Testing the type of an if condition (line 307)
    if_condition_91373 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 307, 4), skipfuncs_91372)
    # Assigning a type to the variable 'if_condition_91373' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'if_condition_91373', if_condition_91373)
    # SSA begins for if statement (line 307)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 308):
    
    # Assigning a Name to a Subscript (line 308):
    # Getting the type of 'skipfuncs' (line 308)
    skipfuncs_91374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 31), 'skipfuncs')
    # Getting the type of 'options' (line 308)
    options_91375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'options')
    str_91376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 16), 'str', 'skipfuncs')
    # Storing an element on a container (line 308)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 8), options_91375, (str_91376, skipfuncs_91374))
    # SSA join for if statement (line 307)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 309):
    
    # Assigning a Name to a Subscript (line 309):
    # Getting the type of 'dolatexdoc' (line 309)
    dolatexdoc_91377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 28), 'dolatexdoc')
    # Getting the type of 'options' (line 309)
    options_91378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'options')
    str_91379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 12), 'str', 'dolatexdoc')
    # Storing an element on a container (line 309)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 4), options_91378, (str_91379, dolatexdoc_91377))
    
    # Assigning a Name to a Subscript (line 310):
    
    # Assigning a Name to a Subscript (line 310):
    # Getting the type of 'dorestdoc' (line 310)
    dorestdoc_91380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 27), 'dorestdoc')
    # Getting the type of 'options' (line 310)
    options_91381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'options')
    str_91382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 12), 'str', 'dorestdoc')
    # Storing an element on a container (line 310)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 4), options_91381, (str_91382, dorestdoc_91380))
    
    # Assigning a Name to a Subscript (line 311):
    
    # Assigning a Name to a Subscript (line 311):
    # Getting the type of 'wrapfuncs' (line 311)
    wrapfuncs_91383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 27), 'wrapfuncs')
    # Getting the type of 'options' (line 311)
    options_91384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'options')
    str_91385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 12), 'str', 'wrapfuncs')
    # Storing an element on a container (line 311)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 4), options_91384, (str_91385, wrapfuncs_91383))
    
    # Assigning a Name to a Subscript (line 312):
    
    # Assigning a Name to a Subscript (line 312):
    # Getting the type of 'buildpath' (line 312)
    buildpath_91386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 27), 'buildpath')
    # Getting the type of 'options' (line 312)
    options_91387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'options')
    str_91388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 12), 'str', 'buildpath')
    # Storing an element on a container (line 312)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 4), options_91387, (str_91388, buildpath_91386))
    
    # Assigning a Name to a Subscript (line 313):
    
    # Assigning a Name to a Subscript (line 313):
    # Getting the type of 'include_paths' (line 313)
    include_paths_91389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 31), 'include_paths')
    # Getting the type of 'options' (line 313)
    options_91390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'options')
    str_91391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 12), 'str', 'include_paths')
    # Storing an element on a container (line 313)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 4), options_91390, (str_91391, include_paths_91389))
    
    # Obtaining an instance of the builtin type 'tuple' (line 314)
    tuple_91392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 314)
    # Adding element type (line 314)
    # Getting the type of 'files' (line 314)
    files_91393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 11), 'files')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 11), tuple_91392, files_91393)
    # Adding element type (line 314)
    # Getting the type of 'options' (line 314)
    options_91394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 18), 'options')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 11), tuple_91392, options_91394)
    
    # Assigning a type to the variable 'stypy_return_type' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'stypy_return_type', tuple_91392)
    
    # ################# End of 'scaninputline(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'scaninputline' in the type store
    # Getting the type of 'stypy_return_type' (line 176)
    stypy_return_type_91395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_91395)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'scaninputline'
    return stypy_return_type_91395

# Assigning a type to the variable 'scaninputline' (line 176)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'scaninputline', scaninputline)

@norecursion
def callcrackfortran(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'callcrackfortran'
    module_type_store = module_type_store.open_function_context('callcrackfortran', 317, 0, False)
    
    # Passed parameters checking function
    callcrackfortran.stypy_localization = localization
    callcrackfortran.stypy_type_of_self = None
    callcrackfortran.stypy_type_store = module_type_store
    callcrackfortran.stypy_function_name = 'callcrackfortran'
    callcrackfortran.stypy_param_names_list = ['files', 'options']
    callcrackfortran.stypy_varargs_param_name = None
    callcrackfortran.stypy_kwargs_param_name = None
    callcrackfortran.stypy_call_defaults = defaults
    callcrackfortran.stypy_call_varargs = varargs
    callcrackfortran.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'callcrackfortran', ['files', 'options'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'callcrackfortran', localization, ['files', 'options'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'callcrackfortran(...)' code ##################

    
    # Assigning a Name to a Attribute (line 318):
    
    # Assigning a Name to a Attribute (line 318):
    # Getting the type of 'options' (line 318)
    options_91396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 20), 'options')
    # Getting the type of 'rules' (line 318)
    rules_91397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'rules')
    # Setting the type of the member 'options' of a type (line 318)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 4), rules_91397, 'options', options_91396)
    
    # Assigning a Subscript to a Attribute (line 319):
    
    # Assigning a Subscript to a Attribute (line 319):
    
    # Obtaining the type of the subscript
    str_91398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 33), 'str', 'debug')
    # Getting the type of 'options' (line 319)
    options_91399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 25), 'options')
    # Obtaining the member '__getitem__' of a type (line 319)
    getitem___91400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 25), options_91399, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 319)
    subscript_call_result_91401 = invoke(stypy.reporting.localization.Localization(__file__, 319, 25), getitem___91400, str_91398)
    
    # Getting the type of 'crackfortran' (line 319)
    crackfortran_91402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'crackfortran')
    # Setting the type of the member 'debug' of a type (line 319)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 4), crackfortran_91402, 'debug', subscript_call_result_91401)
    
    # Assigning a Subscript to a Attribute (line 320):
    
    # Assigning a Subscript to a Attribute (line 320):
    
    # Obtaining the type of the subscript
    str_91403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 35), 'str', 'verbose')
    # Getting the type of 'options' (line 320)
    options_91404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 27), 'options')
    # Obtaining the member '__getitem__' of a type (line 320)
    getitem___91405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 27), options_91404, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 320)
    subscript_call_result_91406 = invoke(stypy.reporting.localization.Localization(__file__, 320, 27), getitem___91405, str_91403)
    
    # Getting the type of 'crackfortran' (line 320)
    crackfortran_91407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'crackfortran')
    # Setting the type of the member 'verbose' of a type (line 320)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 4), crackfortran_91407, 'verbose', subscript_call_result_91406)
    
    
    str_91408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 7), 'str', 'module')
    # Getting the type of 'options' (line 321)
    options_91409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 19), 'options')
    # Applying the binary operator 'in' (line 321)
    result_contains_91410 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 7), 'in', str_91408, options_91409)
    
    # Testing the type of an if condition (line 321)
    if_condition_91411 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 321, 4), result_contains_91410)
    # Assigning a type to the variable 'if_condition_91411' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'if_condition_91411', if_condition_91411)
    # SSA begins for if statement (line 321)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Attribute (line 322):
    
    # Assigning a Subscript to a Attribute (line 322):
    
    # Obtaining the type of the subscript
    str_91412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 45), 'str', 'module')
    # Getting the type of 'options' (line 322)
    options_91413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 37), 'options')
    # Obtaining the member '__getitem__' of a type (line 322)
    getitem___91414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 37), options_91413, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 322)
    subscript_call_result_91415 = invoke(stypy.reporting.localization.Localization(__file__, 322, 37), getitem___91414, str_91412)
    
    # Getting the type of 'crackfortran' (line 322)
    crackfortran_91416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'crackfortran')
    # Setting the type of the member 'f77modulename' of a type (line 322)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 8), crackfortran_91416, 'f77modulename', subscript_call_result_91415)
    # SSA join for if statement (line 321)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_91417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 7), 'str', 'skipfuncs')
    # Getting the type of 'options' (line 323)
    options_91418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 22), 'options')
    # Applying the binary operator 'in' (line 323)
    result_contains_91419 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 7), 'in', str_91417, options_91418)
    
    # Testing the type of an if condition (line 323)
    if_condition_91420 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 323, 4), result_contains_91419)
    # Assigning a type to the variable 'if_condition_91420' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'if_condition_91420', if_condition_91420)
    # SSA begins for if statement (line 323)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Attribute (line 324):
    
    # Assigning a Subscript to a Attribute (line 324):
    
    # Obtaining the type of the subscript
    str_91421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 41), 'str', 'skipfuncs')
    # Getting the type of 'options' (line 324)
    options_91422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 33), 'options')
    # Obtaining the member '__getitem__' of a type (line 324)
    getitem___91423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 33), options_91422, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 324)
    subscript_call_result_91424 = invoke(stypy.reporting.localization.Localization(__file__, 324, 33), getitem___91423, str_91421)
    
    # Getting the type of 'crackfortran' (line 324)
    crackfortran_91425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'crackfortran')
    # Setting the type of the member 'skipfuncs' of a type (line 324)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 8), crackfortran_91425, 'skipfuncs', subscript_call_result_91424)
    # SSA join for if statement (line 323)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_91426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 7), 'str', 'onlyfuncs')
    # Getting the type of 'options' (line 325)
    options_91427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 22), 'options')
    # Applying the binary operator 'in' (line 325)
    result_contains_91428 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 7), 'in', str_91426, options_91427)
    
    # Testing the type of an if condition (line 325)
    if_condition_91429 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 325, 4), result_contains_91428)
    # Assigning a type to the variable 'if_condition_91429' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'if_condition_91429', if_condition_91429)
    # SSA begins for if statement (line 325)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Attribute (line 326):
    
    # Assigning a Subscript to a Attribute (line 326):
    
    # Obtaining the type of the subscript
    str_91430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 41), 'str', 'onlyfuncs')
    # Getting the type of 'options' (line 326)
    options_91431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 33), 'options')
    # Obtaining the member '__getitem__' of a type (line 326)
    getitem___91432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 33), options_91431, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 326)
    subscript_call_result_91433 = invoke(stypy.reporting.localization.Localization(__file__, 326, 33), getitem___91432, str_91430)
    
    # Getting the type of 'crackfortran' (line 326)
    crackfortran_91434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'crackfortran')
    # Setting the type of the member 'onlyfuncs' of a type (line 326)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 8), crackfortran_91434, 'onlyfuncs', subscript_call_result_91433)
    # SSA join for if statement (line 325)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Subscript (line 327):
    
    # Assigning a Subscript to a Subscript (line 327):
    
    # Obtaining the type of the subscript
    str_91435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 44), 'str', 'include_paths')
    # Getting the type of 'options' (line 327)
    options_91436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 36), 'options')
    # Obtaining the member '__getitem__' of a type (line 327)
    getitem___91437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 36), options_91436, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 327)
    subscript_call_result_91438 = invoke(stypy.reporting.localization.Localization(__file__, 327, 36), getitem___91437, str_91435)
    
    # Getting the type of 'crackfortran' (line 327)
    crackfortran_91439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'crackfortran')
    # Obtaining the member 'include_paths' of a type (line 327)
    include_paths_91440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 4), crackfortran_91439, 'include_paths')
    slice_91441 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 327, 4), None, None, None)
    # Storing an element on a container (line 327)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 4), include_paths_91440, (slice_91441, subscript_call_result_91438))
    
    # Assigning a Subscript to a Attribute (line 328):
    
    # Assigning a Subscript to a Attribute (line 328):
    
    # Obtaining the type of the subscript
    str_91442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 39), 'str', 'do-lower')
    # Getting the type of 'options' (line 328)
    options_91443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 31), 'options')
    # Obtaining the member '__getitem__' of a type (line 328)
    getitem___91444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 31), options_91443, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 328)
    subscript_call_result_91445 = invoke(stypy.reporting.localization.Localization(__file__, 328, 31), getitem___91444, str_91442)
    
    # Getting the type of 'crackfortran' (line 328)
    crackfortran_91446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'crackfortran')
    # Setting the type of the member 'dolowercase' of a type (line 328)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 4), crackfortran_91446, 'dolowercase', subscript_call_result_91445)
    
    # Assigning a Call to a Name (line 329):
    
    # Assigning a Call to a Name (line 329):
    
    # Call to crackfortran(...): (line 329)
    # Processing the call arguments (line 329)
    # Getting the type of 'files' (line 329)
    files_91449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 41), 'files', False)
    # Processing the call keyword arguments (line 329)
    kwargs_91450 = {}
    # Getting the type of 'crackfortran' (line 329)
    crackfortran_91447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 15), 'crackfortran', False)
    # Obtaining the member 'crackfortran' of a type (line 329)
    crackfortran_91448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 15), crackfortran_91447, 'crackfortran')
    # Calling crackfortran(args, kwargs) (line 329)
    crackfortran_call_result_91451 = invoke(stypy.reporting.localization.Localization(__file__, 329, 15), crackfortran_91448, *[files_91449], **kwargs_91450)
    
    # Assigning a type to the variable 'postlist' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'postlist', crackfortran_call_result_91451)
    
    
    str_91452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 7), 'str', 'signsfile')
    # Getting the type of 'options' (line 330)
    options_91453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 22), 'options')
    # Applying the binary operator 'in' (line 330)
    result_contains_91454 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 7), 'in', str_91452, options_91453)
    
    # Testing the type of an if condition (line 330)
    if_condition_91455 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 330, 4), result_contains_91454)
    # Assigning a type to the variable 'if_condition_91455' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'if_condition_91455', if_condition_91455)
    # SSA begins for if statement (line 330)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to outmess(...): (line 331)
    # Processing the call arguments (line 331)
    str_91457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 16), 'str', 'Saving signatures to file "%s"\n')
    
    # Obtaining the type of the subscript
    str_91458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 62), 'str', 'signsfile')
    # Getting the type of 'options' (line 331)
    options_91459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 54), 'options', False)
    # Obtaining the member '__getitem__' of a type (line 331)
    getitem___91460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 54), options_91459, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 331)
    subscript_call_result_91461 = invoke(stypy.reporting.localization.Localization(__file__, 331, 54), getitem___91460, str_91458)
    
    # Applying the binary operator '%' (line 331)
    result_mod_91462 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 16), '%', str_91457, subscript_call_result_91461)
    
    # Processing the call keyword arguments (line 331)
    kwargs_91463 = {}
    # Getting the type of 'outmess' (line 331)
    outmess_91456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'outmess', False)
    # Calling outmess(args, kwargs) (line 331)
    outmess_call_result_91464 = invoke(stypy.reporting.localization.Localization(__file__, 331, 8), outmess_91456, *[result_mod_91462], **kwargs_91463)
    
    
    # Assigning a Call to a Name (line 332):
    
    # Assigning a Call to a Name (line 332):
    
    # Call to crack2fortran(...): (line 332)
    # Processing the call arguments (line 332)
    # Getting the type of 'postlist' (line 332)
    postlist_91467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 41), 'postlist', False)
    # Processing the call keyword arguments (line 332)
    kwargs_91468 = {}
    # Getting the type of 'crackfortran' (line 332)
    crackfortran_91465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 14), 'crackfortran', False)
    # Obtaining the member 'crack2fortran' of a type (line 332)
    crack2fortran_91466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 14), crackfortran_91465, 'crack2fortran')
    # Calling crack2fortran(args, kwargs) (line 332)
    crack2fortran_call_result_91469 = invoke(stypy.reporting.localization.Localization(__file__, 332, 14), crack2fortran_91466, *[postlist_91467], **kwargs_91468)
    
    # Assigning a type to the variable 'pyf' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'pyf', crack2fortran_call_result_91469)
    
    
    
    # Obtaining the type of the subscript
    int_91470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 32), 'int')
    slice_91471 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 333, 11), int_91470, None, None)
    
    # Obtaining the type of the subscript
    str_91472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 19), 'str', 'signsfile')
    # Getting the type of 'options' (line 333)
    options_91473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 11), 'options')
    # Obtaining the member '__getitem__' of a type (line 333)
    getitem___91474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 11), options_91473, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 333)
    subscript_call_result_91475 = invoke(stypy.reporting.localization.Localization(__file__, 333, 11), getitem___91474, str_91472)
    
    # Obtaining the member '__getitem__' of a type (line 333)
    getitem___91476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 11), subscript_call_result_91475, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 333)
    subscript_call_result_91477 = invoke(stypy.reporting.localization.Localization(__file__, 333, 11), getitem___91476, slice_91471)
    
    str_91478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 40), 'str', 'stdout')
    # Applying the binary operator '==' (line 333)
    result_eq_91479 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 11), '==', subscript_call_result_91477, str_91478)
    
    # Testing the type of an if condition (line 333)
    if_condition_91480 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 333, 8), result_eq_91479)
    # Assigning a type to the variable 'if_condition_91480' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'if_condition_91480', if_condition_91480)
    # SSA begins for if statement (line 333)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to write(...): (line 334)
    # Processing the call arguments (line 334)
    # Getting the type of 'pyf' (line 334)
    pyf_91484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 29), 'pyf', False)
    # Processing the call keyword arguments (line 334)
    kwargs_91485 = {}
    # Getting the type of 'sys' (line 334)
    sys_91481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'sys', False)
    # Obtaining the member 'stdout' of a type (line 334)
    stdout_91482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 12), sys_91481, 'stdout')
    # Obtaining the member 'write' of a type (line 334)
    write_91483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 12), stdout_91482, 'write')
    # Calling write(args, kwargs) (line 334)
    write_call_result_91486 = invoke(stypy.reporting.localization.Localization(__file__, 334, 12), write_91483, *[pyf_91484], **kwargs_91485)
    
    # SSA branch for the else part of an if statement (line 333)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 336):
    
    # Assigning a Call to a Name (line 336):
    
    # Call to open(...): (line 336)
    # Processing the call arguments (line 336)
    
    # Obtaining the type of the subscript
    str_91488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 29), 'str', 'signsfile')
    # Getting the type of 'options' (line 336)
    options_91489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 21), 'options', False)
    # Obtaining the member '__getitem__' of a type (line 336)
    getitem___91490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 21), options_91489, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 336)
    subscript_call_result_91491 = invoke(stypy.reporting.localization.Localization(__file__, 336, 21), getitem___91490, str_91488)
    
    str_91492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 43), 'str', 'w')
    # Processing the call keyword arguments (line 336)
    kwargs_91493 = {}
    # Getting the type of 'open' (line 336)
    open_91487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 16), 'open', False)
    # Calling open(args, kwargs) (line 336)
    open_call_result_91494 = invoke(stypy.reporting.localization.Localization(__file__, 336, 16), open_91487, *[subscript_call_result_91491, str_91492], **kwargs_91493)
    
    # Assigning a type to the variable 'f' (line 336)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'f', open_call_result_91494)
    
    # Call to write(...): (line 337)
    # Processing the call arguments (line 337)
    # Getting the type of 'pyf' (line 337)
    pyf_91497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 20), 'pyf', False)
    # Processing the call keyword arguments (line 337)
    kwargs_91498 = {}
    # Getting the type of 'f' (line 337)
    f_91495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'f', False)
    # Obtaining the member 'write' of a type (line 337)
    write_91496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 12), f_91495, 'write')
    # Calling write(args, kwargs) (line 337)
    write_call_result_91499 = invoke(stypy.reporting.localization.Localization(__file__, 337, 12), write_91496, *[pyf_91497], **kwargs_91498)
    
    
    # Call to close(...): (line 338)
    # Processing the call keyword arguments (line 338)
    kwargs_91502 = {}
    # Getting the type of 'f' (line 338)
    f_91500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'f', False)
    # Obtaining the member 'close' of a type (line 338)
    close_91501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 12), f_91500, 'close')
    # Calling close(args, kwargs) (line 338)
    close_call_result_91503 = invoke(stypy.reporting.localization.Localization(__file__, 338, 12), close_91501, *[], **kwargs_91502)
    
    # SSA join for if statement (line 333)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 330)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 339)
    
    # Obtaining the type of the subscript
    str_91504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 15), 'str', 'coutput')
    # Getting the type of 'options' (line 339)
    options_91505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 7), 'options')
    # Obtaining the member '__getitem__' of a type (line 339)
    getitem___91506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 7), options_91505, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 339)
    subscript_call_result_91507 = invoke(stypy.reporting.localization.Localization(__file__, 339, 7), getitem___91506, str_91504)
    
    # Getting the type of 'None' (line 339)
    None_91508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 29), 'None')
    
    (may_be_91509, more_types_in_union_91510) = may_be_none(subscript_call_result_91507, None_91508)

    if may_be_91509:

        if more_types_in_union_91510:
            # Runtime conditional SSA (line 339)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Getting the type of 'postlist' (line 340)
        postlist_91511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 19), 'postlist')
        # Testing the type of a for loop iterable (line 340)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 340, 8), postlist_91511)
        # Getting the type of the for loop variable (line 340)
        for_loop_var_91512 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 340, 8), postlist_91511)
        # Assigning a type to the variable 'mod' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'mod', for_loop_var_91512)
        # SSA begins for a for statement (line 340)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Subscript (line 341):
        
        # Assigning a BinOp to a Subscript (line 341):
        str_91513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 29), 'str', '%smodule.c')
        
        # Obtaining the type of the subscript
        str_91514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 48), 'str', 'name')
        # Getting the type of 'mod' (line 341)
        mod_91515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 44), 'mod')
        # Obtaining the member '__getitem__' of a type (line 341)
        getitem___91516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 44), mod_91515, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 341)
        subscript_call_result_91517 = invoke(stypy.reporting.localization.Localization(__file__, 341, 44), getitem___91516, str_91514)
        
        # Applying the binary operator '%' (line 341)
        result_mod_91518 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 29), '%', str_91513, subscript_call_result_91517)
        
        # Getting the type of 'mod' (line 341)
        mod_91519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'mod')
        str_91520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 16), 'str', 'coutput')
        # Storing an element on a container (line 341)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 12), mod_91519, (str_91520, result_mod_91518))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_91510:
            # Runtime conditional SSA for else branch (line 339)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_91509) or more_types_in_union_91510):
        
        # Getting the type of 'postlist' (line 343)
        postlist_91521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 19), 'postlist')
        # Testing the type of a for loop iterable (line 343)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 343, 8), postlist_91521)
        # Getting the type of the for loop variable (line 343)
        for_loop_var_91522 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 343, 8), postlist_91521)
        # Assigning a type to the variable 'mod' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'mod', for_loop_var_91522)
        # SSA begins for a for statement (line 343)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Subscript (line 344):
        
        # Assigning a Subscript to a Subscript (line 344):
        
        # Obtaining the type of the subscript
        str_91523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 37), 'str', 'coutput')
        # Getting the type of 'options' (line 344)
        options_91524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 29), 'options')
        # Obtaining the member '__getitem__' of a type (line 344)
        getitem___91525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 29), options_91524, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 344)
        subscript_call_result_91526 = invoke(stypy.reporting.localization.Localization(__file__, 344, 29), getitem___91525, str_91523)
        
        # Getting the type of 'mod' (line 344)
        mod_91527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'mod')
        str_91528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 16), 'str', 'coutput')
        # Storing an element on a container (line 344)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 12), mod_91527, (str_91528, subscript_call_result_91526))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_91509 and more_types_in_union_91510):
            # SSA join for if statement (line 339)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 345)
    
    # Obtaining the type of the subscript
    str_91529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 15), 'str', 'f2py_wrapper_output')
    # Getting the type of 'options' (line 345)
    options_91530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 7), 'options')
    # Obtaining the member '__getitem__' of a type (line 345)
    getitem___91531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 7), options_91530, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 345)
    subscript_call_result_91532 = invoke(stypy.reporting.localization.Localization(__file__, 345, 7), getitem___91531, str_91529)
    
    # Getting the type of 'None' (line 345)
    None_91533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 41), 'None')
    
    (may_be_91534, more_types_in_union_91535) = may_be_none(subscript_call_result_91532, None_91533)

    if may_be_91534:

        if more_types_in_union_91535:
            # Runtime conditional SSA (line 345)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Getting the type of 'postlist' (line 346)
        postlist_91536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 19), 'postlist')
        # Testing the type of a for loop iterable (line 346)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 346, 8), postlist_91536)
        # Getting the type of the for loop variable (line 346)
        for_loop_var_91537 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 346, 8), postlist_91536)
        # Assigning a type to the variable 'mod' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'mod', for_loop_var_91537)
        # SSA begins for a for statement (line 346)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Subscript (line 347):
        
        # Assigning a BinOp to a Subscript (line 347):
        str_91538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 41), 'str', '%s-f2pywrappers.f')
        
        # Obtaining the type of the subscript
        str_91539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 67), 'str', 'name')
        # Getting the type of 'mod' (line 347)
        mod_91540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 63), 'mod')
        # Obtaining the member '__getitem__' of a type (line 347)
        getitem___91541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 63), mod_91540, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 347)
        subscript_call_result_91542 = invoke(stypy.reporting.localization.Localization(__file__, 347, 63), getitem___91541, str_91539)
        
        # Applying the binary operator '%' (line 347)
        result_mod_91543 = python_operator(stypy.reporting.localization.Localization(__file__, 347, 41), '%', str_91538, subscript_call_result_91542)
        
        # Getting the type of 'mod' (line 347)
        mod_91544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'mod')
        str_91545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 16), 'str', 'f2py_wrapper_output')
        # Storing an element on a container (line 347)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 12), mod_91544, (str_91545, result_mod_91543))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_91535:
            # Runtime conditional SSA for else branch (line 345)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_91534) or more_types_in_union_91535):
        
        # Getting the type of 'postlist' (line 349)
        postlist_91546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 19), 'postlist')
        # Testing the type of a for loop iterable (line 349)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 349, 8), postlist_91546)
        # Getting the type of the for loop variable (line 349)
        for_loop_var_91547 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 349, 8), postlist_91546)
        # Assigning a type to the variable 'mod' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'mod', for_loop_var_91547)
        # SSA begins for a for statement (line 349)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Subscript (line 350):
        
        # Assigning a Subscript to a Subscript (line 350):
        
        # Obtaining the type of the subscript
        str_91548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 49), 'str', 'f2py_wrapper_output')
        # Getting the type of 'options' (line 350)
        options_91549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 41), 'options')
        # Obtaining the member '__getitem__' of a type (line 350)
        getitem___91550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 41), options_91549, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 350)
        subscript_call_result_91551 = invoke(stypy.reporting.localization.Localization(__file__, 350, 41), getitem___91550, str_91548)
        
        # Getting the type of 'mod' (line 350)
        mod_91552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 12), 'mod')
        str_91553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 16), 'str', 'f2py_wrapper_output')
        # Storing an element on a container (line 350)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 12), mod_91552, (str_91553, subscript_call_result_91551))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_91534 and more_types_in_union_91535):
            # SSA join for if statement (line 345)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'postlist' (line 351)
    postlist_91554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 11), 'postlist')
    # Assigning a type to the variable 'stypy_return_type' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'stypy_return_type', postlist_91554)
    
    # ################# End of 'callcrackfortran(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'callcrackfortran' in the type store
    # Getting the type of 'stypy_return_type' (line 317)
    stypy_return_type_91555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_91555)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'callcrackfortran'
    return stypy_return_type_91555

# Assigning a type to the variable 'callcrackfortran' (line 317)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 0), 'callcrackfortran', callcrackfortran)

@norecursion
def buildmodules(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'buildmodules'
    module_type_store = module_type_store.open_function_context('buildmodules', 354, 0, False)
    
    # Passed parameters checking function
    buildmodules.stypy_localization = localization
    buildmodules.stypy_type_of_self = None
    buildmodules.stypy_type_store = module_type_store
    buildmodules.stypy_function_name = 'buildmodules'
    buildmodules.stypy_param_names_list = ['lst']
    buildmodules.stypy_varargs_param_name = None
    buildmodules.stypy_kwargs_param_name = None
    buildmodules.stypy_call_defaults = defaults
    buildmodules.stypy_call_varargs = varargs
    buildmodules.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'buildmodules', ['lst'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'buildmodules', localization, ['lst'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'buildmodules(...)' code ##################

    
    # Call to buildcfuncs(...): (line 355)
    # Processing the call keyword arguments (line 355)
    kwargs_91558 = {}
    # Getting the type of 'cfuncs' (line 355)
    cfuncs_91556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'cfuncs', False)
    # Obtaining the member 'buildcfuncs' of a type (line 355)
    buildcfuncs_91557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 4), cfuncs_91556, 'buildcfuncs')
    # Calling buildcfuncs(args, kwargs) (line 355)
    buildcfuncs_call_result_91559 = invoke(stypy.reporting.localization.Localization(__file__, 355, 4), buildcfuncs_91557, *[], **kwargs_91558)
    
    
    # Call to outmess(...): (line 356)
    # Processing the call arguments (line 356)
    str_91561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 12), 'str', 'Building modules...\n')
    # Processing the call keyword arguments (line 356)
    kwargs_91562 = {}
    # Getting the type of 'outmess' (line 356)
    outmess_91560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'outmess', False)
    # Calling outmess(args, kwargs) (line 356)
    outmess_call_result_91563 = invoke(stypy.reporting.localization.Localization(__file__, 356, 4), outmess_91560, *[str_91561], **kwargs_91562)
    
    
    # Assigning a Tuple to a Tuple (line 357):
    
    # Assigning a List to a Name (line 357):
    
    # Obtaining an instance of the builtin type 'list' (line 357)
    list_91564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 357)
    
    # Assigning a type to the variable 'tuple_assignment_90852' (line 357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'tuple_assignment_90852', list_91564)
    
    # Assigning a List to a Name (line 357):
    
    # Obtaining an instance of the builtin type 'list' (line 357)
    list_91565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 357)
    
    # Assigning a type to the variable 'tuple_assignment_90853' (line 357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'tuple_assignment_90853', list_91565)
    
    # Assigning a Dict to a Name (line 357):
    
    # Obtaining an instance of the builtin type 'dict' (line 357)
    dict_91566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 40), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 357)
    
    # Assigning a type to the variable 'tuple_assignment_90854' (line 357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'tuple_assignment_90854', dict_91566)
    
    # Assigning a Name to a Name (line 357):
    # Getting the type of 'tuple_assignment_90852' (line 357)
    tuple_assignment_90852_91567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'tuple_assignment_90852')
    # Assigning a type to the variable 'modules' (line 357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'modules', tuple_assignment_90852_91567)
    
    # Assigning a Name to a Name (line 357):
    # Getting the type of 'tuple_assignment_90853' (line 357)
    tuple_assignment_90853_91568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'tuple_assignment_90853')
    # Assigning a type to the variable 'mnames' (line 357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 13), 'mnames', tuple_assignment_90853_91568)
    
    # Assigning a Name to a Name (line 357):
    # Getting the type of 'tuple_assignment_90854' (line 357)
    tuple_assignment_90854_91569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'tuple_assignment_90854')
    # Assigning a type to the variable 'isusedby' (line 357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 21), 'isusedby', tuple_assignment_90854_91569)
    
    
    # Call to range(...): (line 358)
    # Processing the call arguments (line 358)
    
    # Call to len(...): (line 358)
    # Processing the call arguments (line 358)
    # Getting the type of 'lst' (line 358)
    lst_91572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 23), 'lst', False)
    # Processing the call keyword arguments (line 358)
    kwargs_91573 = {}
    # Getting the type of 'len' (line 358)
    len_91571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 19), 'len', False)
    # Calling len(args, kwargs) (line 358)
    len_call_result_91574 = invoke(stypy.reporting.localization.Localization(__file__, 358, 19), len_91571, *[lst_91572], **kwargs_91573)
    
    # Processing the call keyword arguments (line 358)
    kwargs_91575 = {}
    # Getting the type of 'range' (line 358)
    range_91570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 13), 'range', False)
    # Calling range(args, kwargs) (line 358)
    range_call_result_91576 = invoke(stypy.reporting.localization.Localization(__file__, 358, 13), range_91570, *[len_call_result_91574], **kwargs_91575)
    
    # Testing the type of a for loop iterable (line 358)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 358, 4), range_call_result_91576)
    # Getting the type of the for loop variable (line 358)
    for_loop_var_91577 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 358, 4), range_call_result_91576)
    # Assigning a type to the variable 'i' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'i', for_loop_var_91577)
    # SSA begins for a for statement (line 358)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    str_91578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 11), 'str', '__user__')
    
    # Obtaining the type of the subscript
    str_91579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 32), 'str', 'name')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 359)
    i_91580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 29), 'i')
    # Getting the type of 'lst' (line 359)
    lst_91581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 25), 'lst')
    # Obtaining the member '__getitem__' of a type (line 359)
    getitem___91582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 25), lst_91581, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 359)
    subscript_call_result_91583 = invoke(stypy.reporting.localization.Localization(__file__, 359, 25), getitem___91582, i_91580)
    
    # Obtaining the member '__getitem__' of a type (line 359)
    getitem___91584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 25), subscript_call_result_91583, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 359)
    subscript_call_result_91585 = invoke(stypy.reporting.localization.Localization(__file__, 359, 25), getitem___91584, str_91579)
    
    # Applying the binary operator 'in' (line 359)
    result_contains_91586 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 11), 'in', str_91578, subscript_call_result_91585)
    
    # Testing the type of an if condition (line 359)
    if_condition_91587 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 359, 8), result_contains_91586)
    # Assigning a type to the variable 'if_condition_91587' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'if_condition_91587', if_condition_91587)
    # SSA begins for if statement (line 359)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to buildcallbacks(...): (line 360)
    # Processing the call arguments (line 360)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 360)
    i_91590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 40), 'i', False)
    # Getting the type of 'lst' (line 360)
    lst_91591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 36), 'lst', False)
    # Obtaining the member '__getitem__' of a type (line 360)
    getitem___91592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 36), lst_91591, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 360)
    subscript_call_result_91593 = invoke(stypy.reporting.localization.Localization(__file__, 360, 36), getitem___91592, i_91590)
    
    # Processing the call keyword arguments (line 360)
    kwargs_91594 = {}
    # Getting the type of 'cb_rules' (line 360)
    cb_rules_91588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'cb_rules', False)
    # Obtaining the member 'buildcallbacks' of a type (line 360)
    buildcallbacks_91589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 12), cb_rules_91588, 'buildcallbacks')
    # Calling buildcallbacks(args, kwargs) (line 360)
    buildcallbacks_call_result_91595 = invoke(stypy.reporting.localization.Localization(__file__, 360, 12), buildcallbacks_91589, *[subscript_call_result_91593], **kwargs_91594)
    
    # SSA branch for the else part of an if statement (line 359)
    module_type_store.open_ssa_branch('else')
    
    
    str_91596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 15), 'str', 'use')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 362)
    i_91597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 28), 'i')
    # Getting the type of 'lst' (line 362)
    lst_91598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 24), 'lst')
    # Obtaining the member '__getitem__' of a type (line 362)
    getitem___91599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 24), lst_91598, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 362)
    subscript_call_result_91600 = invoke(stypy.reporting.localization.Localization(__file__, 362, 24), getitem___91599, i_91597)
    
    # Applying the binary operator 'in' (line 362)
    result_contains_91601 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 15), 'in', str_91596, subscript_call_result_91600)
    
    # Testing the type of an if condition (line 362)
    if_condition_91602 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 362, 12), result_contains_91601)
    # Assigning a type to the variable 'if_condition_91602' (line 362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'if_condition_91602', if_condition_91602)
    # SSA begins for if statement (line 362)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to keys(...): (line 363)
    # Processing the call keyword arguments (line 363)
    kwargs_91611 = {}
    
    # Obtaining the type of the subscript
    str_91603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 32), 'str', 'use')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 363)
    i_91604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 29), 'i', False)
    # Getting the type of 'lst' (line 363)
    lst_91605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 25), 'lst', False)
    # Obtaining the member '__getitem__' of a type (line 363)
    getitem___91606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 25), lst_91605, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 363)
    subscript_call_result_91607 = invoke(stypy.reporting.localization.Localization(__file__, 363, 25), getitem___91606, i_91604)
    
    # Obtaining the member '__getitem__' of a type (line 363)
    getitem___91608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 25), subscript_call_result_91607, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 363)
    subscript_call_result_91609 = invoke(stypy.reporting.localization.Localization(__file__, 363, 25), getitem___91608, str_91603)
    
    # Obtaining the member 'keys' of a type (line 363)
    keys_91610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 25), subscript_call_result_91609, 'keys')
    # Calling keys(args, kwargs) (line 363)
    keys_call_result_91612 = invoke(stypy.reporting.localization.Localization(__file__, 363, 25), keys_91610, *[], **kwargs_91611)
    
    # Testing the type of a for loop iterable (line 363)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 363, 16), keys_call_result_91612)
    # Getting the type of the for loop variable (line 363)
    for_loop_var_91613 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 363, 16), keys_call_result_91612)
    # Assigning a type to the variable 'u' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 16), 'u', for_loop_var_91613)
    # SSA begins for a for statement (line 363)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'u' (line 364)
    u_91614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 23), 'u')
    # Getting the type of 'isusedby' (line 364)
    isusedby_91615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 32), 'isusedby')
    # Applying the binary operator 'notin' (line 364)
    result_contains_91616 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 23), 'notin', u_91614, isusedby_91615)
    
    # Testing the type of an if condition (line 364)
    if_condition_91617 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 364, 20), result_contains_91616)
    # Assigning a type to the variable 'if_condition_91617' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 20), 'if_condition_91617', if_condition_91617)
    # SSA begins for if statement (line 364)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Subscript (line 365):
    
    # Assigning a List to a Subscript (line 365):
    
    # Obtaining an instance of the builtin type 'list' (line 365)
    list_91618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 365)
    
    # Getting the type of 'isusedby' (line 365)
    isusedby_91619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 24), 'isusedby')
    # Getting the type of 'u' (line 365)
    u_91620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 33), 'u')
    # Storing an element on a container (line 365)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 24), isusedby_91619, (u_91620, list_91618))
    # SSA join for if statement (line 364)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 366)
    # Processing the call arguments (line 366)
    
    # Obtaining the type of the subscript
    str_91626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 46), 'str', 'name')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 366)
    i_91627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 43), 'i', False)
    # Getting the type of 'lst' (line 366)
    lst_91628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 39), 'lst', False)
    # Obtaining the member '__getitem__' of a type (line 366)
    getitem___91629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 39), lst_91628, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 366)
    subscript_call_result_91630 = invoke(stypy.reporting.localization.Localization(__file__, 366, 39), getitem___91629, i_91627)
    
    # Obtaining the member '__getitem__' of a type (line 366)
    getitem___91631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 39), subscript_call_result_91630, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 366)
    subscript_call_result_91632 = invoke(stypy.reporting.localization.Localization(__file__, 366, 39), getitem___91631, str_91626)
    
    # Processing the call keyword arguments (line 366)
    kwargs_91633 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'u' (line 366)
    u_91621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 29), 'u', False)
    # Getting the type of 'isusedby' (line 366)
    isusedby_91622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 20), 'isusedby', False)
    # Obtaining the member '__getitem__' of a type (line 366)
    getitem___91623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 20), isusedby_91622, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 366)
    subscript_call_result_91624 = invoke(stypy.reporting.localization.Localization(__file__, 366, 20), getitem___91623, u_91621)
    
    # Obtaining the member 'append' of a type (line 366)
    append_91625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 20), subscript_call_result_91624, 'append')
    # Calling append(args, kwargs) (line 366)
    append_call_result_91634 = invoke(stypy.reporting.localization.Localization(__file__, 366, 20), append_91625, *[subscript_call_result_91632], **kwargs_91633)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 362)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 367)
    # Processing the call arguments (line 367)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 367)
    i_91637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 31), 'i', False)
    # Getting the type of 'lst' (line 367)
    lst_91638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 27), 'lst', False)
    # Obtaining the member '__getitem__' of a type (line 367)
    getitem___91639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 27), lst_91638, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 367)
    subscript_call_result_91640 = invoke(stypy.reporting.localization.Localization(__file__, 367, 27), getitem___91639, i_91637)
    
    # Processing the call keyword arguments (line 367)
    kwargs_91641 = {}
    # Getting the type of 'modules' (line 367)
    modules_91635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'modules', False)
    # Obtaining the member 'append' of a type (line 367)
    append_91636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 12), modules_91635, 'append')
    # Calling append(args, kwargs) (line 367)
    append_call_result_91642 = invoke(stypy.reporting.localization.Localization(__file__, 367, 12), append_91636, *[subscript_call_result_91640], **kwargs_91641)
    
    
    # Call to append(...): (line 368)
    # Processing the call arguments (line 368)
    
    # Obtaining the type of the subscript
    str_91645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 33), 'str', 'name')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 368)
    i_91646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 30), 'i', False)
    # Getting the type of 'lst' (line 368)
    lst_91647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 26), 'lst', False)
    # Obtaining the member '__getitem__' of a type (line 368)
    getitem___91648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 26), lst_91647, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 368)
    subscript_call_result_91649 = invoke(stypy.reporting.localization.Localization(__file__, 368, 26), getitem___91648, i_91646)
    
    # Obtaining the member '__getitem__' of a type (line 368)
    getitem___91650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 26), subscript_call_result_91649, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 368)
    subscript_call_result_91651 = invoke(stypy.reporting.localization.Localization(__file__, 368, 26), getitem___91650, str_91645)
    
    # Processing the call keyword arguments (line 368)
    kwargs_91652 = {}
    # Getting the type of 'mnames' (line 368)
    mnames_91643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 12), 'mnames', False)
    # Obtaining the member 'append' of a type (line 368)
    append_91644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 12), mnames_91643, 'append')
    # Calling append(args, kwargs) (line 368)
    append_call_result_91653 = invoke(stypy.reporting.localization.Localization(__file__, 368, 12), append_91644, *[subscript_call_result_91651], **kwargs_91652)
    
    # SSA join for if statement (line 359)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Dict to a Name (line 369):
    
    # Assigning a Dict to a Name (line 369):
    
    # Obtaining an instance of the builtin type 'dict' (line 369)
    dict_91654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 10), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 369)
    
    # Assigning a type to the variable 'ret' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'ret', dict_91654)
    
    
    # Call to range(...): (line 370)
    # Processing the call arguments (line 370)
    
    # Call to len(...): (line 370)
    # Processing the call arguments (line 370)
    # Getting the type of 'mnames' (line 370)
    mnames_91657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 23), 'mnames', False)
    # Processing the call keyword arguments (line 370)
    kwargs_91658 = {}
    # Getting the type of 'len' (line 370)
    len_91656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 19), 'len', False)
    # Calling len(args, kwargs) (line 370)
    len_call_result_91659 = invoke(stypy.reporting.localization.Localization(__file__, 370, 19), len_91656, *[mnames_91657], **kwargs_91658)
    
    # Processing the call keyword arguments (line 370)
    kwargs_91660 = {}
    # Getting the type of 'range' (line 370)
    range_91655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 13), 'range', False)
    # Calling range(args, kwargs) (line 370)
    range_call_result_91661 = invoke(stypy.reporting.localization.Localization(__file__, 370, 13), range_91655, *[len_call_result_91659], **kwargs_91660)
    
    # Testing the type of a for loop iterable (line 370)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 370, 4), range_call_result_91661)
    # Getting the type of the for loop variable (line 370)
    for_loop_var_91662 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 370, 4), range_call_result_91661)
    # Assigning a type to the variable 'i' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'i', for_loop_var_91662)
    # SSA begins for a for statement (line 370)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 371)
    i_91663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 18), 'i')
    # Getting the type of 'mnames' (line 371)
    mnames_91664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 11), 'mnames')
    # Obtaining the member '__getitem__' of a type (line 371)
    getitem___91665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 11), mnames_91664, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 371)
    subscript_call_result_91666 = invoke(stypy.reporting.localization.Localization(__file__, 371, 11), getitem___91665, i_91663)
    
    # Getting the type of 'isusedby' (line 371)
    isusedby_91667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 24), 'isusedby')
    # Applying the binary operator 'in' (line 371)
    result_contains_91668 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 11), 'in', subscript_call_result_91666, isusedby_91667)
    
    # Testing the type of an if condition (line 371)
    if_condition_91669 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 371, 8), result_contains_91668)
    # Assigning a type to the variable 'if_condition_91669' (line 371)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'if_condition_91669', if_condition_91669)
    # SSA begins for if statement (line 371)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to outmess(...): (line 372)
    # Processing the call arguments (line 372)
    str_91671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 20), 'str', '\tSkipping module "%s" which is used by %s.\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 373)
    tuple_91672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 373)
    # Adding element type (line 373)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 373)
    i_91673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 23), 'i', False)
    # Getting the type of 'mnames' (line 373)
    mnames_91674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 16), 'mnames', False)
    # Obtaining the member '__getitem__' of a type (line 373)
    getitem___91675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 16), mnames_91674, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 373)
    subscript_call_result_91676 = invoke(stypy.reporting.localization.Localization(__file__, 373, 16), getitem___91675, i_91673)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 16), tuple_91672, subscript_call_result_91676)
    # Adding element type (line 373)
    
    # Call to join(...): (line 373)
    # Processing the call arguments (line 373)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 373)
    i_91682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 73), 'i', False)
    # Getting the type of 'mnames' (line 373)
    mnames_91683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 66), 'mnames', False)
    # Obtaining the member '__getitem__' of a type (line 373)
    getitem___91684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 66), mnames_91683, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 373)
    subscript_call_result_91685 = invoke(stypy.reporting.localization.Localization(__file__, 373, 66), getitem___91684, i_91682)
    
    # Getting the type of 'isusedby' (line 373)
    isusedby_91686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 57), 'isusedby', False)
    # Obtaining the member '__getitem__' of a type (line 373)
    getitem___91687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 57), isusedby_91686, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 373)
    subscript_call_result_91688 = invoke(stypy.reporting.localization.Localization(__file__, 373, 57), getitem___91687, subscript_call_result_91685)
    
    comprehension_91689 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 37), subscript_call_result_91688)
    # Assigning a type to the variable 's' (line 373)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 37), 's', comprehension_91689)
    str_91679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 37), 'str', '"%s"')
    # Getting the type of 's' (line 373)
    s_91680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 46), 's', False)
    # Applying the binary operator '%' (line 373)
    result_mod_91681 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 37), '%', str_91679, s_91680)
    
    list_91690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 37), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 37), list_91690, result_mod_91681)
    # Processing the call keyword arguments (line 373)
    kwargs_91691 = {}
    str_91677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 27), 'str', ',')
    # Obtaining the member 'join' of a type (line 373)
    join_91678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 27), str_91677, 'join')
    # Calling join(args, kwargs) (line 373)
    join_call_result_91692 = invoke(stypy.reporting.localization.Localization(__file__, 373, 27), join_91678, *[list_91690], **kwargs_91691)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 16), tuple_91672, join_call_result_91692)
    
    # Applying the binary operator '%' (line 372)
    result_mod_91693 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 20), '%', str_91671, tuple_91672)
    
    # Processing the call keyword arguments (line 372)
    kwargs_91694 = {}
    # Getting the type of 'outmess' (line 372)
    outmess_91670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'outmess', False)
    # Calling outmess(args, kwargs) (line 372)
    outmess_call_result_91695 = invoke(stypy.reporting.localization.Localization(__file__, 372, 12), outmess_91670, *[result_mod_91693], **kwargs_91694)
    
    # SSA branch for the else part of an if statement (line 371)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a List to a Name (line 375):
    
    # Assigning a List to a Name (line 375):
    
    # Obtaining an instance of the builtin type 'list' (line 375)
    list_91696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 375)
    
    # Assigning a type to the variable 'um' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 12), 'um', list_91696)
    
    
    str_91697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 15), 'str', 'use')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 376)
    i_91698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 32), 'i')
    # Getting the type of 'modules' (line 376)
    modules_91699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 24), 'modules')
    # Obtaining the member '__getitem__' of a type (line 376)
    getitem___91700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 24), modules_91699, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 376)
    subscript_call_result_91701 = invoke(stypy.reporting.localization.Localization(__file__, 376, 24), getitem___91700, i_91698)
    
    # Applying the binary operator 'in' (line 376)
    result_contains_91702 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 15), 'in', str_91697, subscript_call_result_91701)
    
    # Testing the type of an if condition (line 376)
    if_condition_91703 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 376, 12), result_contains_91702)
    # Assigning a type to the variable 'if_condition_91703' (line 376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'if_condition_91703', if_condition_91703)
    # SSA begins for if statement (line 376)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to keys(...): (line 377)
    # Processing the call keyword arguments (line 377)
    kwargs_91712 = {}
    
    # Obtaining the type of the subscript
    str_91704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 36), 'str', 'use')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 377)
    i_91705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 33), 'i', False)
    # Getting the type of 'modules' (line 377)
    modules_91706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 25), 'modules', False)
    # Obtaining the member '__getitem__' of a type (line 377)
    getitem___91707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 25), modules_91706, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 377)
    subscript_call_result_91708 = invoke(stypy.reporting.localization.Localization(__file__, 377, 25), getitem___91707, i_91705)
    
    # Obtaining the member '__getitem__' of a type (line 377)
    getitem___91709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 25), subscript_call_result_91708, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 377)
    subscript_call_result_91710 = invoke(stypy.reporting.localization.Localization(__file__, 377, 25), getitem___91709, str_91704)
    
    # Obtaining the member 'keys' of a type (line 377)
    keys_91711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 25), subscript_call_result_91710, 'keys')
    # Calling keys(args, kwargs) (line 377)
    keys_call_result_91713 = invoke(stypy.reporting.localization.Localization(__file__, 377, 25), keys_91711, *[], **kwargs_91712)
    
    # Testing the type of a for loop iterable (line 377)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 377, 16), keys_call_result_91713)
    # Getting the type of the for loop variable (line 377)
    for_loop_var_91714 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 377, 16), keys_call_result_91713)
    # Assigning a type to the variable 'u' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 16), 'u', for_loop_var_91714)
    # SSA begins for a for statement (line 377)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'u' (line 378)
    u_91715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 23), 'u')
    # Getting the type of 'isusedby' (line 378)
    isusedby_91716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 28), 'isusedby')
    # Applying the binary operator 'in' (line 378)
    result_contains_91717 = python_operator(stypy.reporting.localization.Localization(__file__, 378, 23), 'in', u_91715, isusedby_91716)
    
    
    # Getting the type of 'u' (line 378)
    u_91718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 41), 'u')
    # Getting the type of 'mnames' (line 378)
    mnames_91719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 46), 'mnames')
    # Applying the binary operator 'in' (line 378)
    result_contains_91720 = python_operator(stypy.reporting.localization.Localization(__file__, 378, 41), 'in', u_91718, mnames_91719)
    
    # Applying the binary operator 'and' (line 378)
    result_and_keyword_91721 = python_operator(stypy.reporting.localization.Localization(__file__, 378, 23), 'and', result_contains_91717, result_contains_91720)
    
    # Testing the type of an if condition (line 378)
    if_condition_91722 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 378, 20), result_and_keyword_91721)
    # Assigning a type to the variable 'if_condition_91722' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 20), 'if_condition_91722', if_condition_91722)
    # SSA begins for if statement (line 378)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 379)
    # Processing the call arguments (line 379)
    
    # Obtaining the type of the subscript
    
    # Call to index(...): (line 379)
    # Processing the call arguments (line 379)
    # Getting the type of 'u' (line 379)
    u_91727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 55), 'u', False)
    # Processing the call keyword arguments (line 379)
    kwargs_91728 = {}
    # Getting the type of 'mnames' (line 379)
    mnames_91725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 42), 'mnames', False)
    # Obtaining the member 'index' of a type (line 379)
    index_91726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 42), mnames_91725, 'index')
    # Calling index(args, kwargs) (line 379)
    index_call_result_91729 = invoke(stypy.reporting.localization.Localization(__file__, 379, 42), index_91726, *[u_91727], **kwargs_91728)
    
    # Getting the type of 'modules' (line 379)
    modules_91730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 34), 'modules', False)
    # Obtaining the member '__getitem__' of a type (line 379)
    getitem___91731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 34), modules_91730, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 379)
    subscript_call_result_91732 = invoke(stypy.reporting.localization.Localization(__file__, 379, 34), getitem___91731, index_call_result_91729)
    
    # Processing the call keyword arguments (line 379)
    kwargs_91733 = {}
    # Getting the type of 'um' (line 379)
    um_91723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 24), 'um', False)
    # Obtaining the member 'append' of a type (line 379)
    append_91724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 24), um_91723, 'append')
    # Calling append(args, kwargs) (line 379)
    append_call_result_91734 = invoke(stypy.reporting.localization.Localization(__file__, 379, 24), append_91724, *[subscript_call_result_91732], **kwargs_91733)
    
    # SSA branch for the else part of an if statement (line 378)
    module_type_store.open_ssa_branch('else')
    
    # Call to outmess(...): (line 381)
    # Processing the call arguments (line 381)
    str_91736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 28), 'str', '\tModule "%s" uses nonexisting "%s" which will be ignored.\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 382)
    tuple_91737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 94), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 382)
    # Adding element type (line 382)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 382)
    i_91738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 101), 'i', False)
    # Getting the type of 'mnames' (line 382)
    mnames_91739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 94), 'mnames', False)
    # Obtaining the member '__getitem__' of a type (line 382)
    getitem___91740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 94), mnames_91739, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 382)
    subscript_call_result_91741 = invoke(stypy.reporting.localization.Localization(__file__, 382, 94), getitem___91740, i_91738)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 94), tuple_91737, subscript_call_result_91741)
    # Adding element type (line 382)
    # Getting the type of 'u' (line 382)
    u_91742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 105), 'u', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 94), tuple_91737, u_91742)
    
    # Applying the binary operator '%' (line 382)
    result_mod_91743 = python_operator(stypy.reporting.localization.Localization(__file__, 382, 28), '%', str_91736, tuple_91737)
    
    # Processing the call keyword arguments (line 381)
    kwargs_91744 = {}
    # Getting the type of 'outmess' (line 381)
    outmess_91735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 24), 'outmess', False)
    # Calling outmess(args, kwargs) (line 381)
    outmess_call_result_91745 = invoke(stypy.reporting.localization.Localization(__file__, 381, 24), outmess_91735, *[result_mod_91743], **kwargs_91744)
    
    # SSA join for if statement (line 378)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 376)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Dict to a Subscript (line 383):
    
    # Assigning a Dict to a Subscript (line 383):
    
    # Obtaining an instance of the builtin type 'dict' (line 383)
    dict_91746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 29), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 383)
    
    # Getting the type of 'ret' (line 383)
    ret_91747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'ret')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 383)
    i_91748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 23), 'i')
    # Getting the type of 'mnames' (line 383)
    mnames_91749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 16), 'mnames')
    # Obtaining the member '__getitem__' of a type (line 383)
    getitem___91750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 16), mnames_91749, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 383)
    subscript_call_result_91751 = invoke(stypy.reporting.localization.Localization(__file__, 383, 16), getitem___91750, i_91748)
    
    # Storing an element on a container (line 383)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 12), ret_91747, (subscript_call_result_91751, dict_91746))
    
    # Call to dict_append(...): (line 384)
    # Processing the call arguments (line 384)
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 384)
    i_91753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 35), 'i', False)
    # Getting the type of 'mnames' (line 384)
    mnames_91754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 28), 'mnames', False)
    # Obtaining the member '__getitem__' of a type (line 384)
    getitem___91755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 28), mnames_91754, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 384)
    subscript_call_result_91756 = invoke(stypy.reporting.localization.Localization(__file__, 384, 28), getitem___91755, i_91753)
    
    # Getting the type of 'ret' (line 384)
    ret_91757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 24), 'ret', False)
    # Obtaining the member '__getitem__' of a type (line 384)
    getitem___91758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 24), ret_91757, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 384)
    subscript_call_result_91759 = invoke(stypy.reporting.localization.Localization(__file__, 384, 24), getitem___91758, subscript_call_result_91756)
    
    
    # Call to buildmodule(...): (line 384)
    # Processing the call arguments (line 384)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 384)
    i_91762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 66), 'i', False)
    # Getting the type of 'modules' (line 384)
    modules_91763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 58), 'modules', False)
    # Obtaining the member '__getitem__' of a type (line 384)
    getitem___91764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 58), modules_91763, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 384)
    subscript_call_result_91765 = invoke(stypy.reporting.localization.Localization(__file__, 384, 58), getitem___91764, i_91762)
    
    # Getting the type of 'um' (line 384)
    um_91766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 70), 'um', False)
    # Processing the call keyword arguments (line 384)
    kwargs_91767 = {}
    # Getting the type of 'rules' (line 384)
    rules_91760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 40), 'rules', False)
    # Obtaining the member 'buildmodule' of a type (line 384)
    buildmodule_91761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 40), rules_91760, 'buildmodule')
    # Calling buildmodule(args, kwargs) (line 384)
    buildmodule_call_result_91768 = invoke(stypy.reporting.localization.Localization(__file__, 384, 40), buildmodule_91761, *[subscript_call_result_91765, um_91766], **kwargs_91767)
    
    # Processing the call keyword arguments (line 384)
    kwargs_91769 = {}
    # Getting the type of 'dict_append' (line 384)
    dict_append_91752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'dict_append', False)
    # Calling dict_append(args, kwargs) (line 384)
    dict_append_call_result_91770 = invoke(stypy.reporting.localization.Localization(__file__, 384, 12), dict_append_91752, *[subscript_call_result_91759, buildmodule_call_result_91768], **kwargs_91769)
    
    # SSA join for if statement (line 371)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ret' (line 385)
    ret_91771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 11), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'stypy_return_type', ret_91771)
    
    # ################# End of 'buildmodules(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'buildmodules' in the type store
    # Getting the type of 'stypy_return_type' (line 354)
    stypy_return_type_91772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_91772)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'buildmodules'
    return stypy_return_type_91772

# Assigning a type to the variable 'buildmodules' (line 354)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 0), 'buildmodules', buildmodules)

@norecursion
def dict_append(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'dict_append'
    module_type_store = module_type_store.open_function_context('dict_append', 388, 0, False)
    
    # Passed parameters checking function
    dict_append.stypy_localization = localization
    dict_append.stypy_type_of_self = None
    dict_append.stypy_type_store = module_type_store
    dict_append.stypy_function_name = 'dict_append'
    dict_append.stypy_param_names_list = ['d_out', 'd_in']
    dict_append.stypy_varargs_param_name = None
    dict_append.stypy_kwargs_param_name = None
    dict_append.stypy_call_defaults = defaults
    dict_append.stypy_call_varargs = varargs
    dict_append.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dict_append', ['d_out', 'd_in'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dict_append', localization, ['d_out', 'd_in'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dict_append(...)' code ##################

    
    
    # Call to items(...): (line 389)
    # Processing the call keyword arguments (line 389)
    kwargs_91775 = {}
    # Getting the type of 'd_in' (line 389)
    d_in_91773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 18), 'd_in', False)
    # Obtaining the member 'items' of a type (line 389)
    items_91774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 18), d_in_91773, 'items')
    # Calling items(args, kwargs) (line 389)
    items_call_result_91776 = invoke(stypy.reporting.localization.Localization(__file__, 389, 18), items_91774, *[], **kwargs_91775)
    
    # Testing the type of a for loop iterable (line 389)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 389, 4), items_call_result_91776)
    # Getting the type of the for loop variable (line 389)
    for_loop_var_91777 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 389, 4), items_call_result_91776)
    # Assigning a type to the variable 'k' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 4), for_loop_var_91777))
    # Assigning a type to the variable 'v' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 4), for_loop_var_91777))
    # SSA begins for a for statement (line 389)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'k' (line 390)
    k_91778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 11), 'k')
    # Getting the type of 'd_out' (line 390)
    d_out_91779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 20), 'd_out')
    # Applying the binary operator 'notin' (line 390)
    result_contains_91780 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 11), 'notin', k_91778, d_out_91779)
    
    # Testing the type of an if condition (line 390)
    if_condition_91781 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 390, 8), result_contains_91780)
    # Assigning a type to the variable 'if_condition_91781' (line 390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'if_condition_91781', if_condition_91781)
    # SSA begins for if statement (line 390)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Subscript (line 391):
    
    # Assigning a List to a Subscript (line 391):
    
    # Obtaining an instance of the builtin type 'list' (line 391)
    list_91782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 391)
    
    # Getting the type of 'd_out' (line 391)
    d_out_91783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 12), 'd_out')
    # Getting the type of 'k' (line 391)
    k_91784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 18), 'k')
    # Storing an element on a container (line 391)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 12), d_out_91783, (k_91784, list_91782))
    # SSA join for if statement (line 390)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 392)
    # Getting the type of 'list' (line 392)
    list_91785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 25), 'list')
    # Getting the type of 'v' (line 392)
    v_91786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 22), 'v')
    
    (may_be_91787, more_types_in_union_91788) = may_be_subtype(list_91785, v_91786)

    if may_be_91787:

        if more_types_in_union_91788:
            # Runtime conditional SSA (line 392)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'v' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'v', remove_not_subtype_from_union(v_91786, list))
        
        # Assigning a BinOp to a Subscript (line 393):
        
        # Assigning a BinOp to a Subscript (line 393):
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 393)
        k_91789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 29), 'k')
        # Getting the type of 'd_out' (line 393)
        d_out_91790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 23), 'd_out')
        # Obtaining the member '__getitem__' of a type (line 393)
        getitem___91791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 23), d_out_91790, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 393)
        subscript_call_result_91792 = invoke(stypy.reporting.localization.Localization(__file__, 393, 23), getitem___91791, k_91789)
        
        # Getting the type of 'v' (line 393)
        v_91793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 34), 'v')
        # Applying the binary operator '+' (line 393)
        result_add_91794 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 23), '+', subscript_call_result_91792, v_91793)
        
        # Getting the type of 'd_out' (line 393)
        d_out_91795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'd_out')
        # Getting the type of 'k' (line 393)
        k_91796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 18), 'k')
        # Storing an element on a container (line 393)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 12), d_out_91795, (k_91796, result_add_91794))

        if more_types_in_union_91788:
            # Runtime conditional SSA for else branch (line 392)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_91787) or more_types_in_union_91788):
        # Assigning a type to the variable 'v' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'v', remove_subtype_from_union(v_91786, list))
        
        # Call to append(...): (line 395)
        # Processing the call arguments (line 395)
        # Getting the type of 'v' (line 395)
        v_91802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 28), 'v', False)
        # Processing the call keyword arguments (line 395)
        kwargs_91803 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 395)
        k_91797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 18), 'k', False)
        # Getting the type of 'd_out' (line 395)
        d_out_91798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'd_out', False)
        # Obtaining the member '__getitem__' of a type (line 395)
        getitem___91799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 12), d_out_91798, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 395)
        subscript_call_result_91800 = invoke(stypy.reporting.localization.Localization(__file__, 395, 12), getitem___91799, k_91797)
        
        # Obtaining the member 'append' of a type (line 395)
        append_91801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 12), subscript_call_result_91800, 'append')
        # Calling append(args, kwargs) (line 395)
        append_call_result_91804 = invoke(stypy.reporting.localization.Localization(__file__, 395, 12), append_91801, *[v_91802], **kwargs_91803)
        

        if (may_be_91787 and more_types_in_union_91788):
            # SSA join for if statement (line 392)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'dict_append(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dict_append' in the type store
    # Getting the type of 'stypy_return_type' (line 388)
    stypy_return_type_91805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_91805)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dict_append'
    return stypy_return_type_91805

# Assigning a type to the variable 'dict_append' (line 388)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 0), 'dict_append', dict_append)

@norecursion
def run_main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run_main'
    module_type_store = module_type_store.open_function_context('run_main', 398, 0, False)
    
    # Passed parameters checking function
    run_main.stypy_localization = localization
    run_main.stypy_type_of_self = None
    run_main.stypy_type_store = module_type_store
    run_main.stypy_function_name = 'run_main'
    run_main.stypy_param_names_list = ['comline_list']
    run_main.stypy_varargs_param_name = None
    run_main.stypy_kwargs_param_name = None
    run_main.stypy_call_defaults = defaults
    run_main.stypy_call_varargs = varargs
    run_main.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'run_main', ['comline_list'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'run_main', localization, ['comline_list'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'run_main(...)' code ##################

    str_91806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, (-1)), 'str', "Run f2py as if string.join(comline_list,' ') is used as a command line.\n    In case of using -h flag, return None.\n    ")
    
    # Call to reset_global_f2py_vars(...): (line 402)
    # Processing the call keyword arguments (line 402)
    kwargs_91809 = {}
    # Getting the type of 'crackfortran' (line 402)
    crackfortran_91807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'crackfortran', False)
    # Obtaining the member 'reset_global_f2py_vars' of a type (line 402)
    reset_global_f2py_vars_91808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 4), crackfortran_91807, 'reset_global_f2py_vars')
    # Calling reset_global_f2py_vars(args, kwargs) (line 402)
    reset_global_f2py_vars_call_result_91810 = invoke(stypy.reporting.localization.Localization(__file__, 402, 4), reset_global_f2py_vars_91808, *[], **kwargs_91809)
    
    
    # Assigning a Call to a Name (line 403):
    
    # Assigning a Call to a Name (line 403):
    
    # Call to dirname(...): (line 403)
    # Processing the call arguments (line 403)
    
    # Call to abspath(...): (line 403)
    # Processing the call arguments (line 403)
    # Getting the type of 'cfuncs' (line 403)
    cfuncs_91817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 46), 'cfuncs', False)
    # Obtaining the member '__file__' of a type (line 403)
    file___91818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 46), cfuncs_91817, '__file__')
    # Processing the call keyword arguments (line 403)
    kwargs_91819 = {}
    # Getting the type of 'os' (line 403)
    os_91814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 30), 'os', False)
    # Obtaining the member 'path' of a type (line 403)
    path_91815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 30), os_91814, 'path')
    # Obtaining the member 'abspath' of a type (line 403)
    abspath_91816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 30), path_91815, 'abspath')
    # Calling abspath(args, kwargs) (line 403)
    abspath_call_result_91820 = invoke(stypy.reporting.localization.Localization(__file__, 403, 30), abspath_91816, *[file___91818], **kwargs_91819)
    
    # Processing the call keyword arguments (line 403)
    kwargs_91821 = {}
    # Getting the type of 'os' (line 403)
    os_91811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 14), 'os', False)
    # Obtaining the member 'path' of a type (line 403)
    path_91812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 14), os_91811, 'path')
    # Obtaining the member 'dirname' of a type (line 403)
    dirname_91813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 14), path_91812, 'dirname')
    # Calling dirname(args, kwargs) (line 403)
    dirname_call_result_91822 = invoke(stypy.reporting.localization.Localization(__file__, 403, 14), dirname_91813, *[abspath_call_result_91820], **kwargs_91821)
    
    # Assigning a type to the variable 'f2pydir' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'f2pydir', dirname_call_result_91822)
    
    # Assigning a Call to a Name (line 404):
    
    # Assigning a Call to a Name (line 404):
    
    # Call to join(...): (line 404)
    # Processing the call arguments (line 404)
    # Getting the type of 'f2pydir' (line 404)
    f2pydir_91826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 28), 'f2pydir', False)
    str_91827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 37), 'str', 'src')
    str_91828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 44), 'str', 'fortranobject.h')
    # Processing the call keyword arguments (line 404)
    kwargs_91829 = {}
    # Getting the type of 'os' (line 404)
    os_91823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 404)
    path_91824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 15), os_91823, 'path')
    # Obtaining the member 'join' of a type (line 404)
    join_91825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 15), path_91824, 'join')
    # Calling join(args, kwargs) (line 404)
    join_call_result_91830 = invoke(stypy.reporting.localization.Localization(__file__, 404, 15), join_91825, *[f2pydir_91826, str_91827, str_91828], **kwargs_91829)
    
    # Assigning a type to the variable 'fobjhsrc' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'fobjhsrc', join_call_result_91830)
    
    # Assigning a Call to a Name (line 405):
    
    # Assigning a Call to a Name (line 405):
    
    # Call to join(...): (line 405)
    # Processing the call arguments (line 405)
    # Getting the type of 'f2pydir' (line 405)
    f2pydir_91834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 28), 'f2pydir', False)
    str_91835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 37), 'str', 'src')
    str_91836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 44), 'str', 'fortranobject.c')
    # Processing the call keyword arguments (line 405)
    kwargs_91837 = {}
    # Getting the type of 'os' (line 405)
    os_91831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 405)
    path_91832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 15), os_91831, 'path')
    # Obtaining the member 'join' of a type (line 405)
    join_91833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 15), path_91832, 'join')
    # Calling join(args, kwargs) (line 405)
    join_call_result_91838 = invoke(stypy.reporting.localization.Localization(__file__, 405, 15), join_91833, *[f2pydir_91834, str_91835, str_91836], **kwargs_91837)
    
    # Assigning a type to the variable 'fobjcsrc' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'fobjcsrc', join_call_result_91838)
    
    # Assigning a Call to a Tuple (line 406):
    
    # Assigning a Call to a Name:
    
    # Call to scaninputline(...): (line 406)
    # Processing the call arguments (line 406)
    # Getting the type of 'comline_list' (line 406)
    comline_list_91840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 35), 'comline_list', False)
    # Processing the call keyword arguments (line 406)
    kwargs_91841 = {}
    # Getting the type of 'scaninputline' (line 406)
    scaninputline_91839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 21), 'scaninputline', False)
    # Calling scaninputline(args, kwargs) (line 406)
    scaninputline_call_result_91842 = invoke(stypy.reporting.localization.Localization(__file__, 406, 21), scaninputline_91839, *[comline_list_91840], **kwargs_91841)
    
    # Assigning a type to the variable 'call_assignment_90855' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'call_assignment_90855', scaninputline_call_result_91842)
    
    # Assigning a Call to a Name (line 406):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_91845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 4), 'int')
    # Processing the call keyword arguments
    kwargs_91846 = {}
    # Getting the type of 'call_assignment_90855' (line 406)
    call_assignment_90855_91843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'call_assignment_90855', False)
    # Obtaining the member '__getitem__' of a type (line 406)
    getitem___91844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 4), call_assignment_90855_91843, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_91847 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___91844, *[int_91845], **kwargs_91846)
    
    # Assigning a type to the variable 'call_assignment_90856' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'call_assignment_90856', getitem___call_result_91847)
    
    # Assigning a Name to a Name (line 406):
    # Getting the type of 'call_assignment_90856' (line 406)
    call_assignment_90856_91848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'call_assignment_90856')
    # Assigning a type to the variable 'files' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'files', call_assignment_90856_91848)
    
    # Assigning a Call to a Name (line 406):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_91851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 4), 'int')
    # Processing the call keyword arguments
    kwargs_91852 = {}
    # Getting the type of 'call_assignment_90855' (line 406)
    call_assignment_90855_91849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'call_assignment_90855', False)
    # Obtaining the member '__getitem__' of a type (line 406)
    getitem___91850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 4), call_assignment_90855_91849, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_91853 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___91850, *[int_91851], **kwargs_91852)
    
    # Assigning a type to the variable 'call_assignment_90857' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'call_assignment_90857', getitem___call_result_91853)
    
    # Assigning a Name to a Name (line 406):
    # Getting the type of 'call_assignment_90857' (line 406)
    call_assignment_90857_91854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'call_assignment_90857')
    # Assigning a type to the variable 'options' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 11), 'options', call_assignment_90857_91854)
    
    # Assigning a Name to a Attribute (line 407):
    
    # Assigning a Name to a Attribute (line 407):
    # Getting the type of 'options' (line 407)
    options_91855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 23), 'options')
    # Getting the type of 'auxfuncs' (line 407)
    auxfuncs_91856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'auxfuncs')
    # Setting the type of the member 'options' of a type (line 407)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 4), auxfuncs_91856, 'options', options_91855)
    
    # Assigning a Call to a Name (line 408):
    
    # Assigning a Call to a Name (line 408):
    
    # Call to callcrackfortran(...): (line 408)
    # Processing the call arguments (line 408)
    # Getting the type of 'files' (line 408)
    files_91858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 32), 'files', False)
    # Getting the type of 'options' (line 408)
    options_91859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 39), 'options', False)
    # Processing the call keyword arguments (line 408)
    kwargs_91860 = {}
    # Getting the type of 'callcrackfortran' (line 408)
    callcrackfortran_91857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 15), 'callcrackfortran', False)
    # Calling callcrackfortran(args, kwargs) (line 408)
    callcrackfortran_call_result_91861 = invoke(stypy.reporting.localization.Localization(__file__, 408, 15), callcrackfortran_91857, *[files_91858, options_91859], **kwargs_91860)
    
    # Assigning a type to the variable 'postlist' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'postlist', callcrackfortran_call_result_91861)
    
    # Assigning a Dict to a Name (line 409):
    
    # Assigning a Dict to a Name (line 409):
    
    # Obtaining an instance of the builtin type 'dict' (line 409)
    dict_91862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 409)
    
    # Assigning a type to the variable 'isusedby' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'isusedby', dict_91862)
    
    
    # Call to range(...): (line 410)
    # Processing the call arguments (line 410)
    
    # Call to len(...): (line 410)
    # Processing the call arguments (line 410)
    # Getting the type of 'postlist' (line 410)
    postlist_91865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 23), 'postlist', False)
    # Processing the call keyword arguments (line 410)
    kwargs_91866 = {}
    # Getting the type of 'len' (line 410)
    len_91864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 19), 'len', False)
    # Calling len(args, kwargs) (line 410)
    len_call_result_91867 = invoke(stypy.reporting.localization.Localization(__file__, 410, 19), len_91864, *[postlist_91865], **kwargs_91866)
    
    # Processing the call keyword arguments (line 410)
    kwargs_91868 = {}
    # Getting the type of 'range' (line 410)
    range_91863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 13), 'range', False)
    # Calling range(args, kwargs) (line 410)
    range_call_result_91869 = invoke(stypy.reporting.localization.Localization(__file__, 410, 13), range_91863, *[len_call_result_91867], **kwargs_91868)
    
    # Testing the type of a for loop iterable (line 410)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 410, 4), range_call_result_91869)
    # Getting the type of the for loop variable (line 410)
    for_loop_var_91870 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 410, 4), range_call_result_91869)
    # Assigning a type to the variable 'i' (line 410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'i', for_loop_var_91870)
    # SSA begins for a for statement (line 410)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    str_91871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 11), 'str', 'use')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 411)
    i_91872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 29), 'i')
    # Getting the type of 'postlist' (line 411)
    postlist_91873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 20), 'postlist')
    # Obtaining the member '__getitem__' of a type (line 411)
    getitem___91874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 20), postlist_91873, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 411)
    subscript_call_result_91875 = invoke(stypy.reporting.localization.Localization(__file__, 411, 20), getitem___91874, i_91872)
    
    # Applying the binary operator 'in' (line 411)
    result_contains_91876 = python_operator(stypy.reporting.localization.Localization(__file__, 411, 11), 'in', str_91871, subscript_call_result_91875)
    
    # Testing the type of an if condition (line 411)
    if_condition_91877 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 411, 8), result_contains_91876)
    # Assigning a type to the variable 'if_condition_91877' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'if_condition_91877', if_condition_91877)
    # SSA begins for if statement (line 411)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to keys(...): (line 412)
    # Processing the call keyword arguments (line 412)
    kwargs_91886 = {}
    
    # Obtaining the type of the subscript
    str_91878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 33), 'str', 'use')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 412)
    i_91879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 30), 'i', False)
    # Getting the type of 'postlist' (line 412)
    postlist_91880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 21), 'postlist', False)
    # Obtaining the member '__getitem__' of a type (line 412)
    getitem___91881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 21), postlist_91880, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 412)
    subscript_call_result_91882 = invoke(stypy.reporting.localization.Localization(__file__, 412, 21), getitem___91881, i_91879)
    
    # Obtaining the member '__getitem__' of a type (line 412)
    getitem___91883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 21), subscript_call_result_91882, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 412)
    subscript_call_result_91884 = invoke(stypy.reporting.localization.Localization(__file__, 412, 21), getitem___91883, str_91878)
    
    # Obtaining the member 'keys' of a type (line 412)
    keys_91885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 21), subscript_call_result_91884, 'keys')
    # Calling keys(args, kwargs) (line 412)
    keys_call_result_91887 = invoke(stypy.reporting.localization.Localization(__file__, 412, 21), keys_91885, *[], **kwargs_91886)
    
    # Testing the type of a for loop iterable (line 412)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 412, 12), keys_call_result_91887)
    # Getting the type of the for loop variable (line 412)
    for_loop_var_91888 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 412, 12), keys_call_result_91887)
    # Assigning a type to the variable 'u' (line 412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 12), 'u', for_loop_var_91888)
    # SSA begins for a for statement (line 412)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'u' (line 413)
    u_91889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 19), 'u')
    # Getting the type of 'isusedby' (line 413)
    isusedby_91890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 28), 'isusedby')
    # Applying the binary operator 'notin' (line 413)
    result_contains_91891 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 19), 'notin', u_91889, isusedby_91890)
    
    # Testing the type of an if condition (line 413)
    if_condition_91892 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 413, 16), result_contains_91891)
    # Assigning a type to the variable 'if_condition_91892' (line 413)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 16), 'if_condition_91892', if_condition_91892)
    # SSA begins for if statement (line 413)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Subscript (line 414):
    
    # Assigning a List to a Subscript (line 414):
    
    # Obtaining an instance of the builtin type 'list' (line 414)
    list_91893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 414)
    
    # Getting the type of 'isusedby' (line 414)
    isusedby_91894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 20), 'isusedby')
    # Getting the type of 'u' (line 414)
    u_91895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 29), 'u')
    # Storing an element on a container (line 414)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 20), isusedby_91894, (u_91895, list_91893))
    # SSA join for if statement (line 413)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 415)
    # Processing the call arguments (line 415)
    
    # Obtaining the type of the subscript
    str_91901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 47), 'str', 'name')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 415)
    i_91902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 44), 'i', False)
    # Getting the type of 'postlist' (line 415)
    postlist_91903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 35), 'postlist', False)
    # Obtaining the member '__getitem__' of a type (line 415)
    getitem___91904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 35), postlist_91903, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 415)
    subscript_call_result_91905 = invoke(stypy.reporting.localization.Localization(__file__, 415, 35), getitem___91904, i_91902)
    
    # Obtaining the member '__getitem__' of a type (line 415)
    getitem___91906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 35), subscript_call_result_91905, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 415)
    subscript_call_result_91907 = invoke(stypy.reporting.localization.Localization(__file__, 415, 35), getitem___91906, str_91901)
    
    # Processing the call keyword arguments (line 415)
    kwargs_91908 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'u' (line 415)
    u_91896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 25), 'u', False)
    # Getting the type of 'isusedby' (line 415)
    isusedby_91897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 16), 'isusedby', False)
    # Obtaining the member '__getitem__' of a type (line 415)
    getitem___91898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 16), isusedby_91897, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 415)
    subscript_call_result_91899 = invoke(stypy.reporting.localization.Localization(__file__, 415, 16), getitem___91898, u_91896)
    
    # Obtaining the member 'append' of a type (line 415)
    append_91900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 16), subscript_call_result_91899, 'append')
    # Calling append(args, kwargs) (line 415)
    append_call_result_91909 = invoke(stypy.reporting.localization.Localization(__file__, 415, 16), append_91900, *[subscript_call_result_91907], **kwargs_91908)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 411)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 416)
    # Processing the call arguments (line 416)
    
    # Call to len(...): (line 416)
    # Processing the call arguments (line 416)
    # Getting the type of 'postlist' (line 416)
    postlist_91912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 23), 'postlist', False)
    # Processing the call keyword arguments (line 416)
    kwargs_91913 = {}
    # Getting the type of 'len' (line 416)
    len_91911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 19), 'len', False)
    # Calling len(args, kwargs) (line 416)
    len_call_result_91914 = invoke(stypy.reporting.localization.Localization(__file__, 416, 19), len_91911, *[postlist_91912], **kwargs_91913)
    
    # Processing the call keyword arguments (line 416)
    kwargs_91915 = {}
    # Getting the type of 'range' (line 416)
    range_91910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 13), 'range', False)
    # Calling range(args, kwargs) (line 416)
    range_call_result_91916 = invoke(stypy.reporting.localization.Localization(__file__, 416, 13), range_91910, *[len_call_result_91914], **kwargs_91915)
    
    # Testing the type of a for loop iterable (line 416)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 416, 4), range_call_result_91916)
    # Getting the type of the for loop variable (line 416)
    for_loop_var_91917 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 416, 4), range_call_result_91916)
    # Assigning a type to the variable 'i' (line 416)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'i', for_loop_var_91917)
    # SSA begins for a for statement (line 416)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    str_91918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 23), 'str', 'block')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 417)
    i_91919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 20), 'i')
    # Getting the type of 'postlist' (line 417)
    postlist_91920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 11), 'postlist')
    # Obtaining the member '__getitem__' of a type (line 417)
    getitem___91921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 11), postlist_91920, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 417)
    subscript_call_result_91922 = invoke(stypy.reporting.localization.Localization(__file__, 417, 11), getitem___91921, i_91919)
    
    # Obtaining the member '__getitem__' of a type (line 417)
    getitem___91923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 11), subscript_call_result_91922, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 417)
    subscript_call_result_91924 = invoke(stypy.reporting.localization.Localization(__file__, 417, 11), getitem___91923, str_91918)
    
    str_91925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 35), 'str', 'python module')
    # Applying the binary operator '==' (line 417)
    result_eq_91926 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 11), '==', subscript_call_result_91924, str_91925)
    
    
    str_91927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 55), 'str', '__user__')
    
    # Obtaining the type of the subscript
    str_91928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 81), 'str', 'name')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 417)
    i_91929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 78), 'i')
    # Getting the type of 'postlist' (line 417)
    postlist_91930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 69), 'postlist')
    # Obtaining the member '__getitem__' of a type (line 417)
    getitem___91931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 69), postlist_91930, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 417)
    subscript_call_result_91932 = invoke(stypy.reporting.localization.Localization(__file__, 417, 69), getitem___91931, i_91929)
    
    # Obtaining the member '__getitem__' of a type (line 417)
    getitem___91933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 69), subscript_call_result_91932, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 417)
    subscript_call_result_91934 = invoke(stypy.reporting.localization.Localization(__file__, 417, 69), getitem___91933, str_91928)
    
    # Applying the binary operator 'in' (line 417)
    result_contains_91935 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 55), 'in', str_91927, subscript_call_result_91934)
    
    # Applying the binary operator 'and' (line 417)
    result_and_keyword_91936 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 11), 'and', result_eq_91926, result_contains_91935)
    
    # Testing the type of an if condition (line 417)
    if_condition_91937 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 417, 8), result_and_keyword_91936)
    # Assigning a type to the variable 'if_condition_91937' (line 417)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'if_condition_91937', if_condition_91937)
    # SSA begins for if statement (line 417)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    str_91938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 27), 'str', 'name')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 418)
    i_91939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 24), 'i')
    # Getting the type of 'postlist' (line 418)
    postlist_91940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 15), 'postlist')
    # Obtaining the member '__getitem__' of a type (line 418)
    getitem___91941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 15), postlist_91940, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 418)
    subscript_call_result_91942 = invoke(stypy.reporting.localization.Localization(__file__, 418, 15), getitem___91941, i_91939)
    
    # Obtaining the member '__getitem__' of a type (line 418)
    getitem___91943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 15), subscript_call_result_91942, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 418)
    subscript_call_result_91944 = invoke(stypy.reporting.localization.Localization(__file__, 418, 15), getitem___91943, str_91938)
    
    # Getting the type of 'isusedby' (line 418)
    isusedby_91945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 38), 'isusedby')
    # Applying the binary operator 'in' (line 418)
    result_contains_91946 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 15), 'in', subscript_call_result_91944, isusedby_91945)
    
    # Testing the type of an if condition (line 418)
    if_condition_91947 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 418, 12), result_contains_91946)
    # Assigning a type to the variable 'if_condition_91947' (line 418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 12), 'if_condition_91947', if_condition_91947)
    # SSA begins for if statement (line 418)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to outmess(...): (line 420)
    # Processing the call arguments (line 420)
    str_91949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 24), 'str', 'Skipping Makefile build for module "%s" which is used by %s\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 421)
    tuple_91950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 421)
    # Adding element type (line 421)
    
    # Obtaining the type of the subscript
    str_91951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 32), 'str', 'name')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 421)
    i_91952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 29), 'i', False)
    # Getting the type of 'postlist' (line 421)
    postlist_91953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 20), 'postlist', False)
    # Obtaining the member '__getitem__' of a type (line 421)
    getitem___91954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 20), postlist_91953, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 421)
    subscript_call_result_91955 = invoke(stypy.reporting.localization.Localization(__file__, 421, 20), getitem___91954, i_91952)
    
    # Obtaining the member '__getitem__' of a type (line 421)
    getitem___91956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 20), subscript_call_result_91955, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 421)
    subscript_call_result_91957 = invoke(stypy.reporting.localization.Localization(__file__, 421, 20), getitem___91956, str_91951)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 20), tuple_91950, subscript_call_result_91957)
    # Adding element type (line 421)
    
    # Call to join(...): (line 421)
    # Processing the call arguments (line 421)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    str_91963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 92), 'str', 'name')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 421)
    i_91964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 89), 'i', False)
    # Getting the type of 'postlist' (line 421)
    postlist_91965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 80), 'postlist', False)
    # Obtaining the member '__getitem__' of a type (line 421)
    getitem___91966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 80), postlist_91965, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 421)
    subscript_call_result_91967 = invoke(stypy.reporting.localization.Localization(__file__, 421, 80), getitem___91966, i_91964)
    
    # Obtaining the member '__getitem__' of a type (line 421)
    getitem___91968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 80), subscript_call_result_91967, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 421)
    subscript_call_result_91969 = invoke(stypy.reporting.localization.Localization(__file__, 421, 80), getitem___91968, str_91963)
    
    # Getting the type of 'isusedby' (line 421)
    isusedby_91970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 71), 'isusedby', False)
    # Obtaining the member '__getitem__' of a type (line 421)
    getitem___91971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 71), isusedby_91970, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 421)
    subscript_call_result_91972 = invoke(stypy.reporting.localization.Localization(__file__, 421, 71), getitem___91971, subscript_call_result_91969)
    
    comprehension_91973 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 51), subscript_call_result_91972)
    # Assigning a type to the variable 's' (line 421)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 51), 's', comprehension_91973)
    str_91960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 51), 'str', '"%s"')
    # Getting the type of 's' (line 421)
    s_91961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 60), 's', False)
    # Applying the binary operator '%' (line 421)
    result_mod_91962 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 51), '%', str_91960, s_91961)
    
    list_91974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 51), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 51), list_91974, result_mod_91962)
    # Processing the call keyword arguments (line 421)
    kwargs_91975 = {}
    str_91958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 41), 'str', ',')
    # Obtaining the member 'join' of a type (line 421)
    join_91959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 41), str_91958, 'join')
    # Calling join(args, kwargs) (line 421)
    join_call_result_91976 = invoke(stypy.reporting.localization.Localization(__file__, 421, 41), join_91959, *[list_91974], **kwargs_91975)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 20), tuple_91950, join_call_result_91976)
    
    # Applying the binary operator '%' (line 420)
    result_mod_91977 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 24), '%', str_91949, tuple_91950)
    
    # Processing the call keyword arguments (line 420)
    kwargs_91978 = {}
    # Getting the type of 'outmess' (line 420)
    outmess_91948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 16), 'outmess', False)
    # Calling outmess(args, kwargs) (line 420)
    outmess_call_result_91979 = invoke(stypy.reporting.localization.Localization(__file__, 420, 16), outmess_91948, *[result_mod_91977], **kwargs_91978)
    
    # SSA join for if statement (line 418)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 417)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_91980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 7), 'str', 'signsfile')
    # Getting the type of 'options' (line 422)
    options_91981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 22), 'options')
    # Applying the binary operator 'in' (line 422)
    result_contains_91982 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 7), 'in', str_91980, options_91981)
    
    # Testing the type of an if condition (line 422)
    if_condition_91983 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 422, 4), result_contains_91982)
    # Assigning a type to the variable 'if_condition_91983' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'if_condition_91983', if_condition_91983)
    # SSA begins for if statement (line 422)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    str_91984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 19), 'str', 'verbose')
    # Getting the type of 'options' (line 423)
    options_91985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 11), 'options')
    # Obtaining the member '__getitem__' of a type (line 423)
    getitem___91986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 11), options_91985, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 423)
    subscript_call_result_91987 = invoke(stypy.reporting.localization.Localization(__file__, 423, 11), getitem___91986, str_91984)
    
    int_91988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 32), 'int')
    # Applying the binary operator '>' (line 423)
    result_gt_91989 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 11), '>', subscript_call_result_91987, int_91988)
    
    # Testing the type of an if condition (line 423)
    if_condition_91990 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 423, 8), result_gt_91989)
    # Assigning a type to the variable 'if_condition_91990' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'if_condition_91990', if_condition_91990)
    # SSA begins for if statement (line 423)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to outmess(...): (line 424)
    # Processing the call arguments (line 424)
    str_91992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 16), 'str', 'Stopping. Edit the signature file and then run f2py on the signature file: ')
    # Processing the call keyword arguments (line 424)
    kwargs_91993 = {}
    # Getting the type of 'outmess' (line 424)
    outmess_91991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'outmess', False)
    # Calling outmess(args, kwargs) (line 424)
    outmess_call_result_91994 = invoke(stypy.reporting.localization.Localization(__file__, 424, 12), outmess_91991, *[str_91992], **kwargs_91993)
    
    
    # Call to outmess(...): (line 426)
    # Processing the call arguments (line 426)
    str_91996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 20), 'str', '%s %s\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 427)
    tuple_91997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 427)
    # Adding element type (line 427)
    
    # Call to basename(...): (line 427)
    # Processing the call arguments (line 427)
    
    # Obtaining the type of the subscript
    int_92001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 47), 'int')
    # Getting the type of 'sys' (line 427)
    sys_92002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 38), 'sys', False)
    # Obtaining the member 'argv' of a type (line 427)
    argv_92003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 38), sys_92002, 'argv')
    # Obtaining the member '__getitem__' of a type (line 427)
    getitem___92004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 38), argv_92003, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 427)
    subscript_call_result_92005 = invoke(stypy.reporting.localization.Localization(__file__, 427, 38), getitem___92004, int_92001)
    
    # Processing the call keyword arguments (line 427)
    kwargs_92006 = {}
    # Getting the type of 'os' (line 427)
    os_91998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 21), 'os', False)
    # Obtaining the member 'path' of a type (line 427)
    path_91999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 21), os_91998, 'path')
    # Obtaining the member 'basename' of a type (line 427)
    basename_92000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 21), path_91999, 'basename')
    # Calling basename(args, kwargs) (line 427)
    basename_call_result_92007 = invoke(stypy.reporting.localization.Localization(__file__, 427, 21), basename_92000, *[subscript_call_result_92005], **kwargs_92006)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 21), tuple_91997, basename_call_result_92007)
    # Adding element type (line 427)
    
    # Obtaining the type of the subscript
    str_92008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 60), 'str', 'signsfile')
    # Getting the type of 'options' (line 427)
    options_92009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 52), 'options', False)
    # Obtaining the member '__getitem__' of a type (line 427)
    getitem___92010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 52), options_92009, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 427)
    subscript_call_result_92011 = invoke(stypy.reporting.localization.Localization(__file__, 427, 52), getitem___92010, str_92008)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 21), tuple_91997, subscript_call_result_92011)
    
    # Applying the binary operator '%' (line 426)
    result_mod_92012 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 20), '%', str_91996, tuple_91997)
    
    # Processing the call keyword arguments (line 426)
    kwargs_92013 = {}
    # Getting the type of 'outmess' (line 426)
    outmess_91995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'outmess', False)
    # Calling outmess(args, kwargs) (line 426)
    outmess_call_result_92014 = invoke(stypy.reporting.localization.Localization(__file__, 426, 12), outmess_91995, *[result_mod_92012], **kwargs_92013)
    
    # SSA join for if statement (line 423)
    module_type_store = module_type_store.join_ssa_context()
    
    # Assigning a type to the variable 'stypy_return_type' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 422)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 429)
    # Processing the call arguments (line 429)
    
    # Call to len(...): (line 429)
    # Processing the call arguments (line 429)
    # Getting the type of 'postlist' (line 429)
    postlist_92017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 23), 'postlist', False)
    # Processing the call keyword arguments (line 429)
    kwargs_92018 = {}
    # Getting the type of 'len' (line 429)
    len_92016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 19), 'len', False)
    # Calling len(args, kwargs) (line 429)
    len_call_result_92019 = invoke(stypy.reporting.localization.Localization(__file__, 429, 19), len_92016, *[postlist_92017], **kwargs_92018)
    
    # Processing the call keyword arguments (line 429)
    kwargs_92020 = {}
    # Getting the type of 'range' (line 429)
    range_92015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 13), 'range', False)
    # Calling range(args, kwargs) (line 429)
    range_call_result_92021 = invoke(stypy.reporting.localization.Localization(__file__, 429, 13), range_92015, *[len_call_result_92019], **kwargs_92020)
    
    # Testing the type of a for loop iterable (line 429)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 429, 4), range_call_result_92021)
    # Getting the type of the for loop variable (line 429)
    for_loop_var_92022 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 429, 4), range_call_result_92021)
    # Assigning a type to the variable 'i' (line 429)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 4), 'i', for_loop_var_92022)
    # SSA begins for a for statement (line 429)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    str_92023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 23), 'str', 'block')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 430)
    i_92024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 20), 'i')
    # Getting the type of 'postlist' (line 430)
    postlist_92025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 11), 'postlist')
    # Obtaining the member '__getitem__' of a type (line 430)
    getitem___92026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 11), postlist_92025, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 430)
    subscript_call_result_92027 = invoke(stypy.reporting.localization.Localization(__file__, 430, 11), getitem___92026, i_92024)
    
    # Obtaining the member '__getitem__' of a type (line 430)
    getitem___92028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 11), subscript_call_result_92027, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 430)
    subscript_call_result_92029 = invoke(stypy.reporting.localization.Localization(__file__, 430, 11), getitem___92028, str_92023)
    
    str_92030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 35), 'str', 'python module')
    # Applying the binary operator '!=' (line 430)
    result_ne_92031 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 11), '!=', subscript_call_result_92029, str_92030)
    
    # Testing the type of an if condition (line 430)
    if_condition_92032 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 430, 8), result_ne_92031)
    # Assigning a type to the variable 'if_condition_92032' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'if_condition_92032', if_condition_92032)
    # SSA begins for if statement (line 430)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    str_92033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 15), 'str', 'python module')
    # Getting the type of 'options' (line 431)
    options_92034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 38), 'options')
    # Applying the binary operator 'notin' (line 431)
    result_contains_92035 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 15), 'notin', str_92033, options_92034)
    
    # Testing the type of an if condition (line 431)
    if_condition_92036 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 431, 12), result_contains_92035)
    # Assigning a type to the variable 'if_condition_92036' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 12), 'if_condition_92036', if_condition_92036)
    # SSA begins for if statement (line 431)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to errmess(...): (line 432)
    # Processing the call arguments (line 432)
    str_92038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 20), 'str', 'Tip: If your original code is Fortran source then you must use -m option.\n')
    # Processing the call keyword arguments (line 432)
    kwargs_92039 = {}
    # Getting the type of 'errmess' (line 432)
    errmess_92037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 16), 'errmess', False)
    # Calling errmess(args, kwargs) (line 432)
    errmess_call_result_92040 = invoke(stypy.reporting.localization.Localization(__file__, 432, 16), errmess_92037, *[str_92038], **kwargs_92039)
    
    # SSA join for if statement (line 431)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to TypeError(...): (line 434)
    # Processing the call arguments (line 434)
    str_92042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 28), 'str', 'All blocks must be python module blocks but got %s')
    
    # Call to repr(...): (line 435)
    # Processing the call arguments (line 435)
    
    # Obtaining the type of the subscript
    str_92044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 33), 'str', 'block')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 435)
    i_92045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 30), 'i', False)
    # Getting the type of 'postlist' (line 435)
    postlist_92046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 21), 'postlist', False)
    # Obtaining the member '__getitem__' of a type (line 435)
    getitem___92047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 21), postlist_92046, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 435)
    subscript_call_result_92048 = invoke(stypy.reporting.localization.Localization(__file__, 435, 21), getitem___92047, i_92045)
    
    # Obtaining the member '__getitem__' of a type (line 435)
    getitem___92049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 21), subscript_call_result_92048, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 435)
    subscript_call_result_92050 = invoke(stypy.reporting.localization.Localization(__file__, 435, 21), getitem___92049, str_92044)
    
    # Processing the call keyword arguments (line 435)
    kwargs_92051 = {}
    # Getting the type of 'repr' (line 435)
    repr_92043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 16), 'repr', False)
    # Calling repr(args, kwargs) (line 435)
    repr_call_result_92052 = invoke(stypy.reporting.localization.Localization(__file__, 435, 16), repr_92043, *[subscript_call_result_92050], **kwargs_92051)
    
    # Applying the binary operator '%' (line 434)
    result_mod_92053 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 28), '%', str_92042, repr_call_result_92052)
    
    # Processing the call keyword arguments (line 434)
    kwargs_92054 = {}
    # Getting the type of 'TypeError' (line 434)
    TypeError_92041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 18), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 434)
    TypeError_call_result_92055 = invoke(stypy.reporting.localization.Localization(__file__, 434, 18), TypeError_92041, *[result_mod_92053], **kwargs_92054)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 434, 12), TypeError_call_result_92055, 'raise parameter', BaseException)
    # SSA join for if statement (line 430)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Attribute (line 436):
    
    # Assigning a Subscript to a Attribute (line 436):
    
    # Obtaining the type of the subscript
    str_92056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 36), 'str', 'debug')
    # Getting the type of 'options' (line 436)
    options_92057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 28), 'options')
    # Obtaining the member '__getitem__' of a type (line 436)
    getitem___92058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 28), options_92057, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 436)
    subscript_call_result_92059 = invoke(stypy.reporting.localization.Localization(__file__, 436, 28), getitem___92058, str_92056)
    
    # Getting the type of 'auxfuncs' (line 436)
    auxfuncs_92060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'auxfuncs')
    # Setting the type of the member 'debugoptions' of a type (line 436)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 4), auxfuncs_92060, 'debugoptions', subscript_call_result_92059)
    
    # Assigning a Name to a Attribute (line 437):
    
    # Assigning a Name to a Attribute (line 437):
    # Getting the type of 'options' (line 437)
    options_92061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 27), 'options')
    # Getting the type of 'f90mod_rules' (line 437)
    f90mod_rules_92062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 4), 'f90mod_rules')
    # Setting the type of the member 'options' of a type (line 437)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 4), f90mod_rules_92062, 'options', options_92061)
    
    # Assigning a Subscript to a Attribute (line 438):
    
    # Assigning a Subscript to a Attribute (line 438):
    
    # Obtaining the type of the subscript
    str_92063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 33), 'str', 'wrapfuncs')
    # Getting the type of 'options' (line 438)
    options_92064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 25), 'options')
    # Obtaining the member '__getitem__' of a type (line 438)
    getitem___92065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 25), options_92064, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 438)
    subscript_call_result_92066 = invoke(stypy.reporting.localization.Localization(__file__, 438, 25), getitem___92065, str_92063)
    
    # Getting the type of 'auxfuncs' (line 438)
    auxfuncs_92067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 4), 'auxfuncs')
    # Setting the type of the member 'wrapfuncs' of a type (line 438)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 4), auxfuncs_92067, 'wrapfuncs', subscript_call_result_92066)
    
    # Assigning a Call to a Name (line 440):
    
    # Assigning a Call to a Name (line 440):
    
    # Call to buildmodules(...): (line 440)
    # Processing the call arguments (line 440)
    # Getting the type of 'postlist' (line 440)
    postlist_92069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 23), 'postlist', False)
    # Processing the call keyword arguments (line 440)
    kwargs_92070 = {}
    # Getting the type of 'buildmodules' (line 440)
    buildmodules_92068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 10), 'buildmodules', False)
    # Calling buildmodules(args, kwargs) (line 440)
    buildmodules_call_result_92071 = invoke(stypy.reporting.localization.Localization(__file__, 440, 10), buildmodules_92068, *[postlist_92069], **kwargs_92070)
    
    # Assigning a type to the variable 'ret' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'ret', buildmodules_call_result_92071)
    
    
    # Call to keys(...): (line 442)
    # Processing the call keyword arguments (line 442)
    kwargs_92074 = {}
    # Getting the type of 'ret' (line 442)
    ret_92072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 14), 'ret', False)
    # Obtaining the member 'keys' of a type (line 442)
    keys_92073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 14), ret_92072, 'keys')
    # Calling keys(args, kwargs) (line 442)
    keys_call_result_92075 = invoke(stypy.reporting.localization.Localization(__file__, 442, 14), keys_92073, *[], **kwargs_92074)
    
    # Testing the type of a for loop iterable (line 442)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 442, 4), keys_call_result_92075)
    # Getting the type of the for loop variable (line 442)
    for_loop_var_92076 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 442, 4), keys_call_result_92075)
    # Assigning a type to the variable 'mn' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'mn', for_loop_var_92076)
    # SSA begins for a for statement (line 442)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to dict_append(...): (line 443)
    # Processing the call arguments (line 443)
    
    # Obtaining the type of the subscript
    # Getting the type of 'mn' (line 443)
    mn_92078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 24), 'mn', False)
    # Getting the type of 'ret' (line 443)
    ret_92079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 20), 'ret', False)
    # Obtaining the member '__getitem__' of a type (line 443)
    getitem___92080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 20), ret_92079, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 443)
    subscript_call_result_92081 = invoke(stypy.reporting.localization.Localization(__file__, 443, 20), getitem___92080, mn_92078)
    
    
    # Obtaining an instance of the builtin type 'dict' (line 443)
    dict_92082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 29), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 443)
    # Adding element type (key, value) (line 443)
    str_92083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 30), 'str', 'csrc')
    # Getting the type of 'fobjcsrc' (line 443)
    fobjcsrc_92084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 38), 'fobjcsrc', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 29), dict_92082, (str_92083, fobjcsrc_92084))
    # Adding element type (key, value) (line 443)
    str_92085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 48), 'str', 'h')
    # Getting the type of 'fobjhsrc' (line 443)
    fobjhsrc_92086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 53), 'fobjhsrc', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 29), dict_92082, (str_92085, fobjhsrc_92086))
    
    # Processing the call keyword arguments (line 443)
    kwargs_92087 = {}
    # Getting the type of 'dict_append' (line 443)
    dict_append_92077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'dict_append', False)
    # Calling dict_append(args, kwargs) (line 443)
    dict_append_call_result_92088 = invoke(stypy.reporting.localization.Localization(__file__, 443, 8), dict_append_92077, *[subscript_call_result_92081, dict_92082], **kwargs_92087)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ret' (line 444)
    ret_92089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 11), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'stypy_return_type', ret_92089)
    
    # ################# End of 'run_main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run_main' in the type store
    # Getting the type of 'stypy_return_type' (line 398)
    stypy_return_type_92090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_92090)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run_main'
    return stypy_return_type_92090

# Assigning a type to the variable 'run_main' (line 398)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 0), 'run_main', run_main)

@norecursion
def filter_files(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 447)
    None_92091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 54), 'None')
    defaults = [None_92091]
    # Create a new context for function 'filter_files'
    module_type_store = module_type_store.open_function_context('filter_files', 447, 0, False)
    
    # Passed parameters checking function
    filter_files.stypy_localization = localization
    filter_files.stypy_type_of_self = None
    filter_files.stypy_type_store = module_type_store
    filter_files.stypy_function_name = 'filter_files'
    filter_files.stypy_param_names_list = ['prefix', 'suffix', 'files', 'remove_prefix']
    filter_files.stypy_varargs_param_name = None
    filter_files.stypy_kwargs_param_name = None
    filter_files.stypy_call_defaults = defaults
    filter_files.stypy_call_varargs = varargs
    filter_files.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'filter_files', ['prefix', 'suffix', 'files', 'remove_prefix'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'filter_files', localization, ['prefix', 'suffix', 'files', 'remove_prefix'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'filter_files(...)' code ##################

    str_92092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, (-1)), 'str', '\n    Filter files by prefix and suffix.\n    ')
    
    # Assigning a Tuple to a Tuple (line 451):
    
    # Assigning a List to a Name (line 451):
    
    # Obtaining an instance of the builtin type 'list' (line 451)
    list_92093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 451)
    
    # Assigning a type to the variable 'tuple_assignment_90858' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'tuple_assignment_90858', list_92093)
    
    # Assigning a List to a Name (line 451):
    
    # Obtaining an instance of the builtin type 'list' (line 451)
    list_92094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 451)
    
    # Assigning a type to the variable 'tuple_assignment_90859' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'tuple_assignment_90859', list_92094)
    
    # Assigning a Name to a Name (line 451):
    # Getting the type of 'tuple_assignment_90858' (line 451)
    tuple_assignment_90858_92095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'tuple_assignment_90858')
    # Assigning a type to the variable 'filtered' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'filtered', tuple_assignment_90858_92095)
    
    # Assigning a Name to a Name (line 451):
    # Getting the type of 'tuple_assignment_90859' (line 451)
    tuple_assignment_90859_92096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'tuple_assignment_90859')
    # Assigning a type to the variable 'rest' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 14), 'rest', tuple_assignment_90859_92096)
    
    # Assigning a Attribute to a Name (line 452):
    
    # Assigning a Attribute to a Name (line 452):
    
    # Call to compile(...): (line 452)
    # Processing the call arguments (line 452)
    # Getting the type of 'prefix' (line 452)
    prefix_92099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 23), 'prefix', False)
    str_92100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 32), 'str', '.*')
    # Applying the binary operator '+' (line 452)
    result_add_92101 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 23), '+', prefix_92099, str_92100)
    
    # Getting the type of 'suffix' (line 452)
    suffix_92102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 40), 'suffix', False)
    # Applying the binary operator '+' (line 452)
    result_add_92103 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 38), '+', result_add_92101, suffix_92102)
    
    str_92104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 49), 'str', '\\Z')
    # Applying the binary operator '+' (line 452)
    result_add_92105 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 47), '+', result_add_92103, str_92104)
    
    # Processing the call keyword arguments (line 452)
    kwargs_92106 = {}
    # Getting the type of 're' (line 452)
    re_92097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 12), 're', False)
    # Obtaining the member 'compile' of a type (line 452)
    compile_92098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 12), re_92097, 'compile')
    # Calling compile(args, kwargs) (line 452)
    compile_call_result_92107 = invoke(stypy.reporting.localization.Localization(__file__, 452, 12), compile_92098, *[result_add_92105], **kwargs_92106)
    
    # Obtaining the member 'match' of a type (line 452)
    match_92108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 12), compile_call_result_92107, 'match')
    # Assigning a type to the variable 'match' (line 452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'match', match_92108)
    
    # Getting the type of 'remove_prefix' (line 453)
    remove_prefix_92109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 7), 'remove_prefix')
    # Testing the type of an if condition (line 453)
    if_condition_92110 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 453, 4), remove_prefix_92109)
    # Assigning a type to the variable 'if_condition_92110' (line 453)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'if_condition_92110', if_condition_92110)
    # SSA begins for if statement (line 453)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 454):
    
    # Assigning a Call to a Name (line 454):
    
    # Call to len(...): (line 454)
    # Processing the call arguments (line 454)
    # Getting the type of 'prefix' (line 454)
    prefix_92112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 18), 'prefix', False)
    # Processing the call keyword arguments (line 454)
    kwargs_92113 = {}
    # Getting the type of 'len' (line 454)
    len_92111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 14), 'len', False)
    # Calling len(args, kwargs) (line 454)
    len_call_result_92114 = invoke(stypy.reporting.localization.Localization(__file__, 454, 14), len_92111, *[prefix_92112], **kwargs_92113)
    
    # Assigning a type to the variable 'ind' (line 454)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'ind', len_call_result_92114)
    # SSA branch for the else part of an if statement (line 453)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 456):
    
    # Assigning a Num to a Name (line 456):
    int_92115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 14), 'int')
    # Assigning a type to the variable 'ind' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'ind', int_92115)
    # SSA join for if statement (line 453)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'files' (line 457)
    files_92120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 36), 'files')
    comprehension_92121 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 17), files_92120)
    # Assigning a type to the variable 'x' (line 457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 17), 'x', comprehension_92121)
    
    # Call to strip(...): (line 457)
    # Processing the call keyword arguments (line 457)
    kwargs_92118 = {}
    # Getting the type of 'x' (line 457)
    x_92116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 17), 'x', False)
    # Obtaining the member 'strip' of a type (line 457)
    strip_92117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 17), x_92116, 'strip')
    # Calling strip(args, kwargs) (line 457)
    strip_call_result_92119 = invoke(stypy.reporting.localization.Localization(__file__, 457, 17), strip_92117, *[], **kwargs_92118)
    
    list_92122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 17), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 17), list_92122, strip_call_result_92119)
    # Testing the type of a for loop iterable (line 457)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 457, 4), list_92122)
    # Getting the type of the for loop variable (line 457)
    for_loop_var_92123 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 457, 4), list_92122)
    # Assigning a type to the variable 'file' (line 457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 4), 'file', for_loop_var_92123)
    # SSA begins for a for statement (line 457)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to match(...): (line 458)
    # Processing the call arguments (line 458)
    # Getting the type of 'file' (line 458)
    file_92125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 17), 'file', False)
    # Processing the call keyword arguments (line 458)
    kwargs_92126 = {}
    # Getting the type of 'match' (line 458)
    match_92124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 11), 'match', False)
    # Calling match(args, kwargs) (line 458)
    match_call_result_92127 = invoke(stypy.reporting.localization.Localization(__file__, 458, 11), match_92124, *[file_92125], **kwargs_92126)
    
    # Testing the type of an if condition (line 458)
    if_condition_92128 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 458, 8), match_call_result_92127)
    # Assigning a type to the variable 'if_condition_92128' (line 458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'if_condition_92128', if_condition_92128)
    # SSA begins for if statement (line 458)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 459)
    # Processing the call arguments (line 459)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ind' (line 459)
    ind_92131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 33), 'ind', False)
    slice_92132 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 459, 28), ind_92131, None, None)
    # Getting the type of 'file' (line 459)
    file_92133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 28), 'file', False)
    # Obtaining the member '__getitem__' of a type (line 459)
    getitem___92134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 28), file_92133, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 459)
    subscript_call_result_92135 = invoke(stypy.reporting.localization.Localization(__file__, 459, 28), getitem___92134, slice_92132)
    
    # Processing the call keyword arguments (line 459)
    kwargs_92136 = {}
    # Getting the type of 'filtered' (line 459)
    filtered_92129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'filtered', False)
    # Obtaining the member 'append' of a type (line 459)
    append_92130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 12), filtered_92129, 'append')
    # Calling append(args, kwargs) (line 459)
    append_call_result_92137 = invoke(stypy.reporting.localization.Localization(__file__, 459, 12), append_92130, *[subscript_call_result_92135], **kwargs_92136)
    
    # SSA branch for the else part of an if statement (line 458)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 461)
    # Processing the call arguments (line 461)
    # Getting the type of 'file' (line 461)
    file_92140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 24), 'file', False)
    # Processing the call keyword arguments (line 461)
    kwargs_92141 = {}
    # Getting the type of 'rest' (line 461)
    rest_92138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'rest', False)
    # Obtaining the member 'append' of a type (line 461)
    append_92139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 12), rest_92138, 'append')
    # Calling append(args, kwargs) (line 461)
    append_call_result_92142 = invoke(stypy.reporting.localization.Localization(__file__, 461, 12), append_92139, *[file_92140], **kwargs_92141)
    
    # SSA join for if statement (line 458)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 462)
    tuple_92143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 462)
    # Adding element type (line 462)
    # Getting the type of 'filtered' (line 462)
    filtered_92144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 11), 'filtered')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 11), tuple_92143, filtered_92144)
    # Adding element type (line 462)
    # Getting the type of 'rest' (line 462)
    rest_92145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 21), 'rest')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 11), tuple_92143, rest_92145)
    
    # Assigning a type to the variable 'stypy_return_type' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'stypy_return_type', tuple_92143)
    
    # ################# End of 'filter_files(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'filter_files' in the type store
    # Getting the type of 'stypy_return_type' (line 447)
    stypy_return_type_92146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_92146)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'filter_files'
    return stypy_return_type_92146

# Assigning a type to the variable 'filter_files' (line 447)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 0), 'filter_files', filter_files)

@norecursion
def get_prefix(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_prefix'
    module_type_store = module_type_store.open_function_context('get_prefix', 465, 0, False)
    
    # Passed parameters checking function
    get_prefix.stypy_localization = localization
    get_prefix.stypy_type_of_self = None
    get_prefix.stypy_type_store = module_type_store
    get_prefix.stypy_function_name = 'get_prefix'
    get_prefix.stypy_param_names_list = ['module']
    get_prefix.stypy_varargs_param_name = None
    get_prefix.stypy_kwargs_param_name = None
    get_prefix.stypy_call_defaults = defaults
    get_prefix.stypy_call_varargs = varargs
    get_prefix.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_prefix', ['module'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_prefix', localization, ['module'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_prefix(...)' code ##################

    
    # Assigning a Call to a Name (line 466):
    
    # Assigning a Call to a Name (line 466):
    
    # Call to dirname(...): (line 466)
    # Processing the call arguments (line 466)
    
    # Call to dirname(...): (line 466)
    # Processing the call arguments (line 466)
    # Getting the type of 'module' (line 466)
    module_92153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 40), 'module', False)
    # Obtaining the member '__file__' of a type (line 466)
    file___92154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 40), module_92153, '__file__')
    # Processing the call keyword arguments (line 466)
    kwargs_92155 = {}
    # Getting the type of 'os' (line 466)
    os_92150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 466)
    path_92151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 24), os_92150, 'path')
    # Obtaining the member 'dirname' of a type (line 466)
    dirname_92152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 24), path_92151, 'dirname')
    # Calling dirname(args, kwargs) (line 466)
    dirname_call_result_92156 = invoke(stypy.reporting.localization.Localization(__file__, 466, 24), dirname_92152, *[file___92154], **kwargs_92155)
    
    # Processing the call keyword arguments (line 466)
    kwargs_92157 = {}
    # Getting the type of 'os' (line 466)
    os_92147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'os', False)
    # Obtaining the member 'path' of a type (line 466)
    path_92148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 8), os_92147, 'path')
    # Obtaining the member 'dirname' of a type (line 466)
    dirname_92149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 8), path_92148, 'dirname')
    # Calling dirname(args, kwargs) (line 466)
    dirname_call_result_92158 = invoke(stypy.reporting.localization.Localization(__file__, 466, 8), dirname_92149, *[dirname_call_result_92156], **kwargs_92157)
    
    # Assigning a type to the variable 'p' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'p', dirname_call_result_92158)
    # Getting the type of 'p' (line 467)
    p_92159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 11), 'p')
    # Assigning a type to the variable 'stypy_return_type' (line 467)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 4), 'stypy_return_type', p_92159)
    
    # ################# End of 'get_prefix(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_prefix' in the type store
    # Getting the type of 'stypy_return_type' (line 465)
    stypy_return_type_92160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_92160)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_prefix'
    return stypy_return_type_92160

# Assigning a type to the variable 'get_prefix' (line 465)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 0), 'get_prefix', get_prefix)

@norecursion
def run_compile(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run_compile'
    module_type_store = module_type_store.open_function_context('run_compile', 470, 0, False)
    
    # Passed parameters checking function
    run_compile.stypy_localization = localization
    run_compile.stypy_type_of_self = None
    run_compile.stypy_type_store = module_type_store
    run_compile.stypy_function_name = 'run_compile'
    run_compile.stypy_param_names_list = []
    run_compile.stypy_varargs_param_name = None
    run_compile.stypy_kwargs_param_name = None
    run_compile.stypy_call_defaults = defaults
    run_compile.stypy_call_varargs = varargs
    run_compile.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'run_compile', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'run_compile', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'run_compile(...)' code ##################

    str_92161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, (-1)), 'str', '\n    Do it all in one call!\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 474, 4))
    
    # 'import tempfile' statement (line 474)
    import tempfile

    import_module(stypy.reporting.localization.Localization(__file__, 474, 4), 'tempfile', tempfile, module_type_store)
    
    
    # Assigning a Call to a Name (line 476):
    
    # Assigning a Call to a Name (line 476):
    
    # Call to index(...): (line 476)
    # Processing the call arguments (line 476)
    str_92165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 23), 'str', '-c')
    # Processing the call keyword arguments (line 476)
    kwargs_92166 = {}
    # Getting the type of 'sys' (line 476)
    sys_92162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'sys', False)
    # Obtaining the member 'argv' of a type (line 476)
    argv_92163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 8), sys_92162, 'argv')
    # Obtaining the member 'index' of a type (line 476)
    index_92164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 8), argv_92163, 'index')
    # Calling index(args, kwargs) (line 476)
    index_call_result_92167 = invoke(stypy.reporting.localization.Localization(__file__, 476, 8), index_92164, *[str_92165], **kwargs_92166)
    
    # Assigning a type to the variable 'i' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'i', index_call_result_92167)
    # Deleting a member
    # Getting the type of 'sys' (line 477)
    sys_92168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'sys')
    # Obtaining the member 'argv' of a type (line 477)
    argv_92169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 8), sys_92168, 'argv')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 477)
    i_92170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 17), 'i')
    # Getting the type of 'sys' (line 477)
    sys_92171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'sys')
    # Obtaining the member 'argv' of a type (line 477)
    argv_92172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 8), sys_92171, 'argv')
    # Obtaining the member '__getitem__' of a type (line 477)
    getitem___92173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 8), argv_92172, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 477)
    subscript_call_result_92174 = invoke(stypy.reporting.localization.Localization(__file__, 477, 8), getitem___92173, i_92170)
    
    del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 477, 4), argv_92169, subscript_call_result_92174)
    
    # Assigning a Num to a Name (line 479):
    
    # Assigning a Num to a Name (line 479):
    int_92175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 23), 'int')
    # Assigning a type to the variable 'remove_build_dir' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 4), 'remove_build_dir', int_92175)
    
    
    # SSA begins for try-except statement (line 480)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 481):
    
    # Assigning a Call to a Name (line 481):
    
    # Call to index(...): (line 481)
    # Processing the call arguments (line 481)
    str_92179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 27), 'str', '--build-dir')
    # Processing the call keyword arguments (line 481)
    kwargs_92180 = {}
    # Getting the type of 'sys' (line 481)
    sys_92176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 12), 'sys', False)
    # Obtaining the member 'argv' of a type (line 481)
    argv_92177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 12), sys_92176, 'argv')
    # Obtaining the member 'index' of a type (line 481)
    index_92178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 12), argv_92177, 'index')
    # Calling index(args, kwargs) (line 481)
    index_call_result_92181 = invoke(stypy.reporting.localization.Localization(__file__, 481, 12), index_92178, *[str_92179], **kwargs_92180)
    
    # Assigning a type to the variable 'i' (line 481)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'i', index_call_result_92181)
    # SSA branch for the except part of a try statement (line 480)
    # SSA branch for the except 'ValueError' branch of a try statement (line 480)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 483):
    
    # Assigning a Name to a Name (line 483):
    # Getting the type of 'None' (line 483)
    None_92182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 12), 'None')
    # Assigning a type to the variable 'i' (line 483)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'i', None_92182)
    # SSA join for try-except statement (line 480)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 484)
    # Getting the type of 'i' (line 484)
    i_92183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 4), 'i')
    # Getting the type of 'None' (line 484)
    None_92184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 16), 'None')
    
    (may_be_92185, more_types_in_union_92186) = may_not_be_none(i_92183, None_92184)

    if may_be_92185:

        if more_types_in_union_92186:
            # Runtime conditional SSA (line 484)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Subscript to a Name (line 485):
        
        # Assigning a Subscript to a Name (line 485):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 485)
        i_92187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 29), 'i')
        int_92188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 33), 'int')
        # Applying the binary operator '+' (line 485)
        result_add_92189 = python_operator(stypy.reporting.localization.Localization(__file__, 485, 29), '+', i_92187, int_92188)
        
        # Getting the type of 'sys' (line 485)
        sys_92190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 20), 'sys')
        # Obtaining the member 'argv' of a type (line 485)
        argv_92191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 20), sys_92190, 'argv')
        # Obtaining the member '__getitem__' of a type (line 485)
        getitem___92192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 20), argv_92191, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 485)
        subscript_call_result_92193 = invoke(stypy.reporting.localization.Localization(__file__, 485, 20), getitem___92192, result_add_92189)
        
        # Assigning a type to the variable 'build_dir' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'build_dir', subscript_call_result_92193)
        # Deleting a member
        # Getting the type of 'sys' (line 486)
        sys_92194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 12), 'sys')
        # Obtaining the member 'argv' of a type (line 486)
        argv_92195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 12), sys_92194, 'argv')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 486)
        i_92196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 21), 'i')
        int_92197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 25), 'int')
        # Applying the binary operator '+' (line 486)
        result_add_92198 = python_operator(stypy.reporting.localization.Localization(__file__, 486, 21), '+', i_92196, int_92197)
        
        # Getting the type of 'sys' (line 486)
        sys_92199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 12), 'sys')
        # Obtaining the member 'argv' of a type (line 486)
        argv_92200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 12), sys_92199, 'argv')
        # Obtaining the member '__getitem__' of a type (line 486)
        getitem___92201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 12), argv_92200, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 486)
        subscript_call_result_92202 = invoke(stypy.reporting.localization.Localization(__file__, 486, 12), getitem___92201, result_add_92198)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 486, 8), argv_92195, subscript_call_result_92202)
        # Deleting a member
        # Getting the type of 'sys' (line 487)
        sys_92203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'sys')
        # Obtaining the member 'argv' of a type (line 487)
        argv_92204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 12), sys_92203, 'argv')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 487)
        i_92205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 21), 'i')
        # Getting the type of 'sys' (line 487)
        sys_92206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'sys')
        # Obtaining the member 'argv' of a type (line 487)
        argv_92207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 12), sys_92206, 'argv')
        # Obtaining the member '__getitem__' of a type (line 487)
        getitem___92208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 12), argv_92207, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 487)
        subscript_call_result_92209 = invoke(stypy.reporting.localization.Localization(__file__, 487, 12), getitem___92208, i_92205)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 8), argv_92204, subscript_call_result_92209)

        if more_types_in_union_92186:
            # Runtime conditional SSA for else branch (line 484)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_92185) or more_types_in_union_92186):
        
        # Assigning a Num to a Name (line 489):
        
        # Assigning a Num to a Name (line 489):
        int_92210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 27), 'int')
        # Assigning a type to the variable 'remove_build_dir' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'remove_build_dir', int_92210)
        
        # Assigning a Call to a Name (line 490):
        
        # Assigning a Call to a Name (line 490):
        
        # Call to mkdtemp(...): (line 490)
        # Processing the call keyword arguments (line 490)
        kwargs_92213 = {}
        # Getting the type of 'tempfile' (line 490)
        tempfile_92211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 20), 'tempfile', False)
        # Obtaining the member 'mkdtemp' of a type (line 490)
        mkdtemp_92212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 20), tempfile_92211, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 490)
        mkdtemp_call_result_92214 = invoke(stypy.reporting.localization.Localization(__file__, 490, 20), mkdtemp_92212, *[], **kwargs_92213)
        
        # Assigning a type to the variable 'build_dir' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'build_dir', mkdtemp_call_result_92214)

        if (may_be_92185 and more_types_in_union_92186):
            # SSA join for if statement (line 484)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 492):
    
    # Assigning a Call to a Name (line 492):
    
    # Call to compile(...): (line 492)
    # Processing the call arguments (line 492)
    str_92217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 23), 'str', '[-][-]link[-]')
    # Processing the call keyword arguments (line 492)
    kwargs_92218 = {}
    # Getting the type of 're' (line 492)
    re_92215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 12), 're', False)
    # Obtaining the member 'compile' of a type (line 492)
    compile_92216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 12), re_92215, 'compile')
    # Calling compile(args, kwargs) (line 492)
    compile_call_result_92219 = invoke(stypy.reporting.localization.Localization(__file__, 492, 12), compile_92216, *[str_92217], **kwargs_92218)
    
    # Assigning a type to the variable '_reg1' (line 492)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 4), '_reg1', compile_call_result_92219)
    
    # Assigning a ListComp to a Name (line 493):
    
    # Assigning a ListComp to a Name (line 493):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    int_92226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 43), 'int')
    slice_92227 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 493, 34), int_92226, None, None)
    # Getting the type of 'sys' (line 493)
    sys_92228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 34), 'sys')
    # Obtaining the member 'argv' of a type (line 493)
    argv_92229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 34), sys_92228, 'argv')
    # Obtaining the member '__getitem__' of a type (line 493)
    getitem___92230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 34), argv_92229, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 493)
    subscript_call_result_92231 = invoke(stypy.reporting.localization.Localization(__file__, 493, 34), getitem___92230, slice_92227)
    
    comprehension_92232 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 21), subscript_call_result_92231)
    # Assigning a type to the variable '_m' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 21), '_m', comprehension_92232)
    
    # Call to match(...): (line 493)
    # Processing the call arguments (line 493)
    # Getting the type of '_m' (line 493)
    _m_92223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 62), '_m', False)
    # Processing the call keyword arguments (line 493)
    kwargs_92224 = {}
    # Getting the type of '_reg1' (line 493)
    _reg1_92221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 50), '_reg1', False)
    # Obtaining the member 'match' of a type (line 493)
    match_92222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 50), _reg1_92221, 'match')
    # Calling match(args, kwargs) (line 493)
    match_call_result_92225 = invoke(stypy.reporting.localization.Localization(__file__, 493, 50), match_92222, *[_m_92223], **kwargs_92224)
    
    # Getting the type of '_m' (line 493)
    _m_92220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 21), '_m')
    list_92233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 21), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 21), list_92233, _m_92220)
    # Assigning a type to the variable 'sysinfo_flags' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'sysinfo_flags', list_92233)
    
    # Assigning a ListComp to a Attribute (line 494):
    
    # Assigning a ListComp to a Attribute (line 494):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'sys' (line 494)
    sys_92238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 29), 'sys')
    # Obtaining the member 'argv' of a type (line 494)
    argv_92239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 29), sys_92238, 'argv')
    comprehension_92240 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 494, 16), argv_92239)
    # Assigning a type to the variable '_m' (line 494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 16), '_m', comprehension_92240)
    
    # Getting the type of '_m' (line 494)
    _m_92235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 41), '_m')
    # Getting the type of 'sysinfo_flags' (line 494)
    sysinfo_flags_92236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 51), 'sysinfo_flags')
    # Applying the binary operator 'notin' (line 494)
    result_contains_92237 = python_operator(stypy.reporting.localization.Localization(__file__, 494, 41), 'notin', _m_92235, sysinfo_flags_92236)
    
    # Getting the type of '_m' (line 494)
    _m_92234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 16), '_m')
    list_92241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 494, 16), list_92241, _m_92234)
    # Getting the type of 'sys' (line 494)
    sys_92242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 4), 'sys')
    # Setting the type of the member 'argv' of a type (line 494)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 4), sys_92242, 'argv', list_92241)
    
    # Getting the type of 'sysinfo_flags' (line 495)
    sysinfo_flags_92243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 7), 'sysinfo_flags')
    # Testing the type of an if condition (line 495)
    if_condition_92244 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 495, 4), sysinfo_flags_92243)
    # Assigning a type to the variable 'if_condition_92244' (line 495)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 4), 'if_condition_92244', if_condition_92244)
    # SSA begins for if statement (line 495)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a ListComp to a Name (line 496):
    
    # Assigning a ListComp to a Name (line 496):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'sysinfo_flags' (line 496)
    sysinfo_flags_92250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 40), 'sysinfo_flags')
    comprehension_92251 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 25), sysinfo_flags_92250)
    # Assigning a type to the variable 'f' (line 496)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 25), 'f', comprehension_92251)
    
    # Obtaining the type of the subscript
    int_92245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 27), 'int')
    slice_92246 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 496, 25), int_92245, None, None)
    # Getting the type of 'f' (line 496)
    f_92247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 25), 'f')
    # Obtaining the member '__getitem__' of a type (line 496)
    getitem___92248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 25), f_92247, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 496)
    subscript_call_result_92249 = invoke(stypy.reporting.localization.Localization(__file__, 496, 25), getitem___92248, slice_92246)
    
    list_92252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 25), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 25), list_92252, subscript_call_result_92249)
    # Assigning a type to the variable 'sysinfo_flags' (line 496)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'sysinfo_flags', list_92252)
    # SSA join for if statement (line 495)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 498):
    
    # Assigning a Call to a Name (line 498):
    
    # Call to compile(...): (line 498)
    # Processing the call arguments (line 498)
    str_92255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 8), 'str', '[-][-]((no[-]|)(wrap[-]functions|lower)|debug[-]capi|quiet)|[-]include')
    # Processing the call keyword arguments (line 498)
    kwargs_92256 = {}
    # Getting the type of 're' (line 498)
    re_92253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 12), 're', False)
    # Obtaining the member 'compile' of a type (line 498)
    compile_92254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 12), re_92253, 'compile')
    # Calling compile(args, kwargs) (line 498)
    compile_call_result_92257 = invoke(stypy.reporting.localization.Localization(__file__, 498, 12), compile_92254, *[str_92255], **kwargs_92256)
    
    # Assigning a type to the variable '_reg2' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 4), '_reg2', compile_call_result_92257)
    
    # Assigning a ListComp to a Name (line 500):
    
    # Assigning a ListComp to a Name (line 500):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    int_92264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 40), 'int')
    slice_92265 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 500, 31), int_92264, None, None)
    # Getting the type of 'sys' (line 500)
    sys_92266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 31), 'sys')
    # Obtaining the member 'argv' of a type (line 500)
    argv_92267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 31), sys_92266, 'argv')
    # Obtaining the member '__getitem__' of a type (line 500)
    getitem___92268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 31), argv_92267, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 500)
    subscript_call_result_92269 = invoke(stypy.reporting.localization.Localization(__file__, 500, 31), getitem___92268, slice_92265)
    
    comprehension_92270 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 18), subscript_call_result_92269)
    # Assigning a type to the variable '_m' (line 500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 18), '_m', comprehension_92270)
    
    # Call to match(...): (line 500)
    # Processing the call arguments (line 500)
    # Getting the type of '_m' (line 500)
    _m_92261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 59), '_m', False)
    # Processing the call keyword arguments (line 500)
    kwargs_92262 = {}
    # Getting the type of '_reg2' (line 500)
    _reg2_92259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 47), '_reg2', False)
    # Obtaining the member 'match' of a type (line 500)
    match_92260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 47), _reg2_92259, 'match')
    # Calling match(args, kwargs) (line 500)
    match_call_result_92263 = invoke(stypy.reporting.localization.Localization(__file__, 500, 47), match_92260, *[_m_92261], **kwargs_92262)
    
    # Getting the type of '_m' (line 500)
    _m_92258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 18), '_m')
    list_92271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 18), list_92271, _m_92258)
    # Assigning a type to the variable 'f2py_flags' (line 500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 4), 'f2py_flags', list_92271)
    
    # Assigning a ListComp to a Attribute (line 501):
    
    # Assigning a ListComp to a Attribute (line 501):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'sys' (line 501)
    sys_92276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 29), 'sys')
    # Obtaining the member 'argv' of a type (line 501)
    argv_92277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 29), sys_92276, 'argv')
    comprehension_92278 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 501, 16), argv_92277)
    # Assigning a type to the variable '_m' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 16), '_m', comprehension_92278)
    
    # Getting the type of '_m' (line 501)
    _m_92273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 41), '_m')
    # Getting the type of 'f2py_flags' (line 501)
    f2py_flags_92274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 51), 'f2py_flags')
    # Applying the binary operator 'notin' (line 501)
    result_contains_92275 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 41), 'notin', _m_92273, f2py_flags_92274)
    
    # Getting the type of '_m' (line 501)
    _m_92272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 16), '_m')
    list_92279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 501, 16), list_92279, _m_92272)
    # Getting the type of 'sys' (line 501)
    sys_92280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'sys')
    # Setting the type of the member 'argv' of a type (line 501)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 4), sys_92280, 'argv', list_92279)
    
    # Assigning a List to a Name (line 502):
    
    # Assigning a List to a Name (line 502):
    
    # Obtaining an instance of the builtin type 'list' (line 502)
    list_92281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 502)
    
    # Assigning a type to the variable 'f2py_flags2' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'f2py_flags2', list_92281)
    
    # Assigning a Num to a Name (line 503):
    
    # Assigning a Num to a Name (line 503):
    int_92282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 9), 'int')
    # Assigning a type to the variable 'fl' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 4), 'fl', int_92282)
    
    
    # Obtaining the type of the subscript
    int_92283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 22), 'int')
    slice_92284 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 504, 13), int_92283, None, None)
    # Getting the type of 'sys' (line 504)
    sys_92285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 13), 'sys')
    # Obtaining the member 'argv' of a type (line 504)
    argv_92286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 13), sys_92285, 'argv')
    # Obtaining the member '__getitem__' of a type (line 504)
    getitem___92287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 13), argv_92286, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 504)
    subscript_call_result_92288 = invoke(stypy.reporting.localization.Localization(__file__, 504, 13), getitem___92287, slice_92284)
    
    # Testing the type of a for loop iterable (line 504)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 504, 4), subscript_call_result_92288)
    # Getting the type of the for loop variable (line 504)
    for_loop_var_92289 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 504, 4), subscript_call_result_92288)
    # Assigning a type to the variable 'a' (line 504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 4), 'a', for_loop_var_92289)
    # SSA begins for a for statement (line 504)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'a' (line 505)
    a_92290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 11), 'a')
    
    # Obtaining an instance of the builtin type 'list' (line 505)
    list_92291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 505)
    # Adding element type (line 505)
    str_92292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 17), 'str', 'only:')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 16), list_92291, str_92292)
    # Adding element type (line 505)
    str_92293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 26), 'str', 'skip:')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 16), list_92291, str_92293)
    
    # Applying the binary operator 'in' (line 505)
    result_contains_92294 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 11), 'in', a_92290, list_92291)
    
    # Testing the type of an if condition (line 505)
    if_condition_92295 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 505, 8), result_contains_92294)
    # Assigning a type to the variable 'if_condition_92295' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), 'if_condition_92295', if_condition_92295)
    # SSA begins for if statement (line 505)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 506):
    
    # Assigning a Num to a Name (line 506):
    int_92296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 17), 'int')
    # Assigning a type to the variable 'fl' (line 506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 12), 'fl', int_92296)
    # SSA branch for the else part of an if statement (line 505)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'a' (line 507)
    a_92297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 13), 'a')
    str_92298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 18), 'str', ':')
    # Applying the binary operator '==' (line 507)
    result_eq_92299 = python_operator(stypy.reporting.localization.Localization(__file__, 507, 13), '==', a_92297, str_92298)
    
    # Testing the type of an if condition (line 507)
    if_condition_92300 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 507, 13), result_eq_92299)
    # Assigning a type to the variable 'if_condition_92300' (line 507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 13), 'if_condition_92300', if_condition_92300)
    # SSA begins for if statement (line 507)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 508):
    
    # Assigning a Num to a Name (line 508):
    int_92301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 17), 'int')
    # Assigning a type to the variable 'fl' (line 508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'fl', int_92301)
    # SSA join for if statement (line 507)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 505)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'fl' (line 509)
    fl_92302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 11), 'fl')
    
    # Getting the type of 'a' (line 509)
    a_92303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 17), 'a')
    str_92304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 22), 'str', ':')
    # Applying the binary operator '==' (line 509)
    result_eq_92305 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 17), '==', a_92303, str_92304)
    
    # Applying the binary operator 'or' (line 509)
    result_or_keyword_92306 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 11), 'or', fl_92302, result_eq_92305)
    
    # Testing the type of an if condition (line 509)
    if_condition_92307 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 509, 8), result_or_keyword_92306)
    # Assigning a type to the variable 'if_condition_92307' (line 509)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'if_condition_92307', if_condition_92307)
    # SSA begins for if statement (line 509)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 510)
    # Processing the call arguments (line 510)
    # Getting the type of 'a' (line 510)
    a_92310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 31), 'a', False)
    # Processing the call keyword arguments (line 510)
    kwargs_92311 = {}
    # Getting the type of 'f2py_flags2' (line 510)
    f2py_flags2_92308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 12), 'f2py_flags2', False)
    # Obtaining the member 'append' of a type (line 510)
    append_92309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 12), f2py_flags2_92308, 'append')
    # Calling append(args, kwargs) (line 510)
    append_call_result_92312 = invoke(stypy.reporting.localization.Localization(__file__, 510, 12), append_92309, *[a_92310], **kwargs_92311)
    
    # SSA join for if statement (line 509)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'f2py_flags2' (line 511)
    f2py_flags2_92313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 7), 'f2py_flags2')
    
    
    # Obtaining the type of the subscript
    int_92314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 35), 'int')
    # Getting the type of 'f2py_flags2' (line 511)
    f2py_flags2_92315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 23), 'f2py_flags2')
    # Obtaining the member '__getitem__' of a type (line 511)
    getitem___92316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 23), f2py_flags2_92315, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 511)
    subscript_call_result_92317 = invoke(stypy.reporting.localization.Localization(__file__, 511, 23), getitem___92316, int_92314)
    
    str_92318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 42), 'str', ':')
    # Applying the binary operator '!=' (line 511)
    result_ne_92319 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 23), '!=', subscript_call_result_92317, str_92318)
    
    # Applying the binary operator 'and' (line 511)
    result_and_keyword_92320 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 7), 'and', f2py_flags2_92313, result_ne_92319)
    
    # Testing the type of an if condition (line 511)
    if_condition_92321 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 511, 4), result_and_keyword_92320)
    # Assigning a type to the variable 'if_condition_92321' (line 511)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 4), 'if_condition_92321', if_condition_92321)
    # SSA begins for if statement (line 511)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 512)
    # Processing the call arguments (line 512)
    str_92324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 27), 'str', ':')
    # Processing the call keyword arguments (line 512)
    kwargs_92325 = {}
    # Getting the type of 'f2py_flags2' (line 512)
    f2py_flags2_92322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'f2py_flags2', False)
    # Obtaining the member 'append' of a type (line 512)
    append_92323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 8), f2py_flags2_92322, 'append')
    # Calling append(args, kwargs) (line 512)
    append_call_result_92326 = invoke(stypy.reporting.localization.Localization(__file__, 512, 8), append_92323, *[str_92324], **kwargs_92325)
    
    # SSA join for if statement (line 511)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to extend(...): (line 513)
    # Processing the call arguments (line 513)
    # Getting the type of 'f2py_flags2' (line 513)
    f2py_flags2_92329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 22), 'f2py_flags2', False)
    # Processing the call keyword arguments (line 513)
    kwargs_92330 = {}
    # Getting the type of 'f2py_flags' (line 513)
    f2py_flags_92327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 4), 'f2py_flags', False)
    # Obtaining the member 'extend' of a type (line 513)
    extend_92328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 4), f2py_flags_92327, 'extend')
    # Calling extend(args, kwargs) (line 513)
    extend_call_result_92331 = invoke(stypy.reporting.localization.Localization(__file__, 513, 4), extend_92328, *[f2py_flags2_92329], **kwargs_92330)
    
    
    # Assigning a ListComp to a Attribute (line 515):
    
    # Assigning a ListComp to a Attribute (line 515):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'sys' (line 515)
    sys_92336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 29), 'sys')
    # Obtaining the member 'argv' of a type (line 515)
    argv_92337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 29), sys_92336, 'argv')
    comprehension_92338 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 16), argv_92337)
    # Assigning a type to the variable '_m' (line 515)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 16), '_m', comprehension_92338)
    
    # Getting the type of '_m' (line 515)
    _m_92333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 41), '_m')
    # Getting the type of 'f2py_flags2' (line 515)
    f2py_flags2_92334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 51), 'f2py_flags2')
    # Applying the binary operator 'notin' (line 515)
    result_contains_92335 = python_operator(stypy.reporting.localization.Localization(__file__, 515, 41), 'notin', _m_92333, f2py_flags2_92334)
    
    # Getting the type of '_m' (line 515)
    _m_92332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 16), '_m')
    list_92339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 16), list_92339, _m_92332)
    # Getting the type of 'sys' (line 515)
    sys_92340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'sys')
    # Setting the type of the member 'argv' of a type (line 515)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 4), sys_92340, 'argv', list_92339)
    
    # Assigning a Call to a Name (line 516):
    
    # Assigning a Call to a Name (line 516):
    
    # Call to compile(...): (line 516)
    # Processing the call arguments (line 516)
    str_92343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 8), 'str', '[-][-]((f(90)?compiler([-]exec|)|compiler)=|help[-]compiler)')
    # Processing the call keyword arguments (line 516)
    kwargs_92344 = {}
    # Getting the type of 're' (line 516)
    re_92341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 12), 're', False)
    # Obtaining the member 'compile' of a type (line 516)
    compile_92342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 12), re_92341, 'compile')
    # Calling compile(args, kwargs) (line 516)
    compile_call_result_92345 = invoke(stypy.reporting.localization.Localization(__file__, 516, 12), compile_92342, *[str_92343], **kwargs_92344)
    
    # Assigning a type to the variable '_reg3' (line 516)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), '_reg3', compile_call_result_92345)
    
    # Assigning a ListComp to a Name (line 518):
    
    # Assigning a ListComp to a Name (line 518):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    int_92352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 40), 'int')
    slice_92353 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 518, 31), int_92352, None, None)
    # Getting the type of 'sys' (line 518)
    sys_92354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 31), 'sys')
    # Obtaining the member 'argv' of a type (line 518)
    argv_92355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 31), sys_92354, 'argv')
    # Obtaining the member '__getitem__' of a type (line 518)
    getitem___92356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 31), argv_92355, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 518)
    subscript_call_result_92357 = invoke(stypy.reporting.localization.Localization(__file__, 518, 31), getitem___92356, slice_92353)
    
    comprehension_92358 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 18), subscript_call_result_92357)
    # Assigning a type to the variable '_m' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 18), '_m', comprehension_92358)
    
    # Call to match(...): (line 518)
    # Processing the call arguments (line 518)
    # Getting the type of '_m' (line 518)
    _m_92349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 59), '_m', False)
    # Processing the call keyword arguments (line 518)
    kwargs_92350 = {}
    # Getting the type of '_reg3' (line 518)
    _reg3_92347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 47), '_reg3', False)
    # Obtaining the member 'match' of a type (line 518)
    match_92348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 47), _reg3_92347, 'match')
    # Calling match(args, kwargs) (line 518)
    match_call_result_92351 = invoke(stypy.reporting.localization.Localization(__file__, 518, 47), match_92348, *[_m_92349], **kwargs_92350)
    
    # Getting the type of '_m' (line 518)
    _m_92346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 18), '_m')
    list_92359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 18), list_92359, _m_92346)
    # Assigning a type to the variable 'flib_flags' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 4), 'flib_flags', list_92359)
    
    # Assigning a ListComp to a Attribute (line 519):
    
    # Assigning a ListComp to a Attribute (line 519):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'sys' (line 519)
    sys_92364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 29), 'sys')
    # Obtaining the member 'argv' of a type (line 519)
    argv_92365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 29), sys_92364, 'argv')
    comprehension_92366 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 16), argv_92365)
    # Assigning a type to the variable '_m' (line 519)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 16), '_m', comprehension_92366)
    
    # Getting the type of '_m' (line 519)
    _m_92361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 41), '_m')
    # Getting the type of 'flib_flags' (line 519)
    flib_flags_92362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 51), 'flib_flags')
    # Applying the binary operator 'notin' (line 519)
    result_contains_92363 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 41), 'notin', _m_92361, flib_flags_92362)
    
    # Getting the type of '_m' (line 519)
    _m_92360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 16), '_m')
    list_92367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 16), list_92367, _m_92360)
    # Getting the type of 'sys' (line 519)
    sys_92368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 4), 'sys')
    # Setting the type of the member 'argv' of a type (line 519)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 4), sys_92368, 'argv', list_92367)
    
    # Assigning a Call to a Name (line 520):
    
    # Assigning a Call to a Name (line 520):
    
    # Call to compile(...): (line 520)
    # Processing the call arguments (line 520)
    str_92371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 8), 'str', '[-][-]((f(77|90)(flags|exec)|opt|arch)=|(debug|noopt|noarch|help[-]fcompiler))')
    # Processing the call keyword arguments (line 520)
    kwargs_92372 = {}
    # Getting the type of 're' (line 520)
    re_92369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 're', False)
    # Obtaining the member 'compile' of a type (line 520)
    compile_92370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 12), re_92369, 'compile')
    # Calling compile(args, kwargs) (line 520)
    compile_call_result_92373 = invoke(stypy.reporting.localization.Localization(__file__, 520, 12), compile_92370, *[str_92371], **kwargs_92372)
    
    # Assigning a type to the variable '_reg4' (line 520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 4), '_reg4', compile_call_result_92373)
    
    # Assigning a ListComp to a Name (line 522):
    
    # Assigning a ListComp to a Name (line 522):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    int_92380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 38), 'int')
    slice_92381 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 522, 29), int_92380, None, None)
    # Getting the type of 'sys' (line 522)
    sys_92382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 29), 'sys')
    # Obtaining the member 'argv' of a type (line 522)
    argv_92383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 29), sys_92382, 'argv')
    # Obtaining the member '__getitem__' of a type (line 522)
    getitem___92384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 29), argv_92383, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 522)
    subscript_call_result_92385 = invoke(stypy.reporting.localization.Localization(__file__, 522, 29), getitem___92384, slice_92381)
    
    comprehension_92386 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 16), subscript_call_result_92385)
    # Assigning a type to the variable '_m' (line 522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 16), '_m', comprehension_92386)
    
    # Call to match(...): (line 522)
    # Processing the call arguments (line 522)
    # Getting the type of '_m' (line 522)
    _m_92377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 57), '_m', False)
    # Processing the call keyword arguments (line 522)
    kwargs_92378 = {}
    # Getting the type of '_reg4' (line 522)
    _reg4_92375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 45), '_reg4', False)
    # Obtaining the member 'match' of a type (line 522)
    match_92376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 45), _reg4_92375, 'match')
    # Calling match(args, kwargs) (line 522)
    match_call_result_92379 = invoke(stypy.reporting.localization.Localization(__file__, 522, 45), match_92376, *[_m_92377], **kwargs_92378)
    
    # Getting the type of '_m' (line 522)
    _m_92374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 16), '_m')
    list_92387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 16), list_92387, _m_92374)
    # Assigning a type to the variable 'fc_flags' (line 522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 4), 'fc_flags', list_92387)
    
    # Assigning a ListComp to a Attribute (line 523):
    
    # Assigning a ListComp to a Attribute (line 523):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'sys' (line 523)
    sys_92392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 29), 'sys')
    # Obtaining the member 'argv' of a type (line 523)
    argv_92393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 29), sys_92392, 'argv')
    comprehension_92394 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 16), argv_92393)
    # Assigning a type to the variable '_m' (line 523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 16), '_m', comprehension_92394)
    
    # Getting the type of '_m' (line 523)
    _m_92389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 41), '_m')
    # Getting the type of 'fc_flags' (line 523)
    fc_flags_92390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 51), 'fc_flags')
    # Applying the binary operator 'notin' (line 523)
    result_contains_92391 = python_operator(stypy.reporting.localization.Localization(__file__, 523, 41), 'notin', _m_92389, fc_flags_92390)
    
    # Getting the type of '_m' (line 523)
    _m_92388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 16), '_m')
    list_92395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 16), list_92395, _m_92388)
    # Getting the type of 'sys' (line 523)
    sys_92396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 4), 'sys')
    # Setting the type of the member 'argv' of a type (line 523)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 4), sys_92396, 'argv', list_92395)
    
    int_92397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 7), 'int')
    # Testing the type of an if condition (line 525)
    if_condition_92398 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 525, 4), int_92397)
    # Assigning a type to the variable 'if_condition_92398' (line 525)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 4), 'if_condition_92398', if_condition_92398)
    # SSA begins for if statement (line 525)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 526):
    
    # Assigning a List to a Name (line 526):
    
    # Obtaining an instance of the builtin type 'list' (line 526)
    list_92399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 526)
    
    # Assigning a type to the variable 'del_list' (line 526)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'del_list', list_92399)
    
    # Getting the type of 'flib_flags' (line 527)
    flib_flags_92400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 17), 'flib_flags')
    # Testing the type of a for loop iterable (line 527)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 527, 8), flib_flags_92400)
    # Getting the type of the for loop variable (line 527)
    for_loop_var_92401 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 527, 8), flib_flags_92400)
    # Assigning a type to the variable 's' (line 527)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 's', for_loop_var_92401)
    # SSA begins for a for statement (line 527)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Str to a Name (line 528):
    
    # Assigning a Str to a Name (line 528):
    str_92402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 16), 'str', '--fcompiler=')
    # Assigning a type to the variable 'v' (line 528)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 12), 'v', str_92402)
    
    
    
    # Obtaining the type of the subscript
    
    # Call to len(...): (line 529)
    # Processing the call arguments (line 529)
    # Getting the type of 'v' (line 529)
    v_92404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 22), 'v', False)
    # Processing the call keyword arguments (line 529)
    kwargs_92405 = {}
    # Getting the type of 'len' (line 529)
    len_92403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 18), 'len', False)
    # Calling len(args, kwargs) (line 529)
    len_call_result_92406 = invoke(stypy.reporting.localization.Localization(__file__, 529, 18), len_92403, *[v_92404], **kwargs_92405)
    
    slice_92407 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 529, 15), None, len_call_result_92406, None)
    # Getting the type of 's' (line 529)
    s_92408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 15), 's')
    # Obtaining the member '__getitem__' of a type (line 529)
    getitem___92409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 15), s_92408, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 529)
    subscript_call_result_92410 = invoke(stypy.reporting.localization.Localization(__file__, 529, 15), getitem___92409, slice_92407)
    
    # Getting the type of 'v' (line 529)
    v_92411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 29), 'v')
    # Applying the binary operator '==' (line 529)
    result_eq_92412 = python_operator(stypy.reporting.localization.Localization(__file__, 529, 15), '==', subscript_call_result_92410, v_92411)
    
    # Testing the type of an if condition (line 529)
    if_condition_92413 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 529, 12), result_eq_92412)
    # Assigning a type to the variable 'if_condition_92413' (line 529)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'if_condition_92413', if_condition_92413)
    # SSA begins for if statement (line 529)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 530, 16))
    
    # 'from numpy.distutils import fcompiler' statement (line 530)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
    import_92414 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 530, 16), 'numpy.distutils')

    if (type(import_92414) is not StypyTypeError):

        if (import_92414 != 'pyd_module'):
            __import__(import_92414)
            sys_modules_92415 = sys.modules[import_92414]
            import_from_module(stypy.reporting.localization.Localization(__file__, 530, 16), 'numpy.distutils', sys_modules_92415.module_type_store, module_type_store, ['fcompiler'])
            nest_module(stypy.reporting.localization.Localization(__file__, 530, 16), __file__, sys_modules_92415, sys_modules_92415.module_type_store, module_type_store)
        else:
            from numpy.distutils import fcompiler

            import_from_module(stypy.reporting.localization.Localization(__file__, 530, 16), 'numpy.distutils', None, module_type_store, ['fcompiler'], [fcompiler])

    else:
        # Assigning a type to the variable 'numpy.distutils' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 16), 'numpy.distutils', import_92414)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')
    
    
    # Call to load_all_fcompiler_classes(...): (line 531)
    # Processing the call keyword arguments (line 531)
    kwargs_92418 = {}
    # Getting the type of 'fcompiler' (line 531)
    fcompiler_92416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 16), 'fcompiler', False)
    # Obtaining the member 'load_all_fcompiler_classes' of a type (line 531)
    load_all_fcompiler_classes_92417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 16), fcompiler_92416, 'load_all_fcompiler_classes')
    # Calling load_all_fcompiler_classes(args, kwargs) (line 531)
    load_all_fcompiler_classes_call_result_92419 = invoke(stypy.reporting.localization.Localization(__file__, 531, 16), load_all_fcompiler_classes_92417, *[], **kwargs_92418)
    
    
    # Assigning a Call to a Name (line 532):
    
    # Assigning a Call to a Name (line 532):
    
    # Call to list(...): (line 532)
    # Processing the call arguments (line 532)
    
    # Call to keys(...): (line 532)
    # Processing the call keyword arguments (line 532)
    kwargs_92424 = {}
    # Getting the type of 'fcompiler' (line 532)
    fcompiler_92421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 36), 'fcompiler', False)
    # Obtaining the member 'fcompiler_class' of a type (line 532)
    fcompiler_class_92422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 36), fcompiler_92421, 'fcompiler_class')
    # Obtaining the member 'keys' of a type (line 532)
    keys_92423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 36), fcompiler_class_92422, 'keys')
    # Calling keys(args, kwargs) (line 532)
    keys_call_result_92425 = invoke(stypy.reporting.localization.Localization(__file__, 532, 36), keys_92423, *[], **kwargs_92424)
    
    # Processing the call keyword arguments (line 532)
    kwargs_92426 = {}
    # Getting the type of 'list' (line 532)
    list_92420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 31), 'list', False)
    # Calling list(args, kwargs) (line 532)
    list_call_result_92427 = invoke(stypy.reporting.localization.Localization(__file__, 532, 31), list_92420, *[keys_call_result_92425], **kwargs_92426)
    
    # Assigning a type to the variable 'allowed_keys' (line 532)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 16), 'allowed_keys', list_call_result_92427)
    
    # Multiple assignment of 2 elements.
    
    # Assigning a Call to a Name (line 533):
    
    # Call to lower(...): (line 533)
    # Processing the call keyword arguments (line 533)
    kwargs_92437 = {}
    
    # Obtaining the type of the subscript
    
    # Call to len(...): (line 533)
    # Processing the call arguments (line 533)
    # Getting the type of 'v' (line 533)
    v_92429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 32), 'v', False)
    # Processing the call keyword arguments (line 533)
    kwargs_92430 = {}
    # Getting the type of 'len' (line 533)
    len_92428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 28), 'len', False)
    # Calling len(args, kwargs) (line 533)
    len_call_result_92431 = invoke(stypy.reporting.localization.Localization(__file__, 533, 28), len_92428, *[v_92429], **kwargs_92430)
    
    slice_92432 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 533, 26), len_call_result_92431, None, None)
    # Getting the type of 's' (line 533)
    s_92433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 26), 's', False)
    # Obtaining the member '__getitem__' of a type (line 533)
    getitem___92434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 26), s_92433, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 533)
    subscript_call_result_92435 = invoke(stypy.reporting.localization.Localization(__file__, 533, 26), getitem___92434, slice_92432)
    
    # Obtaining the member 'lower' of a type (line 533)
    lower_92436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 26), subscript_call_result_92435, 'lower')
    # Calling lower(args, kwargs) (line 533)
    lower_call_result_92438 = invoke(stypy.reporting.localization.Localization(__file__, 533, 26), lower_92436, *[], **kwargs_92437)
    
    # Assigning a type to the variable 'ov' (line 533)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 21), 'ov', lower_call_result_92438)
    
    # Assigning a Name to a Name (line 533):
    # Getting the type of 'ov' (line 533)
    ov_92439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 21), 'ov')
    # Assigning a type to the variable 'nv' (line 533)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 16), 'nv', ov_92439)
    
    
    # Getting the type of 'ov' (line 534)
    ov_92440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 19), 'ov')
    # Getting the type of 'allowed_keys' (line 534)
    allowed_keys_92441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 29), 'allowed_keys')
    # Applying the binary operator 'notin' (line 534)
    result_contains_92442 = python_operator(stypy.reporting.localization.Localization(__file__, 534, 19), 'notin', ov_92440, allowed_keys_92441)
    
    # Testing the type of an if condition (line 534)
    if_condition_92443 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 534, 16), result_contains_92442)
    # Assigning a type to the variable 'if_condition_92443' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 16), 'if_condition_92443', if_condition_92443)
    # SSA begins for if statement (line 534)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Dict to a Name (line 535):
    
    # Assigning a Dict to a Name (line 535):
    
    # Obtaining an instance of the builtin type 'dict' (line 535)
    dict_92444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 27), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 535)
    
    # Assigning a type to the variable 'vmap' (line 535)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 20), 'vmap', dict_92444)
    
    
    # SSA begins for try-except statement (line 536)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 537):
    
    # Assigning a Subscript to a Name (line 537):
    
    # Obtaining the type of the subscript
    # Getting the type of 'ov' (line 537)
    ov_92445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 34), 'ov')
    # Getting the type of 'vmap' (line 537)
    vmap_92446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 29), 'vmap')
    # Obtaining the member '__getitem__' of a type (line 537)
    getitem___92447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 29), vmap_92446, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 537)
    subscript_call_result_92448 = invoke(stypy.reporting.localization.Localization(__file__, 537, 29), getitem___92447, ov_92445)
    
    # Assigning a type to the variable 'nv' (line 537)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 24), 'nv', subscript_call_result_92448)
    # SSA branch for the except part of a try statement (line 536)
    # SSA branch for the except 'KeyError' branch of a try statement (line 536)
    module_type_store.open_ssa_branch('except')
    
    
    # Getting the type of 'ov' (line 539)
    ov_92449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 27), 'ov')
    
    # Call to values(...): (line 539)
    # Processing the call keyword arguments (line 539)
    kwargs_92452 = {}
    # Getting the type of 'vmap' (line 539)
    vmap_92450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 37), 'vmap', False)
    # Obtaining the member 'values' of a type (line 539)
    values_92451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 37), vmap_92450, 'values')
    # Calling values(args, kwargs) (line 539)
    values_call_result_92453 = invoke(stypy.reporting.localization.Localization(__file__, 539, 37), values_92451, *[], **kwargs_92452)
    
    # Applying the binary operator 'notin' (line 539)
    result_contains_92454 = python_operator(stypy.reporting.localization.Localization(__file__, 539, 27), 'notin', ov_92449, values_call_result_92453)
    
    # Testing the type of an if condition (line 539)
    if_condition_92455 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 539, 24), result_contains_92454)
    # Assigning a type to the variable 'if_condition_92455' (line 539)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 24), 'if_condition_92455', if_condition_92455)
    # SSA begins for if statement (line 539)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 540)
    # Processing the call arguments (line 540)
    str_92457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 34), 'str', 'Unknown vendor: "%s"')
    
    # Obtaining the type of the subscript
    
    # Call to len(...): (line 540)
    # Processing the call arguments (line 540)
    # Getting the type of 'v' (line 540)
    v_92459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 66), 'v', False)
    # Processing the call keyword arguments (line 540)
    kwargs_92460 = {}
    # Getting the type of 'len' (line 540)
    len_92458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 62), 'len', False)
    # Calling len(args, kwargs) (line 540)
    len_call_result_92461 = invoke(stypy.reporting.localization.Localization(__file__, 540, 62), len_92458, *[v_92459], **kwargs_92460)
    
    slice_92462 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 540, 60), len_call_result_92461, None, None)
    # Getting the type of 's' (line 540)
    s_92463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 60), 's', False)
    # Obtaining the member '__getitem__' of a type (line 540)
    getitem___92464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 60), s_92463, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 540)
    subscript_call_result_92465 = invoke(stypy.reporting.localization.Localization(__file__, 540, 60), getitem___92464, slice_92462)
    
    # Applying the binary operator '%' (line 540)
    result_mod_92466 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 34), '%', str_92457, subscript_call_result_92465)
    
    # Processing the call keyword arguments (line 540)
    kwargs_92467 = {}
    # Getting the type of 'print' (line 540)
    print_92456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 28), 'print', False)
    # Calling print(args, kwargs) (line 540)
    print_call_result_92468 = invoke(stypy.reporting.localization.Localization(__file__, 540, 28), print_92456, *[result_mod_92466], **kwargs_92467)
    
    # SSA join for if statement (line 539)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 536)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 541):
    
    # Assigning a Name to a Name (line 541):
    # Getting the type of 'ov' (line 541)
    ov_92469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 25), 'ov')
    # Assigning a type to the variable 'nv' (line 541)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 20), 'nv', ov_92469)
    # SSA join for if statement (line 534)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 542):
    
    # Assigning a Call to a Name (line 542):
    
    # Call to index(...): (line 542)
    # Processing the call arguments (line 542)
    # Getting the type of 's' (line 542)
    s_92472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 37), 's', False)
    # Processing the call keyword arguments (line 542)
    kwargs_92473 = {}
    # Getting the type of 'flib_flags' (line 542)
    flib_flags_92470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 20), 'flib_flags', False)
    # Obtaining the member 'index' of a type (line 542)
    index_92471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 20), flib_flags_92470, 'index')
    # Calling index(args, kwargs) (line 542)
    index_call_result_92474 = invoke(stypy.reporting.localization.Localization(__file__, 542, 20), index_92471, *[s_92472], **kwargs_92473)
    
    # Assigning a type to the variable 'i' (line 542)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 16), 'i', index_call_result_92474)
    
    # Assigning a BinOp to a Subscript (line 543):
    
    # Assigning a BinOp to a Subscript (line 543):
    str_92475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 32), 'str', '--fcompiler=')
    # Getting the type of 'nv' (line 543)
    nv_92476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 49), 'nv')
    # Applying the binary operator '+' (line 543)
    result_add_92477 = python_operator(stypy.reporting.localization.Localization(__file__, 543, 32), '+', str_92475, nv_92476)
    
    # Getting the type of 'flib_flags' (line 543)
    flib_flags_92478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 16), 'flib_flags')
    # Getting the type of 'i' (line 543)
    i_92479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 27), 'i')
    # Storing an element on a container (line 543)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 543, 16), flib_flags_92478, (i_92479, result_add_92477))
    # SSA join for if statement (line 529)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'del_list' (line 545)
    del_list_92480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 17), 'del_list')
    # Testing the type of a for loop iterable (line 545)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 545, 8), del_list_92480)
    # Getting the type of the for loop variable (line 545)
    for_loop_var_92481 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 545, 8), del_list_92480)
    # Assigning a type to the variable 's' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 's', for_loop_var_92481)
    # SSA begins for a for statement (line 545)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 546):
    
    # Assigning a Call to a Name (line 546):
    
    # Call to index(...): (line 546)
    # Processing the call arguments (line 546)
    # Getting the type of 's' (line 546)
    s_92484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 33), 's', False)
    # Processing the call keyword arguments (line 546)
    kwargs_92485 = {}
    # Getting the type of 'flib_flags' (line 546)
    flib_flags_92482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 16), 'flib_flags', False)
    # Obtaining the member 'index' of a type (line 546)
    index_92483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 16), flib_flags_92482, 'index')
    # Calling index(args, kwargs) (line 546)
    index_call_result_92486 = invoke(stypy.reporting.localization.Localization(__file__, 546, 16), index_92483, *[s_92484], **kwargs_92485)
    
    # Assigning a type to the variable 'i' (line 546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 12), 'i', index_call_result_92486)
    # Deleting a member
    # Getting the type of 'flib_flags' (line 547)
    flib_flags_92487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 16), 'flib_flags')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 547)
    i_92488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 27), 'i')
    # Getting the type of 'flib_flags' (line 547)
    flib_flags_92489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 16), 'flib_flags')
    # Obtaining the member '__getitem__' of a type (line 547)
    getitem___92490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 16), flib_flags_92489, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 547)
    subscript_call_result_92491 = invoke(stypy.reporting.localization.Localization(__file__, 547, 16), getitem___92490, i_92488)
    
    del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 547, 12), flib_flags_92487, subscript_call_result_92491)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Evaluating assert statement condition
    
    
    # Call to len(...): (line 548)
    # Processing the call arguments (line 548)
    # Getting the type of 'flib_flags' (line 548)
    flib_flags_92493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 19), 'flib_flags', False)
    # Processing the call keyword arguments (line 548)
    kwargs_92494 = {}
    # Getting the type of 'len' (line 548)
    len_92492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 15), 'len', False)
    # Calling len(args, kwargs) (line 548)
    len_call_result_92495 = invoke(stypy.reporting.localization.Localization(__file__, 548, 15), len_92492, *[flib_flags_92493], **kwargs_92494)
    
    int_92496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 34), 'int')
    # Applying the binary operator '<=' (line 548)
    result_le_92497 = python_operator(stypy.reporting.localization.Localization(__file__, 548, 15), '<=', len_call_result_92495, int_92496)
    
    # SSA join for if statement (line 525)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 550):
    
    # Assigning a Call to a Name (line 550):
    
    # Call to compile(...): (line 550)
    # Processing the call arguments (line 550)
    str_92500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 23), 'str', '[-][-](verbose)')
    # Processing the call keyword arguments (line 550)
    kwargs_92501 = {}
    # Getting the type of 're' (line 550)
    re_92498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 12), 're', False)
    # Obtaining the member 'compile' of a type (line 550)
    compile_92499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 12), re_92498, 'compile')
    # Calling compile(args, kwargs) (line 550)
    compile_call_result_92502 = invoke(stypy.reporting.localization.Localization(__file__, 550, 12), compile_92499, *[str_92500], **kwargs_92501)
    
    # Assigning a type to the variable '_reg5' (line 550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), '_reg5', compile_call_result_92502)
    
    # Assigning a ListComp to a Name (line 551):
    
    # Assigning a ListComp to a Name (line 551):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    int_92509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 41), 'int')
    slice_92510 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 551, 32), int_92509, None, None)
    # Getting the type of 'sys' (line 551)
    sys_92511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 32), 'sys')
    # Obtaining the member 'argv' of a type (line 551)
    argv_92512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 32), sys_92511, 'argv')
    # Obtaining the member '__getitem__' of a type (line 551)
    getitem___92513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 32), argv_92512, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 551)
    subscript_call_result_92514 = invoke(stypy.reporting.localization.Localization(__file__, 551, 32), getitem___92513, slice_92510)
    
    comprehension_92515 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 551, 19), subscript_call_result_92514)
    # Assigning a type to the variable '_m' (line 551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 19), '_m', comprehension_92515)
    
    # Call to match(...): (line 551)
    # Processing the call arguments (line 551)
    # Getting the type of '_m' (line 551)
    _m_92506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 60), '_m', False)
    # Processing the call keyword arguments (line 551)
    kwargs_92507 = {}
    # Getting the type of '_reg5' (line 551)
    _reg5_92504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 48), '_reg5', False)
    # Obtaining the member 'match' of a type (line 551)
    match_92505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 48), _reg5_92504, 'match')
    # Calling match(args, kwargs) (line 551)
    match_call_result_92508 = invoke(stypy.reporting.localization.Localization(__file__, 551, 48), match_92505, *[_m_92506], **kwargs_92507)
    
    # Getting the type of '_m' (line 551)
    _m_92503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 19), '_m')
    list_92516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 551, 19), list_92516, _m_92503)
    # Assigning a type to the variable 'setup_flags' (line 551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 4), 'setup_flags', list_92516)
    
    # Assigning a ListComp to a Attribute (line 552):
    
    # Assigning a ListComp to a Attribute (line 552):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'sys' (line 552)
    sys_92521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 29), 'sys')
    # Obtaining the member 'argv' of a type (line 552)
    argv_92522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 29), sys_92521, 'argv')
    comprehension_92523 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 552, 16), argv_92522)
    # Assigning a type to the variable '_m' (line 552)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 16), '_m', comprehension_92523)
    
    # Getting the type of '_m' (line 552)
    _m_92518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 41), '_m')
    # Getting the type of 'setup_flags' (line 552)
    setup_flags_92519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 51), 'setup_flags')
    # Applying the binary operator 'notin' (line 552)
    result_contains_92520 = python_operator(stypy.reporting.localization.Localization(__file__, 552, 41), 'notin', _m_92518, setup_flags_92519)
    
    # Getting the type of '_m' (line 552)
    _m_92517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 16), '_m')
    list_92524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 552, 16), list_92524, _m_92517)
    # Getting the type of 'sys' (line 552)
    sys_92525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 4), 'sys')
    # Setting the type of the member 'argv' of a type (line 552)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 4), sys_92525, 'argv', list_92524)
    
    
    str_92526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 7), 'str', '--quiet')
    # Getting the type of 'f2py_flags' (line 554)
    f2py_flags_92527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 20), 'f2py_flags')
    # Applying the binary operator 'in' (line 554)
    result_contains_92528 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 7), 'in', str_92526, f2py_flags_92527)
    
    # Testing the type of an if condition (line 554)
    if_condition_92529 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 554, 4), result_contains_92528)
    # Assigning a type to the variable 'if_condition_92529' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'if_condition_92529', if_condition_92529)
    # SSA begins for if statement (line 554)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 555)
    # Processing the call arguments (line 555)
    str_92532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 27), 'str', '--quiet')
    # Processing the call keyword arguments (line 555)
    kwargs_92533 = {}
    # Getting the type of 'setup_flags' (line 555)
    setup_flags_92530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 8), 'setup_flags', False)
    # Obtaining the member 'append' of a type (line 555)
    append_92531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 8), setup_flags_92530, 'append')
    # Calling append(args, kwargs) (line 555)
    append_call_result_92534 = invoke(stypy.reporting.localization.Localization(__file__, 555, 8), append_92531, *[str_92532], **kwargs_92533)
    
    # SSA join for if statement (line 554)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Str to a Name (line 557):
    
    # Assigning a Str to a Name (line 557):
    str_92535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 17), 'str', 'untitled')
    # Assigning a type to the variable 'modulename' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 4), 'modulename', str_92535)
    
    # Assigning a Subscript to a Name (line 558):
    
    # Assigning a Subscript to a Name (line 558):
    
    # Obtaining the type of the subscript
    int_92536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 23), 'int')
    slice_92537 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 558, 14), int_92536, None, None)
    # Getting the type of 'sys' (line 558)
    sys_92538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 14), 'sys')
    # Obtaining the member 'argv' of a type (line 558)
    argv_92539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 14), sys_92538, 'argv')
    # Obtaining the member '__getitem__' of a type (line 558)
    getitem___92540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 14), argv_92539, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 558)
    subscript_call_result_92541 = invoke(stypy.reporting.localization.Localization(__file__, 558, 14), getitem___92540, slice_92537)
    
    # Assigning a type to the variable 'sources' (line 558)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 4), 'sources', subscript_call_result_92541)
    
    
    # Obtaining an instance of the builtin type 'list' (line 560)
    list_92542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 560)
    # Adding element type (line 560)
    str_92543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 20), 'str', '--include_paths')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 19), list_92542, str_92543)
    # Adding element type (line 560)
    str_92544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 39), 'str', '--include-paths')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 19), list_92542, str_92544)
    
    # Testing the type of a for loop iterable (line 560)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 560, 4), list_92542)
    # Getting the type of the for loop variable (line 560)
    for_loop_var_92545 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 560, 4), list_92542)
    # Assigning a type to the variable 'optname' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'optname', for_loop_var_92545)
    # SSA begins for a for statement (line 560)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'optname' (line 561)
    optname_92546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 11), 'optname')
    # Getting the type of 'sys' (line 561)
    sys_92547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 22), 'sys')
    # Obtaining the member 'argv' of a type (line 561)
    argv_92548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 22), sys_92547, 'argv')
    # Applying the binary operator 'in' (line 561)
    result_contains_92549 = python_operator(stypy.reporting.localization.Localization(__file__, 561, 11), 'in', optname_92546, argv_92548)
    
    # Testing the type of an if condition (line 561)
    if_condition_92550 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 561, 8), result_contains_92549)
    # Assigning a type to the variable 'if_condition_92550' (line 561)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 8), 'if_condition_92550', if_condition_92550)
    # SSA begins for if statement (line 561)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 562):
    
    # Assigning a Call to a Name (line 562):
    
    # Call to index(...): (line 562)
    # Processing the call arguments (line 562)
    # Getting the type of 'optname' (line 562)
    optname_92554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 31), 'optname', False)
    # Processing the call keyword arguments (line 562)
    kwargs_92555 = {}
    # Getting the type of 'sys' (line 562)
    sys_92551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 16), 'sys', False)
    # Obtaining the member 'argv' of a type (line 562)
    argv_92552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 16), sys_92551, 'argv')
    # Obtaining the member 'index' of a type (line 562)
    index_92553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 16), argv_92552, 'index')
    # Calling index(args, kwargs) (line 562)
    index_call_result_92556 = invoke(stypy.reporting.localization.Localization(__file__, 562, 16), index_92553, *[optname_92554], **kwargs_92555)
    
    # Assigning a type to the variable 'i' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 12), 'i', index_call_result_92556)
    
    # Call to extend(...): (line 563)
    # Processing the call arguments (line 563)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 563)
    i_92559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 39), 'i', False)
    # Getting the type of 'i' (line 563)
    i_92560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 41), 'i', False)
    int_92561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 45), 'int')
    # Applying the binary operator '+' (line 563)
    result_add_92562 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 41), '+', i_92560, int_92561)
    
    slice_92563 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 563, 30), i_92559, result_add_92562, None)
    # Getting the type of 'sys' (line 563)
    sys_92564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 30), 'sys', False)
    # Obtaining the member 'argv' of a type (line 563)
    argv_92565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 30), sys_92564, 'argv')
    # Obtaining the member '__getitem__' of a type (line 563)
    getitem___92566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 30), argv_92565, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 563)
    subscript_call_result_92567 = invoke(stypy.reporting.localization.Localization(__file__, 563, 30), getitem___92566, slice_92563)
    
    # Processing the call keyword arguments (line 563)
    kwargs_92568 = {}
    # Getting the type of 'f2py_flags' (line 563)
    f2py_flags_92557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 12), 'f2py_flags', False)
    # Obtaining the member 'extend' of a type (line 563)
    extend_92558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 12), f2py_flags_92557, 'extend')
    # Calling extend(args, kwargs) (line 563)
    extend_call_result_92569 = invoke(stypy.reporting.localization.Localization(__file__, 563, 12), extend_92558, *[subscript_call_result_92567], **kwargs_92568)
    
    # Deleting a member
    # Getting the type of 'sys' (line 564)
    sys_92570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 16), 'sys')
    # Obtaining the member 'argv' of a type (line 564)
    argv_92571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 16), sys_92570, 'argv')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 564)
    i_92572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 25), 'i')
    int_92573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 29), 'int')
    # Applying the binary operator '+' (line 564)
    result_add_92574 = python_operator(stypy.reporting.localization.Localization(__file__, 564, 25), '+', i_92572, int_92573)
    
    # Getting the type of 'sys' (line 564)
    sys_92575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 16), 'sys')
    # Obtaining the member 'argv' of a type (line 564)
    argv_92576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 16), sys_92575, 'argv')
    # Obtaining the member '__getitem__' of a type (line 564)
    getitem___92577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 16), argv_92576, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 564)
    subscript_call_result_92578 = invoke(stypy.reporting.localization.Localization(__file__, 564, 16), getitem___92577, result_add_92574)
    
    del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 564, 12), argv_92571, subscript_call_result_92578)
    # Deleting a member
    # Getting the type of 'sys' (line 564)
    sys_92579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 33), 'sys')
    # Obtaining the member 'argv' of a type (line 564)
    argv_92580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 33), sys_92579, 'argv')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 564)
    i_92581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 42), 'i')
    # Getting the type of 'sys' (line 564)
    sys_92582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 33), 'sys')
    # Obtaining the member 'argv' of a type (line 564)
    argv_92583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 33), sys_92582, 'argv')
    # Obtaining the member '__getitem__' of a type (line 564)
    getitem___92584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 33), argv_92583, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 564)
    subscript_call_result_92585 = invoke(stypy.reporting.localization.Localization(__file__, 564, 33), getitem___92584, i_92581)
    
    del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 564, 12), argv_92580, subscript_call_result_92585)
    
    # Assigning a Subscript to a Name (line 565):
    
    # Assigning a Subscript to a Name (line 565):
    
    # Obtaining the type of the subscript
    int_92586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 31), 'int')
    slice_92587 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 565, 22), int_92586, None, None)
    # Getting the type of 'sys' (line 565)
    sys_92588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 22), 'sys')
    # Obtaining the member 'argv' of a type (line 565)
    argv_92589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 22), sys_92588, 'argv')
    # Obtaining the member '__getitem__' of a type (line 565)
    getitem___92590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 22), argv_92589, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 565)
    subscript_call_result_92591 = invoke(stypy.reporting.localization.Localization(__file__, 565, 22), getitem___92590, slice_92587)
    
    # Assigning a type to the variable 'sources' (line 565)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 12), 'sources', subscript_call_result_92591)
    # SSA join for if statement (line 561)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_92592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 7), 'str', '-m')
    # Getting the type of 'sys' (line 567)
    sys_92593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 15), 'sys')
    # Obtaining the member 'argv' of a type (line 567)
    argv_92594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 15), sys_92593, 'argv')
    # Applying the binary operator 'in' (line 567)
    result_contains_92595 = python_operator(stypy.reporting.localization.Localization(__file__, 567, 7), 'in', str_92592, argv_92594)
    
    # Testing the type of an if condition (line 567)
    if_condition_92596 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 567, 4), result_contains_92595)
    # Assigning a type to the variable 'if_condition_92596' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 4), 'if_condition_92596', if_condition_92596)
    # SSA begins for if statement (line 567)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 568):
    
    # Assigning a Call to a Name (line 568):
    
    # Call to index(...): (line 568)
    # Processing the call arguments (line 568)
    str_92600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 27), 'str', '-m')
    # Processing the call keyword arguments (line 568)
    kwargs_92601 = {}
    # Getting the type of 'sys' (line 568)
    sys_92597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 12), 'sys', False)
    # Obtaining the member 'argv' of a type (line 568)
    argv_92598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 12), sys_92597, 'argv')
    # Obtaining the member 'index' of a type (line 568)
    index_92599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 12), argv_92598, 'index')
    # Calling index(args, kwargs) (line 568)
    index_call_result_92602 = invoke(stypy.reporting.localization.Localization(__file__, 568, 12), index_92599, *[str_92600], **kwargs_92601)
    
    # Assigning a type to the variable 'i' (line 568)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 8), 'i', index_call_result_92602)
    
    # Assigning a Subscript to a Name (line 569):
    
    # Assigning a Subscript to a Name (line 569):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 569)
    i_92603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 30), 'i')
    int_92604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 34), 'int')
    # Applying the binary operator '+' (line 569)
    result_add_92605 = python_operator(stypy.reporting.localization.Localization(__file__, 569, 30), '+', i_92603, int_92604)
    
    # Getting the type of 'sys' (line 569)
    sys_92606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 21), 'sys')
    # Obtaining the member 'argv' of a type (line 569)
    argv_92607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 21), sys_92606, 'argv')
    # Obtaining the member '__getitem__' of a type (line 569)
    getitem___92608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 21), argv_92607, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 569)
    subscript_call_result_92609 = invoke(stypy.reporting.localization.Localization(__file__, 569, 21), getitem___92608, result_add_92605)
    
    # Assigning a type to the variable 'modulename' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 8), 'modulename', subscript_call_result_92609)
    # Deleting a member
    # Getting the type of 'sys' (line 570)
    sys_92610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 12), 'sys')
    # Obtaining the member 'argv' of a type (line 570)
    argv_92611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 12), sys_92610, 'argv')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 570)
    i_92612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 21), 'i')
    int_92613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 25), 'int')
    # Applying the binary operator '+' (line 570)
    result_add_92614 = python_operator(stypy.reporting.localization.Localization(__file__, 570, 21), '+', i_92612, int_92613)
    
    # Getting the type of 'sys' (line 570)
    sys_92615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 12), 'sys')
    # Obtaining the member 'argv' of a type (line 570)
    argv_92616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 12), sys_92615, 'argv')
    # Obtaining the member '__getitem__' of a type (line 570)
    getitem___92617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 12), argv_92616, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 570)
    subscript_call_result_92618 = invoke(stypy.reporting.localization.Localization(__file__, 570, 12), getitem___92617, result_add_92614)
    
    del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 570, 8), argv_92611, subscript_call_result_92618)
    # Deleting a member
    # Getting the type of 'sys' (line 570)
    sys_92619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 29), 'sys')
    # Obtaining the member 'argv' of a type (line 570)
    argv_92620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 29), sys_92619, 'argv')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 570)
    i_92621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 38), 'i')
    # Getting the type of 'sys' (line 570)
    sys_92622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 29), 'sys')
    # Obtaining the member 'argv' of a type (line 570)
    argv_92623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 29), sys_92622, 'argv')
    # Obtaining the member '__getitem__' of a type (line 570)
    getitem___92624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 29), argv_92623, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 570)
    subscript_call_result_92625 = invoke(stypy.reporting.localization.Localization(__file__, 570, 29), getitem___92624, i_92621)
    
    del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 570, 8), argv_92620, subscript_call_result_92625)
    
    # Assigning a Subscript to a Name (line 571):
    
    # Assigning a Subscript to a Name (line 571):
    
    # Obtaining the type of the subscript
    int_92626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 27), 'int')
    slice_92627 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 571, 18), int_92626, None, None)
    # Getting the type of 'sys' (line 571)
    sys_92628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 18), 'sys')
    # Obtaining the member 'argv' of a type (line 571)
    argv_92629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 18), sys_92628, 'argv')
    # Obtaining the member '__getitem__' of a type (line 571)
    getitem___92630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 18), argv_92629, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 571)
    subscript_call_result_92631 = invoke(stypy.reporting.localization.Localization(__file__, 571, 18), getitem___92630, slice_92627)
    
    # Assigning a type to the variable 'sources' (line 571)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'sources', subscript_call_result_92631)
    # SSA branch for the else part of an if statement (line 567)
    module_type_store.open_ssa_branch('else')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 573, 8))
    
    # 'from numpy.distutils.command.build_src import get_f2py_modulename' statement (line 573)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
    import_92632 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 573, 8), 'numpy.distutils.command.build_src')

    if (type(import_92632) is not StypyTypeError):

        if (import_92632 != 'pyd_module'):
            __import__(import_92632)
            sys_modules_92633 = sys.modules[import_92632]
            import_from_module(stypy.reporting.localization.Localization(__file__, 573, 8), 'numpy.distutils.command.build_src', sys_modules_92633.module_type_store, module_type_store, ['get_f2py_modulename'])
            nest_module(stypy.reporting.localization.Localization(__file__, 573, 8), __file__, sys_modules_92633, sys_modules_92633.module_type_store, module_type_store)
        else:
            from numpy.distutils.command.build_src import get_f2py_modulename

            import_from_module(stypy.reporting.localization.Localization(__file__, 573, 8), 'numpy.distutils.command.build_src', None, module_type_store, ['get_f2py_modulename'], [get_f2py_modulename])

    else:
        # Assigning a type to the variable 'numpy.distutils.command.build_src' (line 573)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'numpy.distutils.command.build_src', import_92632)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')
    
    
    # Assigning a Call to a Tuple (line 574):
    
    # Assigning a Call to a Name:
    
    # Call to filter_files(...): (line 574)
    # Processing the call arguments (line 574)
    str_92635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 42), 'str', '')
    str_92636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 46), 'str', '[.]pyf([.]src|)')
    # Getting the type of 'sources' (line 574)
    sources_92637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 65), 'sources', False)
    # Processing the call keyword arguments (line 574)
    kwargs_92638 = {}
    # Getting the type of 'filter_files' (line 574)
    filter_files_92634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 29), 'filter_files', False)
    # Calling filter_files(args, kwargs) (line 574)
    filter_files_call_result_92639 = invoke(stypy.reporting.localization.Localization(__file__, 574, 29), filter_files_92634, *[str_92635, str_92636, sources_92637], **kwargs_92638)
    
    # Assigning a type to the variable 'call_assignment_90860' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'call_assignment_90860', filter_files_call_result_92639)
    
    # Assigning a Call to a Name (line 574):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_92642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 8), 'int')
    # Processing the call keyword arguments
    kwargs_92643 = {}
    # Getting the type of 'call_assignment_90860' (line 574)
    call_assignment_90860_92640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'call_assignment_90860', False)
    # Obtaining the member '__getitem__' of a type (line 574)
    getitem___92641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 8), call_assignment_90860_92640, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_92644 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___92641, *[int_92642], **kwargs_92643)
    
    # Assigning a type to the variable 'call_assignment_90861' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'call_assignment_90861', getitem___call_result_92644)
    
    # Assigning a Name to a Name (line 574):
    # Getting the type of 'call_assignment_90861' (line 574)
    call_assignment_90861_92645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'call_assignment_90861')
    # Assigning a type to the variable 'pyf_files' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'pyf_files', call_assignment_90861_92645)
    
    # Assigning a Call to a Name (line 574):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_92648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 8), 'int')
    # Processing the call keyword arguments
    kwargs_92649 = {}
    # Getting the type of 'call_assignment_90860' (line 574)
    call_assignment_90860_92646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'call_assignment_90860', False)
    # Obtaining the member '__getitem__' of a type (line 574)
    getitem___92647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 8), call_assignment_90860_92646, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_92650 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___92647, *[int_92648], **kwargs_92649)
    
    # Assigning a type to the variable 'call_assignment_90862' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'call_assignment_90862', getitem___call_result_92650)
    
    # Assigning a Name to a Name (line 574):
    # Getting the type of 'call_assignment_90862' (line 574)
    call_assignment_90862_92651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'call_assignment_90862')
    # Assigning a type to the variable 'sources' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 19), 'sources', call_assignment_90862_92651)
    
    # Assigning a BinOp to a Name (line 575):
    
    # Assigning a BinOp to a Name (line 575):
    # Getting the type of 'pyf_files' (line 575)
    pyf_files_92652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 18), 'pyf_files')
    # Getting the type of 'sources' (line 575)
    sources_92653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 30), 'sources')
    # Applying the binary operator '+' (line 575)
    result_add_92654 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 18), '+', pyf_files_92652, sources_92653)
    
    # Assigning a type to the variable 'sources' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 8), 'sources', result_add_92654)
    
    # Getting the type of 'pyf_files' (line 576)
    pyf_files_92655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 17), 'pyf_files')
    # Testing the type of a for loop iterable (line 576)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 576, 8), pyf_files_92655)
    # Getting the type of the for loop variable (line 576)
    for_loop_var_92656 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 576, 8), pyf_files_92655)
    # Assigning a type to the variable 'f' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 8), 'f', for_loop_var_92656)
    # SSA begins for a for statement (line 576)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 577):
    
    # Assigning a Call to a Name (line 577):
    
    # Call to get_f2py_modulename(...): (line 577)
    # Processing the call arguments (line 577)
    # Getting the type of 'f' (line 577)
    f_92658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 45), 'f', False)
    # Processing the call keyword arguments (line 577)
    kwargs_92659 = {}
    # Getting the type of 'get_f2py_modulename' (line 577)
    get_f2py_modulename_92657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 25), 'get_f2py_modulename', False)
    # Calling get_f2py_modulename(args, kwargs) (line 577)
    get_f2py_modulename_call_result_92660 = invoke(stypy.reporting.localization.Localization(__file__, 577, 25), get_f2py_modulename_92657, *[f_92658], **kwargs_92659)
    
    # Assigning a type to the variable 'modulename' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 12), 'modulename', get_f2py_modulename_call_result_92660)
    
    # Getting the type of 'modulename' (line 578)
    modulename_92661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 15), 'modulename')
    # Testing the type of an if condition (line 578)
    if_condition_92662 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 578, 12), modulename_92661)
    # Assigning a type to the variable 'if_condition_92662' (line 578)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 12), 'if_condition_92662', if_condition_92662)
    # SSA begins for if statement (line 578)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 578)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 567)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 581):
    
    # Assigning a Call to a Name:
    
    # Call to filter_files(...): (line 581)
    # Processing the call arguments (line 581)
    str_92664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 42), 'str', '')
    str_92665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 46), 'str', '[.](o|a|so)')
    # Getting the type of 'sources' (line 581)
    sources_92666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 61), 'sources', False)
    # Processing the call keyword arguments (line 581)
    kwargs_92667 = {}
    # Getting the type of 'filter_files' (line 581)
    filter_files_92663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 29), 'filter_files', False)
    # Calling filter_files(args, kwargs) (line 581)
    filter_files_call_result_92668 = invoke(stypy.reporting.localization.Localization(__file__, 581, 29), filter_files_92663, *[str_92664, str_92665, sources_92666], **kwargs_92667)
    
    # Assigning a type to the variable 'call_assignment_90863' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 4), 'call_assignment_90863', filter_files_call_result_92668)
    
    # Assigning a Call to a Name (line 581):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_92671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 4), 'int')
    # Processing the call keyword arguments
    kwargs_92672 = {}
    # Getting the type of 'call_assignment_90863' (line 581)
    call_assignment_90863_92669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 4), 'call_assignment_90863', False)
    # Obtaining the member '__getitem__' of a type (line 581)
    getitem___92670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 4), call_assignment_90863_92669, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_92673 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___92670, *[int_92671], **kwargs_92672)
    
    # Assigning a type to the variable 'call_assignment_90864' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 4), 'call_assignment_90864', getitem___call_result_92673)
    
    # Assigning a Name to a Name (line 581):
    # Getting the type of 'call_assignment_90864' (line 581)
    call_assignment_90864_92674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 4), 'call_assignment_90864')
    # Assigning a type to the variable 'extra_objects' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 4), 'extra_objects', call_assignment_90864_92674)
    
    # Assigning a Call to a Name (line 581):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_92677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 4), 'int')
    # Processing the call keyword arguments
    kwargs_92678 = {}
    # Getting the type of 'call_assignment_90863' (line 581)
    call_assignment_90863_92675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 4), 'call_assignment_90863', False)
    # Obtaining the member '__getitem__' of a type (line 581)
    getitem___92676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 4), call_assignment_90863_92675, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_92679 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___92676, *[int_92677], **kwargs_92678)
    
    # Assigning a type to the variable 'call_assignment_90865' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 4), 'call_assignment_90865', getitem___call_result_92679)
    
    # Assigning a Name to a Name (line 581):
    # Getting the type of 'call_assignment_90865' (line 581)
    call_assignment_90865_92680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 4), 'call_assignment_90865')
    # Assigning a type to the variable 'sources' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 19), 'sources', call_assignment_90865_92680)
    
    # Assigning a Call to a Tuple (line 582):
    
    # Assigning a Call to a Name:
    
    # Call to filter_files(...): (line 582)
    # Processing the call arguments (line 582)
    str_92682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 41), 'str', '-I')
    str_92683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 47), 'str', '')
    # Getting the type of 'sources' (line 582)
    sources_92684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 51), 'sources', False)
    # Processing the call keyword arguments (line 582)
    int_92685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 74), 'int')
    keyword_92686 = int_92685
    kwargs_92687 = {'remove_prefix': keyword_92686}
    # Getting the type of 'filter_files' (line 582)
    filter_files_92681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 28), 'filter_files', False)
    # Calling filter_files(args, kwargs) (line 582)
    filter_files_call_result_92688 = invoke(stypy.reporting.localization.Localization(__file__, 582, 28), filter_files_92681, *[str_92682, str_92683, sources_92684], **kwargs_92687)
    
    # Assigning a type to the variable 'call_assignment_90866' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'call_assignment_90866', filter_files_call_result_92688)
    
    # Assigning a Call to a Name (line 582):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_92691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 4), 'int')
    # Processing the call keyword arguments
    kwargs_92692 = {}
    # Getting the type of 'call_assignment_90866' (line 582)
    call_assignment_90866_92689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'call_assignment_90866', False)
    # Obtaining the member '__getitem__' of a type (line 582)
    getitem___92690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 4), call_assignment_90866_92689, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_92693 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___92690, *[int_92691], **kwargs_92692)
    
    # Assigning a type to the variable 'call_assignment_90867' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'call_assignment_90867', getitem___call_result_92693)
    
    # Assigning a Name to a Name (line 582):
    # Getting the type of 'call_assignment_90867' (line 582)
    call_assignment_90867_92694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'call_assignment_90867')
    # Assigning a type to the variable 'include_dirs' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'include_dirs', call_assignment_90867_92694)
    
    # Assigning a Call to a Name (line 582):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_92697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 4), 'int')
    # Processing the call keyword arguments
    kwargs_92698 = {}
    # Getting the type of 'call_assignment_90866' (line 582)
    call_assignment_90866_92695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'call_assignment_90866', False)
    # Obtaining the member '__getitem__' of a type (line 582)
    getitem___92696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 4), call_assignment_90866_92695, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_92699 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___92696, *[int_92697], **kwargs_92698)
    
    # Assigning a type to the variable 'call_assignment_90868' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'call_assignment_90868', getitem___call_result_92699)
    
    # Assigning a Name to a Name (line 582):
    # Getting the type of 'call_assignment_90868' (line 582)
    call_assignment_90868_92700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'call_assignment_90868')
    # Assigning a type to the variable 'sources' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 18), 'sources', call_assignment_90868_92700)
    
    # Assigning a Call to a Tuple (line 583):
    
    # Assigning a Call to a Name:
    
    # Call to filter_files(...): (line 583)
    # Processing the call arguments (line 583)
    str_92702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 41), 'str', '-L')
    str_92703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 47), 'str', '')
    # Getting the type of 'sources' (line 583)
    sources_92704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 51), 'sources', False)
    # Processing the call keyword arguments (line 583)
    int_92705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 74), 'int')
    keyword_92706 = int_92705
    kwargs_92707 = {'remove_prefix': keyword_92706}
    # Getting the type of 'filter_files' (line 583)
    filter_files_92701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 28), 'filter_files', False)
    # Calling filter_files(args, kwargs) (line 583)
    filter_files_call_result_92708 = invoke(stypy.reporting.localization.Localization(__file__, 583, 28), filter_files_92701, *[str_92702, str_92703, sources_92704], **kwargs_92707)
    
    # Assigning a type to the variable 'call_assignment_90869' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'call_assignment_90869', filter_files_call_result_92708)
    
    # Assigning a Call to a Name (line 583):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_92711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 4), 'int')
    # Processing the call keyword arguments
    kwargs_92712 = {}
    # Getting the type of 'call_assignment_90869' (line 583)
    call_assignment_90869_92709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'call_assignment_90869', False)
    # Obtaining the member '__getitem__' of a type (line 583)
    getitem___92710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 4), call_assignment_90869_92709, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_92713 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___92710, *[int_92711], **kwargs_92712)
    
    # Assigning a type to the variable 'call_assignment_90870' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'call_assignment_90870', getitem___call_result_92713)
    
    # Assigning a Name to a Name (line 583):
    # Getting the type of 'call_assignment_90870' (line 583)
    call_assignment_90870_92714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'call_assignment_90870')
    # Assigning a type to the variable 'library_dirs' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'library_dirs', call_assignment_90870_92714)
    
    # Assigning a Call to a Name (line 583):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_92717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 4), 'int')
    # Processing the call keyword arguments
    kwargs_92718 = {}
    # Getting the type of 'call_assignment_90869' (line 583)
    call_assignment_90869_92715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'call_assignment_90869', False)
    # Obtaining the member '__getitem__' of a type (line 583)
    getitem___92716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 4), call_assignment_90869_92715, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_92719 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___92716, *[int_92717], **kwargs_92718)
    
    # Assigning a type to the variable 'call_assignment_90871' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'call_assignment_90871', getitem___call_result_92719)
    
    # Assigning a Name to a Name (line 583):
    # Getting the type of 'call_assignment_90871' (line 583)
    call_assignment_90871_92720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'call_assignment_90871')
    # Assigning a type to the variable 'sources' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 18), 'sources', call_assignment_90871_92720)
    
    # Assigning a Call to a Tuple (line 584):
    
    # Assigning a Call to a Name:
    
    # Call to filter_files(...): (line 584)
    # Processing the call arguments (line 584)
    str_92722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 38), 'str', '-l')
    str_92723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 44), 'str', '')
    # Getting the type of 'sources' (line 584)
    sources_92724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 48), 'sources', False)
    # Processing the call keyword arguments (line 584)
    int_92725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 71), 'int')
    keyword_92726 = int_92725
    kwargs_92727 = {'remove_prefix': keyword_92726}
    # Getting the type of 'filter_files' (line 584)
    filter_files_92721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 25), 'filter_files', False)
    # Calling filter_files(args, kwargs) (line 584)
    filter_files_call_result_92728 = invoke(stypy.reporting.localization.Localization(__file__, 584, 25), filter_files_92721, *[str_92722, str_92723, sources_92724], **kwargs_92727)
    
    # Assigning a type to the variable 'call_assignment_90872' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 4), 'call_assignment_90872', filter_files_call_result_92728)
    
    # Assigning a Call to a Name (line 584):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_92731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 4), 'int')
    # Processing the call keyword arguments
    kwargs_92732 = {}
    # Getting the type of 'call_assignment_90872' (line 584)
    call_assignment_90872_92729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 4), 'call_assignment_90872', False)
    # Obtaining the member '__getitem__' of a type (line 584)
    getitem___92730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 4), call_assignment_90872_92729, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_92733 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___92730, *[int_92731], **kwargs_92732)
    
    # Assigning a type to the variable 'call_assignment_90873' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 4), 'call_assignment_90873', getitem___call_result_92733)
    
    # Assigning a Name to a Name (line 584):
    # Getting the type of 'call_assignment_90873' (line 584)
    call_assignment_90873_92734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 4), 'call_assignment_90873')
    # Assigning a type to the variable 'libraries' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 4), 'libraries', call_assignment_90873_92734)
    
    # Assigning a Call to a Name (line 584):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_92737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 4), 'int')
    # Processing the call keyword arguments
    kwargs_92738 = {}
    # Getting the type of 'call_assignment_90872' (line 584)
    call_assignment_90872_92735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 4), 'call_assignment_90872', False)
    # Obtaining the member '__getitem__' of a type (line 584)
    getitem___92736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 4), call_assignment_90872_92735, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_92739 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___92736, *[int_92737], **kwargs_92738)
    
    # Assigning a type to the variable 'call_assignment_90874' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 4), 'call_assignment_90874', getitem___call_result_92739)
    
    # Assigning a Name to a Name (line 584):
    # Getting the type of 'call_assignment_90874' (line 584)
    call_assignment_90874_92740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 4), 'call_assignment_90874')
    # Assigning a type to the variable 'sources' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 15), 'sources', call_assignment_90874_92740)
    
    # Assigning a Call to a Tuple (line 585):
    
    # Assigning a Call to a Name:
    
    # Call to filter_files(...): (line 585)
    # Processing the call arguments (line 585)
    str_92742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 41), 'str', '-U')
    str_92743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 47), 'str', '')
    # Getting the type of 'sources' (line 585)
    sources_92744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 51), 'sources', False)
    # Processing the call keyword arguments (line 585)
    int_92745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 74), 'int')
    keyword_92746 = int_92745
    kwargs_92747 = {'remove_prefix': keyword_92746}
    # Getting the type of 'filter_files' (line 585)
    filter_files_92741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 28), 'filter_files', False)
    # Calling filter_files(args, kwargs) (line 585)
    filter_files_call_result_92748 = invoke(stypy.reporting.localization.Localization(__file__, 585, 28), filter_files_92741, *[str_92742, str_92743, sources_92744], **kwargs_92747)
    
    # Assigning a type to the variable 'call_assignment_90875' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 4), 'call_assignment_90875', filter_files_call_result_92748)
    
    # Assigning a Call to a Name (line 585):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_92751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 4), 'int')
    # Processing the call keyword arguments
    kwargs_92752 = {}
    # Getting the type of 'call_assignment_90875' (line 585)
    call_assignment_90875_92749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 4), 'call_assignment_90875', False)
    # Obtaining the member '__getitem__' of a type (line 585)
    getitem___92750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 4), call_assignment_90875_92749, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_92753 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___92750, *[int_92751], **kwargs_92752)
    
    # Assigning a type to the variable 'call_assignment_90876' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 4), 'call_assignment_90876', getitem___call_result_92753)
    
    # Assigning a Name to a Name (line 585):
    # Getting the type of 'call_assignment_90876' (line 585)
    call_assignment_90876_92754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 4), 'call_assignment_90876')
    # Assigning a type to the variable 'undef_macros' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 4), 'undef_macros', call_assignment_90876_92754)
    
    # Assigning a Call to a Name (line 585):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_92757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 4), 'int')
    # Processing the call keyword arguments
    kwargs_92758 = {}
    # Getting the type of 'call_assignment_90875' (line 585)
    call_assignment_90875_92755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 4), 'call_assignment_90875', False)
    # Obtaining the member '__getitem__' of a type (line 585)
    getitem___92756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 4), call_assignment_90875_92755, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_92759 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___92756, *[int_92757], **kwargs_92758)
    
    # Assigning a type to the variable 'call_assignment_90877' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 4), 'call_assignment_90877', getitem___call_result_92759)
    
    # Assigning a Name to a Name (line 585):
    # Getting the type of 'call_assignment_90877' (line 585)
    call_assignment_90877_92760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 4), 'call_assignment_90877')
    # Assigning a type to the variable 'sources' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 18), 'sources', call_assignment_90877_92760)
    
    # Assigning a Call to a Tuple (line 586):
    
    # Assigning a Call to a Name:
    
    # Call to filter_files(...): (line 586)
    # Processing the call arguments (line 586)
    str_92762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 42), 'str', '-D')
    str_92763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 48), 'str', '')
    # Getting the type of 'sources' (line 586)
    sources_92764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 52), 'sources', False)
    # Processing the call keyword arguments (line 586)
    int_92765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 75), 'int')
    keyword_92766 = int_92765
    kwargs_92767 = {'remove_prefix': keyword_92766}
    # Getting the type of 'filter_files' (line 586)
    filter_files_92761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 29), 'filter_files', False)
    # Calling filter_files(args, kwargs) (line 586)
    filter_files_call_result_92768 = invoke(stypy.reporting.localization.Localization(__file__, 586, 29), filter_files_92761, *[str_92762, str_92763, sources_92764], **kwargs_92767)
    
    # Assigning a type to the variable 'call_assignment_90878' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'call_assignment_90878', filter_files_call_result_92768)
    
    # Assigning a Call to a Name (line 586):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_92771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 4), 'int')
    # Processing the call keyword arguments
    kwargs_92772 = {}
    # Getting the type of 'call_assignment_90878' (line 586)
    call_assignment_90878_92769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'call_assignment_90878', False)
    # Obtaining the member '__getitem__' of a type (line 586)
    getitem___92770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 4), call_assignment_90878_92769, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_92773 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___92770, *[int_92771], **kwargs_92772)
    
    # Assigning a type to the variable 'call_assignment_90879' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'call_assignment_90879', getitem___call_result_92773)
    
    # Assigning a Name to a Name (line 586):
    # Getting the type of 'call_assignment_90879' (line 586)
    call_assignment_90879_92774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'call_assignment_90879')
    # Assigning a type to the variable 'define_macros' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'define_macros', call_assignment_90879_92774)
    
    # Assigning a Call to a Name (line 586):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_92777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 4), 'int')
    # Processing the call keyword arguments
    kwargs_92778 = {}
    # Getting the type of 'call_assignment_90878' (line 586)
    call_assignment_90878_92775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'call_assignment_90878', False)
    # Obtaining the member '__getitem__' of a type (line 586)
    getitem___92776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 4), call_assignment_90878_92775, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_92779 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___92776, *[int_92777], **kwargs_92778)
    
    # Assigning a type to the variable 'call_assignment_90880' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'call_assignment_90880', getitem___call_result_92779)
    
    # Assigning a Name to a Name (line 586):
    # Getting the type of 'call_assignment_90880' (line 586)
    call_assignment_90880_92780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'call_assignment_90880')
    # Assigning a type to the variable 'sources' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 19), 'sources', call_assignment_90880_92780)
    
    
    # Call to range(...): (line 587)
    # Processing the call arguments (line 587)
    
    # Call to len(...): (line 587)
    # Processing the call arguments (line 587)
    # Getting the type of 'define_macros' (line 587)
    define_macros_92783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 23), 'define_macros', False)
    # Processing the call keyword arguments (line 587)
    kwargs_92784 = {}
    # Getting the type of 'len' (line 587)
    len_92782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 19), 'len', False)
    # Calling len(args, kwargs) (line 587)
    len_call_result_92785 = invoke(stypy.reporting.localization.Localization(__file__, 587, 19), len_92782, *[define_macros_92783], **kwargs_92784)
    
    # Processing the call keyword arguments (line 587)
    kwargs_92786 = {}
    # Getting the type of 'range' (line 587)
    range_92781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 13), 'range', False)
    # Calling range(args, kwargs) (line 587)
    range_call_result_92787 = invoke(stypy.reporting.localization.Localization(__file__, 587, 13), range_92781, *[len_call_result_92785], **kwargs_92786)
    
    # Testing the type of a for loop iterable (line 587)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 587, 4), range_call_result_92787)
    # Getting the type of the for loop variable (line 587)
    for_loop_var_92788 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 587, 4), range_call_result_92787)
    # Assigning a type to the variable 'i' (line 587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 4), 'i', for_loop_var_92788)
    # SSA begins for a for statement (line 587)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 588):
    
    # Assigning a Call to a Name (line 588):
    
    # Call to split(...): (line 588)
    # Processing the call arguments (line 588)
    str_92794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 44), 'str', '=')
    int_92795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 49), 'int')
    # Processing the call keyword arguments (line 588)
    kwargs_92796 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 588)
    i_92789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 35), 'i', False)
    # Getting the type of 'define_macros' (line 588)
    define_macros_92790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 21), 'define_macros', False)
    # Obtaining the member '__getitem__' of a type (line 588)
    getitem___92791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 21), define_macros_92790, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 588)
    subscript_call_result_92792 = invoke(stypy.reporting.localization.Localization(__file__, 588, 21), getitem___92791, i_92789)
    
    # Obtaining the member 'split' of a type (line 588)
    split_92793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 21), subscript_call_result_92792, 'split')
    # Calling split(args, kwargs) (line 588)
    split_call_result_92797 = invoke(stypy.reporting.localization.Localization(__file__, 588, 21), split_92793, *[str_92794, int_92795], **kwargs_92796)
    
    # Assigning a type to the variable 'name_value' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'name_value', split_call_result_92797)
    
    
    
    # Call to len(...): (line 589)
    # Processing the call arguments (line 589)
    # Getting the type of 'name_value' (line 589)
    name_value_92799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 15), 'name_value', False)
    # Processing the call keyword arguments (line 589)
    kwargs_92800 = {}
    # Getting the type of 'len' (line 589)
    len_92798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 11), 'len', False)
    # Calling len(args, kwargs) (line 589)
    len_call_result_92801 = invoke(stypy.reporting.localization.Localization(__file__, 589, 11), len_92798, *[name_value_92799], **kwargs_92800)
    
    int_92802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 30), 'int')
    # Applying the binary operator '==' (line 589)
    result_eq_92803 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 11), '==', len_call_result_92801, int_92802)
    
    # Testing the type of an if condition (line 589)
    if_condition_92804 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 589, 8), result_eq_92803)
    # Assigning a type to the variable 'if_condition_92804' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 8), 'if_condition_92804', if_condition_92804)
    # SSA begins for if statement (line 589)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 590)
    # Processing the call arguments (line 590)
    # Getting the type of 'None' (line 590)
    None_92807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 30), 'None', False)
    # Processing the call keyword arguments (line 590)
    kwargs_92808 = {}
    # Getting the type of 'name_value' (line 590)
    name_value_92805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 12), 'name_value', False)
    # Obtaining the member 'append' of a type (line 590)
    append_92806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 12), name_value_92805, 'append')
    # Calling append(args, kwargs) (line 590)
    append_call_result_92809 = invoke(stypy.reporting.localization.Localization(__file__, 590, 12), append_92806, *[None_92807], **kwargs_92808)
    
    # SSA join for if statement (line 589)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 591)
    # Processing the call arguments (line 591)
    # Getting the type of 'name_value' (line 591)
    name_value_92811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 15), 'name_value', False)
    # Processing the call keyword arguments (line 591)
    kwargs_92812 = {}
    # Getting the type of 'len' (line 591)
    len_92810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 11), 'len', False)
    # Calling len(args, kwargs) (line 591)
    len_call_result_92813 = invoke(stypy.reporting.localization.Localization(__file__, 591, 11), len_92810, *[name_value_92811], **kwargs_92812)
    
    int_92814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 30), 'int')
    # Applying the binary operator '==' (line 591)
    result_eq_92815 = python_operator(stypy.reporting.localization.Localization(__file__, 591, 11), '==', len_call_result_92813, int_92814)
    
    # Testing the type of an if condition (line 591)
    if_condition_92816 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 591, 8), result_eq_92815)
    # Assigning a type to the variable 'if_condition_92816' (line 591)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 8), 'if_condition_92816', if_condition_92816)
    # SSA begins for if statement (line 591)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 592):
    
    # Assigning a Call to a Subscript (line 592):
    
    # Call to tuple(...): (line 592)
    # Processing the call arguments (line 592)
    # Getting the type of 'name_value' (line 592)
    name_value_92818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 37), 'name_value', False)
    # Processing the call keyword arguments (line 592)
    kwargs_92819 = {}
    # Getting the type of 'tuple' (line 592)
    tuple_92817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 31), 'tuple', False)
    # Calling tuple(args, kwargs) (line 592)
    tuple_call_result_92820 = invoke(stypy.reporting.localization.Localization(__file__, 592, 31), tuple_92817, *[name_value_92818], **kwargs_92819)
    
    # Getting the type of 'define_macros' (line 592)
    define_macros_92821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 12), 'define_macros')
    # Getting the type of 'i' (line 592)
    i_92822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 26), 'i')
    # Storing an element on a container (line 592)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 592, 12), define_macros_92821, (i_92822, tuple_call_result_92820))
    # SSA branch for the else part of an if statement (line 591)
    module_type_store.open_ssa_branch('else')
    
    # Call to print(...): (line 594)
    # Processing the call arguments (line 594)
    str_92824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 18), 'str', 'Invalid use of -D:')
    # Getting the type of 'name_value' (line 594)
    name_value_92825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 40), 'name_value', False)
    # Processing the call keyword arguments (line 594)
    kwargs_92826 = {}
    # Getting the type of 'print' (line 594)
    print_92823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 12), 'print', False)
    # Calling print(args, kwargs) (line 594)
    print_call_result_92827 = invoke(stypy.reporting.localization.Localization(__file__, 594, 12), print_92823, *[str_92824, name_value_92825], **kwargs_92826)
    
    # SSA join for if statement (line 591)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 596, 4))
    
    # 'from numpy.distutils.system_info import get_info' statement (line 596)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
    import_92828 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 596, 4), 'numpy.distutils.system_info')

    if (type(import_92828) is not StypyTypeError):

        if (import_92828 != 'pyd_module'):
            __import__(import_92828)
            sys_modules_92829 = sys.modules[import_92828]
            import_from_module(stypy.reporting.localization.Localization(__file__, 596, 4), 'numpy.distutils.system_info', sys_modules_92829.module_type_store, module_type_store, ['get_info'])
            nest_module(stypy.reporting.localization.Localization(__file__, 596, 4), __file__, sys_modules_92829, sys_modules_92829.module_type_store, module_type_store)
        else:
            from numpy.distutils.system_info import get_info

            import_from_module(stypy.reporting.localization.Localization(__file__, 596, 4), 'numpy.distutils.system_info', None, module_type_store, ['get_info'], [get_info])

    else:
        # Assigning a type to the variable 'numpy.distutils.system_info' (line 596)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 4), 'numpy.distutils.system_info', import_92828)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')
    
    
    # Assigning a Dict to a Name (line 598):
    
    # Assigning a Dict to a Name (line 598):
    
    # Obtaining an instance of the builtin type 'dict' (line 598)
    dict_92830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 598)
    
    # Assigning a type to the variable 'num_info' (line 598)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'num_info', dict_92830)
    
    # Getting the type of 'num_info' (line 599)
    num_info_92831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 7), 'num_info')
    # Testing the type of an if condition (line 599)
    if_condition_92832 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 599, 4), num_info_92831)
    # Assigning a type to the variable 'if_condition_92832' (line 599)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 4), 'if_condition_92832', if_condition_92832)
    # SSA begins for if statement (line 599)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to extend(...): (line 600)
    # Processing the call arguments (line 600)
    
    # Call to get(...): (line 600)
    # Processing the call arguments (line 600)
    str_92837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 41), 'str', 'include_dirs')
    
    # Obtaining an instance of the builtin type 'list' (line 600)
    list_92838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 57), 'list')
    # Adding type elements to the builtin type 'list' instance (line 600)
    
    # Processing the call keyword arguments (line 600)
    kwargs_92839 = {}
    # Getting the type of 'num_info' (line 600)
    num_info_92835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 28), 'num_info', False)
    # Obtaining the member 'get' of a type (line 600)
    get_92836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 28), num_info_92835, 'get')
    # Calling get(args, kwargs) (line 600)
    get_call_result_92840 = invoke(stypy.reporting.localization.Localization(__file__, 600, 28), get_92836, *[str_92837, list_92838], **kwargs_92839)
    
    # Processing the call keyword arguments (line 600)
    kwargs_92841 = {}
    # Getting the type of 'include_dirs' (line 600)
    include_dirs_92833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 8), 'include_dirs', False)
    # Obtaining the member 'extend' of a type (line 600)
    extend_92834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 8), include_dirs_92833, 'extend')
    # Calling extend(args, kwargs) (line 600)
    extend_call_result_92842 = invoke(stypy.reporting.localization.Localization(__file__, 600, 8), extend_92834, *[get_call_result_92840], **kwargs_92841)
    
    # SSA join for if statement (line 599)
    module_type_store = module_type_store.join_ssa_context()
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 602, 4))
    
    # 'from numpy.distutils.core import setup, Extension' statement (line 602)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
    import_92843 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 602, 4), 'numpy.distutils.core')

    if (type(import_92843) is not StypyTypeError):

        if (import_92843 != 'pyd_module'):
            __import__(import_92843)
            sys_modules_92844 = sys.modules[import_92843]
            import_from_module(stypy.reporting.localization.Localization(__file__, 602, 4), 'numpy.distutils.core', sys_modules_92844.module_type_store, module_type_store, ['setup', 'Extension'])
            nest_module(stypy.reporting.localization.Localization(__file__, 602, 4), __file__, sys_modules_92844, sys_modules_92844.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup, Extension

            import_from_module(stypy.reporting.localization.Localization(__file__, 602, 4), 'numpy.distutils.core', None, module_type_store, ['setup', 'Extension'], [setup, Extension])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 602)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 4), 'numpy.distutils.core', import_92843)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')
    
    
    # Assigning a Dict to a Name (line 603):
    
    # Assigning a Dict to a Name (line 603):
    
    # Obtaining an instance of the builtin type 'dict' (line 603)
    dict_92845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 603)
    # Adding element type (key, value) (line 603)
    str_92846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 16), 'str', 'name')
    # Getting the type of 'modulename' (line 603)
    modulename_92847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 24), 'modulename')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 603, 15), dict_92845, (str_92846, modulename_92847))
    # Adding element type (key, value) (line 603)
    str_92848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 36), 'str', 'sources')
    # Getting the type of 'sources' (line 603)
    sources_92849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 47), 'sources')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 603, 15), dict_92845, (str_92848, sources_92849))
    # Adding element type (key, value) (line 603)
    str_92850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 16), 'str', 'include_dirs')
    # Getting the type of 'include_dirs' (line 604)
    include_dirs_92851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 32), 'include_dirs')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 603, 15), dict_92845, (str_92850, include_dirs_92851))
    # Adding element type (key, value) (line 603)
    str_92852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 16), 'str', 'library_dirs')
    # Getting the type of 'library_dirs' (line 605)
    library_dirs_92853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 32), 'library_dirs')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 603, 15), dict_92845, (str_92852, library_dirs_92853))
    # Adding element type (key, value) (line 603)
    str_92854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 16), 'str', 'libraries')
    # Getting the type of 'libraries' (line 606)
    libraries_92855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 29), 'libraries')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 603, 15), dict_92845, (str_92854, libraries_92855))
    # Adding element type (key, value) (line 603)
    str_92856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 16), 'str', 'define_macros')
    # Getting the type of 'define_macros' (line 607)
    define_macros_92857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 33), 'define_macros')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 603, 15), dict_92845, (str_92856, define_macros_92857))
    # Adding element type (key, value) (line 603)
    str_92858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 16), 'str', 'undef_macros')
    # Getting the type of 'undef_macros' (line 608)
    undef_macros_92859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 32), 'undef_macros')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 603, 15), dict_92845, (str_92858, undef_macros_92859))
    # Adding element type (key, value) (line 603)
    str_92860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 16), 'str', 'extra_objects')
    # Getting the type of 'extra_objects' (line 609)
    extra_objects_92861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 33), 'extra_objects')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 603, 15), dict_92845, (str_92860, extra_objects_92861))
    # Adding element type (key, value) (line 603)
    str_92862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 16), 'str', 'f2py_options')
    # Getting the type of 'f2py_flags' (line 610)
    f2py_flags_92863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 32), 'f2py_flags')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 603, 15), dict_92845, (str_92862, f2py_flags_92863))
    
    # Assigning a type to the variable 'ext_args' (line 603)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 4), 'ext_args', dict_92845)
    
    # Getting the type of 'sysinfo_flags' (line 613)
    sysinfo_flags_92864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 7), 'sysinfo_flags')
    # Testing the type of an if condition (line 613)
    if_condition_92865 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 613, 4), sysinfo_flags_92864)
    # Assigning a type to the variable 'if_condition_92865' (line 613)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 4), 'if_condition_92865', if_condition_92865)
    # SSA begins for if statement (line 613)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 614, 8))
    
    # 'from numpy.distutils.misc_util import dict_append' statement (line 614)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
    import_92866 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 614, 8), 'numpy.distutils.misc_util')

    if (type(import_92866) is not StypyTypeError):

        if (import_92866 != 'pyd_module'):
            __import__(import_92866)
            sys_modules_92867 = sys.modules[import_92866]
            import_from_module(stypy.reporting.localization.Localization(__file__, 614, 8), 'numpy.distutils.misc_util', sys_modules_92867.module_type_store, module_type_store, ['dict_append'])
            nest_module(stypy.reporting.localization.Localization(__file__, 614, 8), __file__, sys_modules_92867, sys_modules_92867.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import dict_append

            import_from_module(stypy.reporting.localization.Localization(__file__, 614, 8), 'numpy.distutils.misc_util', None, module_type_store, ['dict_append'], [dict_append])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 614)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 8), 'numpy.distutils.misc_util', import_92866)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')
    
    
    # Getting the type of 'sysinfo_flags' (line 615)
    sysinfo_flags_92868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 17), 'sysinfo_flags')
    # Testing the type of a for loop iterable (line 615)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 615, 8), sysinfo_flags_92868)
    # Getting the type of the for loop variable (line 615)
    for_loop_var_92869 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 615, 8), sysinfo_flags_92868)
    # Assigning a type to the variable 'n' (line 615)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 8), 'n', for_loop_var_92869)
    # SSA begins for a for statement (line 615)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 616):
    
    # Assigning a Call to a Name (line 616):
    
    # Call to get_info(...): (line 616)
    # Processing the call arguments (line 616)
    # Getting the type of 'n' (line 616)
    n_92871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 25), 'n', False)
    # Processing the call keyword arguments (line 616)
    kwargs_92872 = {}
    # Getting the type of 'get_info' (line 616)
    get_info_92870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 16), 'get_info', False)
    # Calling get_info(args, kwargs) (line 616)
    get_info_call_result_92873 = invoke(stypy.reporting.localization.Localization(__file__, 616, 16), get_info_92870, *[n_92871], **kwargs_92872)
    
    # Assigning a type to the variable 'i' (line 616)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 12), 'i', get_info_call_result_92873)
    
    
    # Getting the type of 'i' (line 617)
    i_92874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 19), 'i')
    # Applying the 'not' unary operator (line 617)
    result_not__92875 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 15), 'not', i_92874)
    
    # Testing the type of an if condition (line 617)
    if_condition_92876 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 617, 12), result_not__92875)
    # Assigning a type to the variable 'if_condition_92876' (line 617)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 12), 'if_condition_92876', if_condition_92876)
    # SSA begins for if statement (line 617)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to outmess(...): (line 618)
    # Processing the call arguments (line 618)
    str_92878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 24), 'str', 'No %s resources found in system (try `f2py --help-link`)\n')
    
    # Call to repr(...): (line 619)
    # Processing the call arguments (line 619)
    # Getting the type of 'n' (line 619)
    n_92880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 62), 'n', False)
    # Processing the call keyword arguments (line 619)
    kwargs_92881 = {}
    # Getting the type of 'repr' (line 619)
    repr_92879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 57), 'repr', False)
    # Calling repr(args, kwargs) (line 619)
    repr_call_result_92882 = invoke(stypy.reporting.localization.Localization(__file__, 619, 57), repr_92879, *[n_92880], **kwargs_92881)
    
    # Applying the binary operator '%' (line 618)
    result_mod_92883 = python_operator(stypy.reporting.localization.Localization(__file__, 618, 24), '%', str_92878, repr_call_result_92882)
    
    # Processing the call keyword arguments (line 618)
    kwargs_92884 = {}
    # Getting the type of 'outmess' (line 618)
    outmess_92877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 16), 'outmess', False)
    # Calling outmess(args, kwargs) (line 618)
    outmess_call_result_92885 = invoke(stypy.reporting.localization.Localization(__file__, 618, 16), outmess_92877, *[result_mod_92883], **kwargs_92884)
    
    # SSA join for if statement (line 617)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to dict_append(...): (line 620)
    # Processing the call arguments (line 620)
    # Getting the type of 'ext_args' (line 620)
    ext_args_92887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 24), 'ext_args', False)
    # Processing the call keyword arguments (line 620)
    # Getting the type of 'i' (line 620)
    i_92888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 36), 'i', False)
    kwargs_92889 = {'i_92888': i_92888}
    # Getting the type of 'dict_append' (line 620)
    dict_append_92886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 12), 'dict_append', False)
    # Calling dict_append(args, kwargs) (line 620)
    dict_append_call_result_92890 = invoke(stypy.reporting.localization.Localization(__file__, 620, 12), dict_append_92886, *[ext_args_92887], **kwargs_92889)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 613)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 622):
    
    # Assigning a Call to a Name (line 622):
    
    # Call to Extension(...): (line 622)
    # Processing the call keyword arguments (line 622)
    # Getting the type of 'ext_args' (line 622)
    ext_args_92892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 22), 'ext_args', False)
    kwargs_92893 = {'ext_args_92892': ext_args_92892}
    # Getting the type of 'Extension' (line 622)
    Extension_92891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 10), 'Extension', False)
    # Calling Extension(args, kwargs) (line 622)
    Extension_call_result_92894 = invoke(stypy.reporting.localization.Localization(__file__, 622, 10), Extension_92891, *[], **kwargs_92893)
    
    # Assigning a type to the variable 'ext' (line 622)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 4), 'ext', Extension_call_result_92894)
    
    # Assigning a BinOp to a Attribute (line 623):
    
    # Assigning a BinOp to a Attribute (line 623):
    
    # Obtaining an instance of the builtin type 'list' (line 623)
    list_92895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 623)
    # Adding element type (line 623)
    
    # Obtaining the type of the subscript
    int_92896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 25), 'int')
    # Getting the type of 'sys' (line 623)
    sys_92897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 16), 'sys')
    # Obtaining the member 'argv' of a type (line 623)
    argv_92898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 16), sys_92897, 'argv')
    # Obtaining the member '__getitem__' of a type (line 623)
    getitem___92899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 16), argv_92898, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 623)
    subscript_call_result_92900 = invoke(stypy.reporting.localization.Localization(__file__, 623, 16), getitem___92899, int_92896)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 623, 15), list_92895, subscript_call_result_92900)
    
    # Getting the type of 'setup_flags' (line 623)
    setup_flags_92901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 31), 'setup_flags')
    # Applying the binary operator '+' (line 623)
    result_add_92902 = python_operator(stypy.reporting.localization.Localization(__file__, 623, 15), '+', list_92895, setup_flags_92901)
    
    # Getting the type of 'sys' (line 623)
    sys_92903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 4), 'sys')
    # Setting the type of the member 'argv' of a type (line 623)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 4), sys_92903, 'argv', result_add_92902)
    
    # Call to extend(...): (line 624)
    # Processing the call arguments (line 624)
    
    # Obtaining an instance of the builtin type 'list' (line 624)
    list_92907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 624)
    # Adding element type (line 624)
    str_92908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 21), 'str', 'build')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 624, 20), list_92907, str_92908)
    # Adding element type (line 624)
    str_92909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 21), 'str', '--build-temp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 624, 20), list_92907, str_92909)
    # Adding element type (line 624)
    # Getting the type of 'build_dir' (line 625)
    build_dir_92910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 37), 'build_dir', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 624, 20), list_92907, build_dir_92910)
    # Adding element type (line 624)
    str_92911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 21), 'str', '--build-base')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 624, 20), list_92907, str_92911)
    # Adding element type (line 624)
    # Getting the type of 'build_dir' (line 626)
    build_dir_92912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 37), 'build_dir', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 624, 20), list_92907, build_dir_92912)
    # Adding element type (line 624)
    str_92913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 21), 'str', '--build-platlib')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 624, 20), list_92907, str_92913)
    # Adding element type (line 624)
    str_92914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 40), 'str', '.')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 624, 20), list_92907, str_92914)
    
    # Processing the call keyword arguments (line 624)
    kwargs_92915 = {}
    # Getting the type of 'sys' (line 624)
    sys_92904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 4), 'sys', False)
    # Obtaining the member 'argv' of a type (line 624)
    argv_92905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 4), sys_92904, 'argv')
    # Obtaining the member 'extend' of a type (line 624)
    extend_92906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 4), argv_92905, 'extend')
    # Calling extend(args, kwargs) (line 624)
    extend_call_result_92916 = invoke(stypy.reporting.localization.Localization(__file__, 624, 4), extend_92906, *[list_92907], **kwargs_92915)
    
    
    # Getting the type of 'fc_flags' (line 628)
    fc_flags_92917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 7), 'fc_flags')
    # Testing the type of an if condition (line 628)
    if_condition_92918 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 628, 4), fc_flags_92917)
    # Assigning a type to the variable 'if_condition_92918' (line 628)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 4), 'if_condition_92918', if_condition_92918)
    # SSA begins for if statement (line 628)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to extend(...): (line 629)
    # Processing the call arguments (line 629)
    
    # Obtaining an instance of the builtin type 'list' (line 629)
    list_92922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 629)
    # Adding element type (line 629)
    str_92923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 25), 'str', 'config_fc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 629, 24), list_92922, str_92923)
    
    # Getting the type of 'fc_flags' (line 629)
    fc_flags_92924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 40), 'fc_flags', False)
    # Applying the binary operator '+' (line 629)
    result_add_92925 = python_operator(stypy.reporting.localization.Localization(__file__, 629, 24), '+', list_92922, fc_flags_92924)
    
    # Processing the call keyword arguments (line 629)
    kwargs_92926 = {}
    # Getting the type of 'sys' (line 629)
    sys_92919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 8), 'sys', False)
    # Obtaining the member 'argv' of a type (line 629)
    argv_92920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 8), sys_92919, 'argv')
    # Obtaining the member 'extend' of a type (line 629)
    extend_92921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 8), argv_92920, 'extend')
    # Calling extend(args, kwargs) (line 629)
    extend_call_result_92927 = invoke(stypy.reporting.localization.Localization(__file__, 629, 8), extend_92921, *[result_add_92925], **kwargs_92926)
    
    # SSA join for if statement (line 628)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'flib_flags' (line 630)
    flib_flags_92928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 7), 'flib_flags')
    # Testing the type of an if condition (line 630)
    if_condition_92929 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 630, 4), flib_flags_92928)
    # Assigning a type to the variable 'if_condition_92929' (line 630)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 4), 'if_condition_92929', if_condition_92929)
    # SSA begins for if statement (line 630)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to extend(...): (line 631)
    # Processing the call arguments (line 631)
    
    # Obtaining an instance of the builtin type 'list' (line 631)
    list_92933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 631)
    # Adding element type (line 631)
    str_92934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 25), 'str', 'build_ext')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 631, 24), list_92933, str_92934)
    
    # Getting the type of 'flib_flags' (line 631)
    flib_flags_92935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 40), 'flib_flags', False)
    # Applying the binary operator '+' (line 631)
    result_add_92936 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 24), '+', list_92933, flib_flags_92935)
    
    # Processing the call keyword arguments (line 631)
    kwargs_92937 = {}
    # Getting the type of 'sys' (line 631)
    sys_92930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 8), 'sys', False)
    # Obtaining the member 'argv' of a type (line 631)
    argv_92931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 8), sys_92930, 'argv')
    # Obtaining the member 'extend' of a type (line 631)
    extend_92932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 8), argv_92931, 'extend')
    # Calling extend(args, kwargs) (line 631)
    extend_call_result_92938 = invoke(stypy.reporting.localization.Localization(__file__, 631, 8), extend_92932, *[result_add_92936], **kwargs_92937)
    
    # SSA join for if statement (line 630)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to setup(...): (line 633)
    # Processing the call keyword arguments (line 633)
    
    # Obtaining an instance of the builtin type 'list' (line 633)
    list_92940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 633)
    # Adding element type (line 633)
    # Getting the type of 'ext' (line 633)
    ext_92941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 23), 'ext', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 633, 22), list_92940, ext_92941)
    
    keyword_92942 = list_92940
    kwargs_92943 = {'ext_modules': keyword_92942}
    # Getting the type of 'setup' (line 633)
    setup_92939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 633)
    setup_call_result_92944 = invoke(stypy.reporting.localization.Localization(__file__, 633, 4), setup_92939, *[], **kwargs_92943)
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'remove_build_dir' (line 635)
    remove_build_dir_92945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 7), 'remove_build_dir')
    
    # Call to exists(...): (line 635)
    # Processing the call arguments (line 635)
    # Getting the type of 'build_dir' (line 635)
    build_dir_92949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 43), 'build_dir', False)
    # Processing the call keyword arguments (line 635)
    kwargs_92950 = {}
    # Getting the type of 'os' (line 635)
    os_92946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 28), 'os', False)
    # Obtaining the member 'path' of a type (line 635)
    path_92947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 28), os_92946, 'path')
    # Obtaining the member 'exists' of a type (line 635)
    exists_92948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 28), path_92947, 'exists')
    # Calling exists(args, kwargs) (line 635)
    exists_call_result_92951 = invoke(stypy.reporting.localization.Localization(__file__, 635, 28), exists_92948, *[build_dir_92949], **kwargs_92950)
    
    # Applying the binary operator 'and' (line 635)
    result_and_keyword_92952 = python_operator(stypy.reporting.localization.Localization(__file__, 635, 7), 'and', remove_build_dir_92945, exists_call_result_92951)
    
    # Testing the type of an if condition (line 635)
    if_condition_92953 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 635, 4), result_and_keyword_92952)
    # Assigning a type to the variable 'if_condition_92953' (line 635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'if_condition_92953', if_condition_92953)
    # SSA begins for if statement (line 635)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 636, 8))
    
    # 'import shutil' statement (line 636)
    import shutil

    import_module(stypy.reporting.localization.Localization(__file__, 636, 8), 'shutil', shutil, module_type_store)
    
    
    # Call to outmess(...): (line 637)
    # Processing the call arguments (line 637)
    str_92955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 16), 'str', 'Removing build directory %s\n')
    # Getting the type of 'build_dir' (line 637)
    build_dir_92956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 51), 'build_dir', False)
    # Applying the binary operator '%' (line 637)
    result_mod_92957 = python_operator(stypy.reporting.localization.Localization(__file__, 637, 16), '%', str_92955, build_dir_92956)
    
    # Processing the call keyword arguments (line 637)
    kwargs_92958 = {}
    # Getting the type of 'outmess' (line 637)
    outmess_92954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 8), 'outmess', False)
    # Calling outmess(args, kwargs) (line 637)
    outmess_call_result_92959 = invoke(stypy.reporting.localization.Localization(__file__, 637, 8), outmess_92954, *[result_mod_92957], **kwargs_92958)
    
    
    # Call to rmtree(...): (line 638)
    # Processing the call arguments (line 638)
    # Getting the type of 'build_dir' (line 638)
    build_dir_92962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 22), 'build_dir', False)
    # Processing the call keyword arguments (line 638)
    kwargs_92963 = {}
    # Getting the type of 'shutil' (line 638)
    shutil_92960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 8), 'shutil', False)
    # Obtaining the member 'rmtree' of a type (line 638)
    rmtree_92961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 8), shutil_92960, 'rmtree')
    # Calling rmtree(args, kwargs) (line 638)
    rmtree_call_result_92964 = invoke(stypy.reporting.localization.Localization(__file__, 638, 8), rmtree_92961, *[build_dir_92962], **kwargs_92963)
    
    # SSA join for if statement (line 635)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'run_compile(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run_compile' in the type store
    # Getting the type of 'stypy_return_type' (line 470)
    stypy_return_type_92965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_92965)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run_compile'
    return stypy_return_type_92965

# Assigning a type to the variable 'run_compile' (line 470)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 0), 'run_compile', run_compile)

@norecursion
def main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'main'
    module_type_store = module_type_store.open_function_context('main', 641, 0, False)
    
    # Passed parameters checking function
    main.stypy_localization = localization
    main.stypy_type_of_self = None
    main.stypy_type_store = module_type_store
    main.stypy_function_name = 'main'
    main.stypy_param_names_list = []
    main.stypy_varargs_param_name = None
    main.stypy_kwargs_param_name = None
    main.stypy_call_defaults = defaults
    main.stypy_call_varargs = varargs
    main.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'main', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'main', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'main(...)' code ##################

    
    
    str_92966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, 7), 'str', '--help-link')
    
    # Obtaining the type of the subscript
    int_92967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, 33), 'int')
    slice_92968 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 642, 24), int_92967, None, None)
    # Getting the type of 'sys' (line 642)
    sys_92969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 24), 'sys')
    # Obtaining the member 'argv' of a type (line 642)
    argv_92970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 24), sys_92969, 'argv')
    # Obtaining the member '__getitem__' of a type (line 642)
    getitem___92971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 24), argv_92970, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 642)
    subscript_call_result_92972 = invoke(stypy.reporting.localization.Localization(__file__, 642, 24), getitem___92971, slice_92968)
    
    # Applying the binary operator 'in' (line 642)
    result_contains_92973 = python_operator(stypy.reporting.localization.Localization(__file__, 642, 7), 'in', str_92966, subscript_call_result_92972)
    
    # Testing the type of an if condition (line 642)
    if_condition_92974 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 642, 4), result_contains_92973)
    # Assigning a type to the variable 'if_condition_92974' (line 642)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 4), 'if_condition_92974', if_condition_92974)
    # SSA begins for if statement (line 642)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to remove(...): (line 643)
    # Processing the call arguments (line 643)
    str_92978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 24), 'str', '--help-link')
    # Processing the call keyword arguments (line 643)
    kwargs_92979 = {}
    # Getting the type of 'sys' (line 643)
    sys_92975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 8), 'sys', False)
    # Obtaining the member 'argv' of a type (line 643)
    argv_92976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 8), sys_92975, 'argv')
    # Obtaining the member 'remove' of a type (line 643)
    remove_92977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 8), argv_92976, 'remove')
    # Calling remove(args, kwargs) (line 643)
    remove_call_result_92980 = invoke(stypy.reporting.localization.Localization(__file__, 643, 8), remove_92977, *[str_92978], **kwargs_92979)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 644, 8))
    
    # 'from numpy.distutils.system_info import show_all' statement (line 644)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
    import_92981 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 644, 8), 'numpy.distutils.system_info')

    if (type(import_92981) is not StypyTypeError):

        if (import_92981 != 'pyd_module'):
            __import__(import_92981)
            sys_modules_92982 = sys.modules[import_92981]
            import_from_module(stypy.reporting.localization.Localization(__file__, 644, 8), 'numpy.distutils.system_info', sys_modules_92982.module_type_store, module_type_store, ['show_all'])
            nest_module(stypy.reporting.localization.Localization(__file__, 644, 8), __file__, sys_modules_92982, sys_modules_92982.module_type_store, module_type_store)
        else:
            from numpy.distutils.system_info import show_all

            import_from_module(stypy.reporting.localization.Localization(__file__, 644, 8), 'numpy.distutils.system_info', None, module_type_store, ['show_all'], [show_all])

    else:
        # Assigning a type to the variable 'numpy.distutils.system_info' (line 644)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 8), 'numpy.distutils.system_info', import_92981)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')
    
    
    # Call to show_all(...): (line 645)
    # Processing the call keyword arguments (line 645)
    kwargs_92984 = {}
    # Getting the type of 'show_all' (line 645)
    show_all_92983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 8), 'show_all', False)
    # Calling show_all(args, kwargs) (line 645)
    show_all_call_result_92985 = invoke(stypy.reporting.localization.Localization(__file__, 645, 8), show_all_92983, *[], **kwargs_92984)
    
    # Assigning a type to the variable 'stypy_return_type' (line 646)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 642)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_92986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 647, 7), 'str', '-c')
    
    # Obtaining the type of the subscript
    int_92987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 647, 24), 'int')
    slice_92988 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 647, 15), int_92987, None, None)
    # Getting the type of 'sys' (line 647)
    sys_92989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 15), 'sys')
    # Obtaining the member 'argv' of a type (line 647)
    argv_92990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 15), sys_92989, 'argv')
    # Obtaining the member '__getitem__' of a type (line 647)
    getitem___92991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 15), argv_92990, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 647)
    subscript_call_result_92992 = invoke(stypy.reporting.localization.Localization(__file__, 647, 15), getitem___92991, slice_92988)
    
    # Applying the binary operator 'in' (line 647)
    result_contains_92993 = python_operator(stypy.reporting.localization.Localization(__file__, 647, 7), 'in', str_92986, subscript_call_result_92992)
    
    # Testing the type of an if condition (line 647)
    if_condition_92994 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 647, 4), result_contains_92993)
    # Assigning a type to the variable 'if_condition_92994' (line 647)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 4), 'if_condition_92994', if_condition_92994)
    # SSA begins for if statement (line 647)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to run_compile(...): (line 648)
    # Processing the call keyword arguments (line 648)
    kwargs_92996 = {}
    # Getting the type of 'run_compile' (line 648)
    run_compile_92995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 8), 'run_compile', False)
    # Calling run_compile(args, kwargs) (line 648)
    run_compile_call_result_92997 = invoke(stypy.reporting.localization.Localization(__file__, 648, 8), run_compile_92995, *[], **kwargs_92996)
    
    # SSA branch for the else part of an if statement (line 647)
    module_type_store.open_ssa_branch('else')
    
    # Call to run_main(...): (line 650)
    # Processing the call arguments (line 650)
    
    # Obtaining the type of the subscript
    int_92999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 26), 'int')
    slice_93000 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 650, 17), int_92999, None, None)
    # Getting the type of 'sys' (line 650)
    sys_93001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 17), 'sys', False)
    # Obtaining the member 'argv' of a type (line 650)
    argv_93002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 17), sys_93001, 'argv')
    # Obtaining the member '__getitem__' of a type (line 650)
    getitem___93003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 17), argv_93002, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 650)
    subscript_call_result_93004 = invoke(stypy.reporting.localization.Localization(__file__, 650, 17), getitem___93003, slice_93000)
    
    # Processing the call keyword arguments (line 650)
    kwargs_93005 = {}
    # Getting the type of 'run_main' (line 650)
    run_main_92998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 8), 'run_main', False)
    # Calling run_main(args, kwargs) (line 650)
    run_main_call_result_93006 = invoke(stypy.reporting.localization.Localization(__file__, 650, 8), run_main_92998, *[subscript_call_result_93004], **kwargs_93005)
    
    # SSA join for if statement (line 647)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 641)
    stypy_return_type_93007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_93007)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_93007

# Assigning a type to the variable 'main' (line 641)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 0), 'main', main)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: '''
3: exec_command
4: 
5: Implements exec_command function that is (almost) equivalent to
6: commands.getstatusoutput function but on NT, DOS systems the
7: returned status is actually correct (though, the returned status
8: values may be different by a factor). In addition, exec_command
9: takes keyword arguments for (re-)defining environment variables.
10: 
11: Provides functions:
12: 
13:   exec_command  --- execute command in a specified directory and
14:                     in the modified environment.
15:   find_executable --- locate a command using info from environment
16:                     variable PATH. Equivalent to posix `which`
17:                     command.
18: 
19: Author: Pearu Peterson <pearu@cens.ioc.ee>
20: Created: 11 January 2003
21: 
22: Requires: Python 2.x
23: 
24: Succesfully tested on:
25: 
26: ========  ============  =================================================
27: os.name   sys.platform  comments
28: ========  ============  =================================================
29: posix     linux2        Debian (sid) Linux, Python 2.1.3+, 2.2.3+, 2.3.3
30:                         PyCrust 0.9.3, Idle 1.0.2
31: posix     linux2        Red Hat 9 Linux, Python 2.1.3, 2.2.2, 2.3.2
32: posix     sunos5        SunOS 5.9, Python 2.2, 2.3.2
33: posix     darwin        Darwin 7.2.0, Python 2.3
34: nt        win32         Windows Me
35:                         Python 2.3(EE), Idle 1.0, PyCrust 0.7.2
36:                         Python 2.1.1 Idle 0.8
37: nt        win32         Windows 98, Python 2.1.1. Idle 0.8
38: nt        win32         Cygwin 98-4.10, Python 2.1.1(MSC) - echo tests
39:                         fail i.e. redefining environment variables may
40:                         not work. FIXED: don't use cygwin echo!
41:                         Comment: also `cmd /c echo` will not work
42:                         but redefining environment variables do work.
43: posix     cygwin        Cygwin 98-4.10, Python 2.3.3(cygming special)
44: nt        win32         Windows XP, Python 2.3.3
45: ========  ============  =================================================
46: 
47: Known bugs:
48: 
49: * Tests, that send messages to stderr, fail when executed from MSYS prompt
50:   because the messages are lost at some point.
51: 
52: '''
53: from __future__ import division, absolute_import, print_function
54: 
55: __all__ = ['exec_command', 'find_executable']
56: 
57: import os
58: import sys
59: import shlex
60: 
61: from numpy.distutils.misc_util import is_sequence, make_temp_file
62: from numpy.distutils import log
63: from numpy.distutils.compat import get_exception
64: 
65: from numpy.compat import open_latin1
66: 
67: def temp_file_name():
68:     fo, name = make_temp_file()
69:     fo.close()
70:     return name
71: 
72: def get_pythonexe():
73:     pythonexe = sys.executable
74:     if os.name in ['nt', 'dos']:
75:         fdir, fn = os.path.split(pythonexe)
76:         fn = fn.upper().replace('PYTHONW', 'PYTHON')
77:         pythonexe = os.path.join(fdir, fn)
78:         assert os.path.isfile(pythonexe), '%r is not a file' % (pythonexe,)
79:     return pythonexe
80: 
81: def find_executable(exe, path=None, _cache={}):
82:     '''Return full path of a executable or None.
83: 
84:     Symbolic links are not followed.
85:     '''
86:     key = exe, path
87:     try:
88:         return _cache[key]
89:     except KeyError:
90:         pass
91:     log.debug('find_executable(%r)' % exe)
92:     orig_exe = exe
93: 
94:     if path is None:
95:         path = os.environ.get('PATH', os.defpath)
96:     if os.name=='posix':
97:         realpath = os.path.realpath
98:     else:
99:         realpath = lambda a:a
100: 
101:     if exe.startswith('"'):
102:         exe = exe[1:-1]
103: 
104:     suffixes = ['']
105:     if os.name in ['nt', 'dos', 'os2']:
106:         fn, ext = os.path.splitext(exe)
107:         extra_suffixes = ['.exe', '.com', '.bat']
108:         if ext.lower() not in extra_suffixes:
109:             suffixes = extra_suffixes
110: 
111:     if os.path.isabs(exe):
112:         paths = ['']
113:     else:
114:         paths = [ os.path.abspath(p) for p in path.split(os.pathsep) ]
115: 
116:     for path in paths:
117:         fn = os.path.join(path, exe)
118:         for s in suffixes:
119:             f_ext = fn+s
120:             if not os.path.islink(f_ext):
121:                 f_ext = realpath(f_ext)
122:             if os.path.isfile(f_ext) and os.access(f_ext, os.X_OK):
123:                 log.info('Found executable %s' % f_ext)
124:                 _cache[key] = f_ext
125:                 return f_ext
126: 
127:     log.warn('Could not locate executable %s' % orig_exe)
128:     return None
129: 
130: ############################################################
131: 
132: def _preserve_environment( names ):
133:     log.debug('_preserve_environment(%r)' % (names))
134:     env = {}
135:     for name in names:
136:         env[name] = os.environ.get(name)
137:     return env
138: 
139: def _update_environment( **env ):
140:     log.debug('_update_environment(...)')
141:     for name, value in env.items():
142:         os.environ[name] = value or ''
143: 
144: def _supports_fileno(stream):
145:     '''
146:     Returns True if 'stream' supports the file descriptor and allows fileno().
147:     '''
148:     if hasattr(stream, 'fileno'):
149:         try:
150:             r = stream.fileno()
151:             return True
152:         except IOError:
153:             return False
154:     else:
155:         return False
156: 
157: def exec_command(command, execute_in='', use_shell=None, use_tee=None,
158:                  _with_python = 1, **env ):
159:     '''
160:     Return (status,output) of executed command.
161: 
162:     Parameters
163:     ----------
164:     command : str
165:         A concatenated string of executable and arguments.
166:     execute_in : str
167:         Before running command ``cd execute_in`` and after ``cd -``.
168:     use_shell : {bool, None}, optional
169:         If True, execute ``sh -c command``. Default None (True)
170:     use_tee : {bool, None}, optional
171:         If True use tee. Default None (True)
172: 
173: 
174:     Returns
175:     -------
176:     res : str
177:         Both stdout and stderr messages.
178: 
179:     Notes
180:     -----
181:     On NT, DOS systems the returned status is correct for external commands.
182:     Wild cards will not work for non-posix systems or when use_shell=0.
183: 
184:     '''
185:     log.debug('exec_command(%r,%s)' % (command,\
186:          ','.join(['%s=%r'%kv for kv in env.items()])))
187: 
188:     if use_tee is None:
189:         use_tee = os.name=='posix'
190:     if use_shell is None:
191:         use_shell = os.name=='posix'
192:     execute_in = os.path.abspath(execute_in)
193:     oldcwd = os.path.abspath(os.getcwd())
194: 
195:     if __name__[-12:] == 'exec_command':
196:         exec_dir = os.path.dirname(os.path.abspath(__file__))
197:     elif os.path.isfile('exec_command.py'):
198:         exec_dir = os.path.abspath('.')
199:     else:
200:         exec_dir = os.path.abspath(sys.argv[0])
201:         if os.path.isfile(exec_dir):
202:             exec_dir = os.path.dirname(exec_dir)
203: 
204:     if oldcwd!=execute_in:
205:         os.chdir(execute_in)
206:         log.debug('New cwd: %s' % execute_in)
207:     else:
208:         log.debug('Retaining cwd: %s' % oldcwd)
209: 
210:     oldenv = _preserve_environment( list(env.keys()) )
211:     _update_environment( **env )
212: 
213:     try:
214:         # _exec_command is robust but slow, it relies on
215:         # usable sys.std*.fileno() descriptors. If they
216:         # are bad (like in win32 Idle, PyCrust environments)
217:         # then _exec_command_python (even slower)
218:         # will be used as a last resort.
219:         #
220:         # _exec_command_posix uses os.system and is faster
221:         # but not on all platforms os.system will return
222:         # a correct status.
223:         if (_with_python and _supports_fileno(sys.stdout) and
224:                             sys.stdout.fileno() == -1):
225:             st = _exec_command_python(command,
226:                                       exec_command_dir = exec_dir,
227:                                       **env)
228:         elif os.name=='posix':
229:             st = _exec_command_posix(command,
230:                                      use_shell=use_shell,
231:                                      use_tee=use_tee,
232:                                      **env)
233:         else:
234:             st = _exec_command(command, use_shell=use_shell,
235:                                use_tee=use_tee,**env)
236:     finally:
237:         if oldcwd!=execute_in:
238:             os.chdir(oldcwd)
239:             log.debug('Restored cwd to %s' % oldcwd)
240:         _update_environment(**oldenv)
241: 
242:     return st
243: 
244: def _exec_command_posix( command,
245:                          use_shell = None,
246:                          use_tee = None,
247:                          **env ):
248:     log.debug('_exec_command_posix(...)')
249: 
250:     if is_sequence(command):
251:         command_str = ' '.join(list(command))
252:     else:
253:         command_str = command
254: 
255:     tmpfile = temp_file_name()
256:     stsfile = None
257:     if use_tee:
258:         stsfile = temp_file_name()
259:         filter = ''
260:         if use_tee == 2:
261:             filter = r'| tr -cd "\n" | tr "\n" "."; echo'
262:         command_posix = '( %s ; echo $? > %s ) 2>&1 | tee %s %s'\
263:                       % (command_str, stsfile, tmpfile, filter)
264:     else:
265:         stsfile = temp_file_name()
266:         command_posix = '( %s ; echo $? > %s ) > %s 2>&1'\
267:                         % (command_str, stsfile, tmpfile)
268:         #command_posix = '( %s ) > %s 2>&1' % (command_str,tmpfile)
269: 
270:     log.debug('Running os.system(%r)' % (command_posix))
271:     status = os.system(command_posix)
272: 
273:     if use_tee:
274:         if status:
275:             # if command_tee fails then fall back to robust exec_command
276:             log.warn('_exec_command_posix failed (status=%s)' % status)
277:             return _exec_command(command, use_shell=use_shell, **env)
278: 
279:     if stsfile is not None:
280:         f = open_latin1(stsfile, 'r')
281:         status_text = f.read()
282:         status = int(status_text)
283:         f.close()
284:         os.remove(stsfile)
285: 
286:     f = open_latin1(tmpfile, 'r')
287:     text = f.read()
288:     f.close()
289:     os.remove(tmpfile)
290: 
291:     if text[-1:]=='\n':
292:         text = text[:-1]
293: 
294:     return status, text
295: 
296: 
297: def _exec_command_python(command,
298:                          exec_command_dir='', **env):
299:     log.debug('_exec_command_python(...)')
300: 
301:     python_exe = get_pythonexe()
302:     cmdfile = temp_file_name()
303:     stsfile = temp_file_name()
304:     outfile = temp_file_name()
305: 
306:     f = open(cmdfile, 'w')
307:     f.write('import os\n')
308:     f.write('import sys\n')
309:     f.write('sys.path.insert(0,%r)\n' % (exec_command_dir))
310:     f.write('from exec_command import exec_command\n')
311:     f.write('del sys.path[0]\n')
312:     f.write('cmd = %r\n' % command)
313:     f.write('os.environ = %r\n' % (os.environ))
314:     f.write('s,o = exec_command(cmd, _with_python=0, **%r)\n' % (env))
315:     f.write('f=open(%r,"w")\nf.write(str(s))\nf.close()\n' % (stsfile))
316:     f.write('f=open(%r,"w")\nf.write(o)\nf.close()\n' % (outfile))
317:     f.close()
318: 
319:     cmd = '%s %s' % (python_exe, cmdfile)
320:     status = os.system(cmd)
321:     if status:
322:         raise RuntimeError("%r failed" % (cmd,))
323:     os.remove(cmdfile)
324: 
325:     f = open_latin1(stsfile, 'r')
326:     status = int(f.read())
327:     f.close()
328:     os.remove(stsfile)
329: 
330:     f = open_latin1(outfile, 'r')
331:     text = f.read()
332:     f.close()
333:     os.remove(outfile)
334: 
335:     return status, text
336: 
337: def quote_arg(arg):
338:     if arg[0]!='"' and ' ' in arg:
339:         return '"%s"' % arg
340:     return arg
341: 
342: def _exec_command( command, use_shell=None, use_tee = None, **env ):
343:     log.debug('_exec_command(...)')
344: 
345:     if use_shell is None:
346:         use_shell = os.name=='posix'
347:     if use_tee is None:
348:         use_tee = os.name=='posix'
349:     using_command = 0
350:     if use_shell:
351:         # We use shell (unless use_shell==0) so that wildcards can be
352:         # used.
353:         sh = os.environ.get('SHELL', '/bin/sh')
354:         if is_sequence(command):
355:             argv = [sh, '-c', ' '.join(list(command))]
356:         else:
357:             argv = [sh, '-c', command]
358:     else:
359:         # On NT, DOS we avoid using command.com as it's exit status is
360:         # not related to the exit status of a command.
361:         if is_sequence(command):
362:             argv = command[:]
363:         else:
364:             argv = shlex.split(command)
365: 
366:     if hasattr(os, 'spawnvpe'):
367:         spawn_command = os.spawnvpe
368:     else:
369:         spawn_command = os.spawnve
370:         argv[0] = find_executable(argv[0]) or argv[0]
371:         if not os.path.isfile(argv[0]):
372:             log.warn('Executable %s does not exist' % (argv[0]))
373:             if os.name in ['nt', 'dos']:
374:                 # argv[0] might be internal command
375:                 argv = [os.environ['COMSPEC'], '/C'] + argv
376:                 using_command = 1
377: 
378:     _so_has_fileno = _supports_fileno(sys.stdout)
379:     _se_has_fileno = _supports_fileno(sys.stderr)
380:     so_flush = sys.stdout.flush
381:     se_flush = sys.stderr.flush
382:     if _so_has_fileno:
383:         so_fileno = sys.stdout.fileno()
384:         so_dup = os.dup(so_fileno)
385:     if _se_has_fileno:
386:         se_fileno = sys.stderr.fileno()
387:         se_dup = os.dup(se_fileno)
388: 
389:     outfile = temp_file_name()
390:     fout = open(outfile, 'w')
391:     if using_command:
392:         errfile = temp_file_name()
393:         ferr = open(errfile, 'w')
394: 
395:     log.debug('Running %s(%s,%r,%r,os.environ)' \
396:               % (spawn_command.__name__, os.P_WAIT, argv[0], argv))
397: 
398:     if sys.version_info[0] >= 3 and os.name == 'nt':
399:         # Pre-encode os.environ, discarding un-encodable entries,
400:         # to avoid it failing during encoding as part of spawn. Failure
401:         # is possible if the environment contains entries that are not
402:         # encoded using the system codepage as windows expects.
403:         #
404:         # This is not necessary on unix, where os.environ is encoded
405:         # using the surrogateescape error handler and decoded using
406:         # it as part of spawn.
407:         encoded_environ = {}
408:         for k, v in os.environ.items():
409:             try:
410:                 encoded_environ[k.encode(sys.getfilesystemencoding())] = v.encode(
411:                     sys.getfilesystemencoding())
412:             except UnicodeEncodeError:
413:                 log.debug("ignoring un-encodable env entry %s", k)
414:     else:
415:         encoded_environ = os.environ
416: 
417:     argv0 = argv[0]
418:     if not using_command:
419:         argv[0] = quote_arg(argv0)
420: 
421:     so_flush()
422:     se_flush()
423:     if _so_has_fileno:
424:         os.dup2(fout.fileno(), so_fileno)
425: 
426:     if _se_has_fileno:
427:         if using_command:
428:             #XXX: disabled for now as it does not work from cmd under win32.
429:             #     Tests fail on msys
430:             os.dup2(ferr.fileno(), se_fileno)
431:         else:
432:             os.dup2(fout.fileno(), se_fileno)
433:     try:
434:         status = spawn_command(os.P_WAIT, argv0, argv, encoded_environ)
435:     except Exception:
436:         errmess = str(get_exception())
437:         status = 999
438:         sys.stderr.write('%s: %s'%(errmess, argv[0]))
439: 
440:     so_flush()
441:     se_flush()
442:     if _so_has_fileno:
443:         os.dup2(so_dup, so_fileno)
444:         os.close(so_dup)
445:     if _se_has_fileno:
446:         os.dup2(se_dup, se_fileno)
447:         os.close(se_dup)
448: 
449:     fout.close()
450:     fout = open_latin1(outfile, 'r')
451:     text = fout.read()
452:     fout.close()
453:     os.remove(outfile)
454: 
455:     if using_command:
456:         ferr.close()
457:         ferr = open_latin1(errfile, 'r')
458:         errmess = ferr.read()
459:         ferr.close()
460:         os.remove(errfile)
461:         if errmess and not status:
462:             # Not sure how to handle the case where errmess
463:             # contains only warning messages and that should
464:             # not be treated as errors.
465:             #status = 998
466:             if text:
467:                 text = text + '\n'
468:             #text = '%sCOMMAND %r FAILED: %s' %(text,command,errmess)
469:             text = text + errmess
470:             print (errmess)
471:     if text[-1:]=='\n':
472:         text = text[:-1]
473:     if status is None:
474:         status = 0
475: 
476:     if use_tee:
477:         print (text)
478: 
479:     return status, text
480: 
481: 
482: def test_nt(**kws):
483:     pythonexe = get_pythonexe()
484:     echo = find_executable('echo')
485:     using_cygwin_echo = echo != 'echo'
486:     if using_cygwin_echo:
487:         log.warn('Using cygwin echo in win32 environment is not supported')
488: 
489:         s, o=exec_command(pythonexe\
490:                          +' -c "import os;print os.environ.get(\'AAA\',\'\')"')
491:         assert s==0 and o=='', (s, o)
492: 
493:         s, o=exec_command(pythonexe\
494:                          +' -c "import os;print os.environ.get(\'AAA\')"',
495:                          AAA='Tere')
496:         assert s==0 and o=='Tere', (s, o)
497: 
498:         os.environ['BBB'] = 'Hi'
499:         s, o=exec_command(pythonexe\
500:                          +' -c "import os;print os.environ.get(\'BBB\',\'\')"')
501:         assert s==0 and o=='Hi', (s, o)
502: 
503:         s, o=exec_command(pythonexe\
504:                          +' -c "import os;print os.environ.get(\'BBB\',\'\')"',
505:                          BBB='Hey')
506:         assert s==0 and o=='Hey', (s, o)
507: 
508:         s, o=exec_command(pythonexe\
509:                          +' -c "import os;print os.environ.get(\'BBB\',\'\')"')
510:         assert s==0 and o=='Hi', (s, o)
511:     elif 0:
512:         s, o=exec_command('echo Hello')
513:         assert s==0 and o=='Hello', (s, o)
514: 
515:         s, o=exec_command('echo a%AAA%')
516:         assert s==0 and o=='a', (s, o)
517: 
518:         s, o=exec_command('echo a%AAA%', AAA='Tere')
519:         assert s==0 and o=='aTere', (s, o)
520: 
521:         os.environ['BBB'] = 'Hi'
522:         s, o=exec_command('echo a%BBB%')
523:         assert s==0 and o=='aHi', (s, o)
524: 
525:         s, o=exec_command('echo a%BBB%', BBB='Hey')
526:         assert s==0 and o=='aHey', (s, o)
527:         s, o=exec_command('echo a%BBB%')
528:         assert s==0 and o=='aHi', (s, o)
529: 
530:         s, o=exec_command('this_is_not_a_command')
531:         assert s and o!='', (s, o)
532: 
533:         s, o=exec_command('type not_existing_file')
534:         assert s and o!='', (s, o)
535: 
536:     s, o=exec_command('echo path=%path%')
537:     assert s==0 and o!='', (s, o)
538: 
539:     s, o=exec_command('%s -c "import sys;sys.stderr.write(sys.platform)"' \
540:                      % pythonexe)
541:     assert s==0 and o=='win32', (s, o)
542: 
543:     s, o=exec_command('%s -c "raise \'Ignore me.\'"' % pythonexe)
544:     assert s==1 and o, (s, o)
545: 
546:     s, o=exec_command('%s -c "import sys;sys.stderr.write(\'0\');sys.stderr.write(\'1\');sys.stderr.write(\'2\')"'\
547:                      % pythonexe)
548:     assert s==0 and o=='012', (s, o)
549: 
550:     s, o=exec_command('%s -c "import sys;sys.exit(15)"' % pythonexe)
551:     assert s==15 and o=='', (s, o)
552: 
553:     s, o=exec_command('%s -c "print \'Heipa\'"' % pythonexe)
554:     assert s==0 and o=='Heipa', (s, o)
555: 
556:     print ('ok')
557: 
558: def test_posix(**kws):
559:     s, o=exec_command("echo Hello",**kws)
560:     assert s==0 and o=='Hello', (s, o)
561: 
562:     s, o=exec_command('echo $AAA',**kws)
563:     assert s==0 and o=='', (s, o)
564: 
565:     s, o=exec_command('echo "$AAA"',AAA='Tere',**kws)
566:     assert s==0 and o=='Tere', (s, o)
567: 
568: 
569:     s, o=exec_command('echo "$AAA"',**kws)
570:     assert s==0 and o=='', (s, o)
571: 
572:     os.environ['BBB'] = 'Hi'
573:     s, o=exec_command('echo "$BBB"',**kws)
574:     assert s==0 and o=='Hi', (s, o)
575: 
576:     s, o=exec_command('echo "$BBB"',BBB='Hey',**kws)
577:     assert s==0 and o=='Hey', (s, o)
578: 
579:     s, o=exec_command('echo "$BBB"',**kws)
580:     assert s==0 and o=='Hi', (s, o)
581: 
582: 
583:     s, o=exec_command('this_is_not_a_command',**kws)
584:     assert s!=0 and o!='', (s, o)
585: 
586:     s, o=exec_command('echo path=$PATH',**kws)
587:     assert s==0 and o!='', (s, o)
588: 
589:     s, o=exec_command('python -c "import sys,os;sys.stderr.write(os.name)"',**kws)
590:     assert s==0 and o=='posix', (s, o)
591: 
592:     s, o=exec_command('python -c "raise \'Ignore me.\'"',**kws)
593:     assert s==1 and o, (s, o)
594: 
595:     s, o=exec_command('python -c "import sys;sys.stderr.write(\'0\');sys.stderr.write(\'1\');sys.stderr.write(\'2\')"',**kws)
596:     assert s==0 and o=='012', (s, o)
597: 
598:     s, o=exec_command('python -c "import sys;sys.exit(15)"',**kws)
599:     assert s==15 and o=='', (s, o)
600: 
601:     s, o=exec_command('python -c "print \'Heipa\'"',**kws)
602:     assert s==0 and o=='Heipa', (s, o)
603: 
604:     print ('ok')
605: 
606: def test_execute_in(**kws):
607:     pythonexe = get_pythonexe()
608:     tmpfile = temp_file_name()
609:     fn = os.path.basename(tmpfile)
610:     tmpdir = os.path.dirname(tmpfile)
611:     f = open(tmpfile, 'w')
612:     f.write('Hello')
613:     f.close()
614: 
615:     s, o = exec_command('%s -c "print \'Ignore the following IOError:\','\
616:                        'open(%r,\'r\')"' % (pythonexe, fn),**kws)
617:     assert s and o!='', (s, o)
618:     s, o = exec_command('%s -c "print open(%r,\'r\').read()"' % (pythonexe, fn),
619:                        execute_in = tmpdir,**kws)
620:     assert s==0 and o=='Hello', (s, o)
621:     os.remove(tmpfile)
622:     print ('ok')
623: 
624: def test_svn(**kws):
625:     s, o = exec_command(['svn', 'status'],**kws)
626:     assert s, (s, o)
627:     print ('svn ok')
628: 
629: def test_cl(**kws):
630:     if os.name=='nt':
631:         s, o = exec_command(['cl', '/V'],**kws)
632:         assert s, (s, o)
633:         print ('cl ok')
634: 
635: if os.name=='posix':
636:     test = test_posix
637: elif os.name in ['nt', 'dos']:
638:     test = test_nt
639: else:
640:     raise NotImplementedError('exec_command tests for ', os.name)
641: 
642: ############################################################
643: 
644: if __name__ == "__main__":
645: 
646:     test(use_tee=0)
647:     test(use_tee=1)
648:     test_execute_in(use_tee=0)
649:     test_execute_in(use_tee=1)
650:     test_svn(use_tee=1)
651:     test_cl(use_tee=1)
652: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_32625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, (-1)), 'str', "\nexec_command\n\nImplements exec_command function that is (almost) equivalent to\ncommands.getstatusoutput function but on NT, DOS systems the\nreturned status is actually correct (though, the returned status\nvalues may be different by a factor). In addition, exec_command\ntakes keyword arguments for (re-)defining environment variables.\n\nProvides functions:\n\n  exec_command  --- execute command in a specified directory and\n                    in the modified environment.\n  find_executable --- locate a command using info from environment\n                    variable PATH. Equivalent to posix `which`\n                    command.\n\nAuthor: Pearu Peterson <pearu@cens.ioc.ee>\nCreated: 11 January 2003\n\nRequires: Python 2.x\n\nSuccesfully tested on:\n\n========  ============  =================================================\nos.name   sys.platform  comments\n========  ============  =================================================\nposix     linux2        Debian (sid) Linux, Python 2.1.3+, 2.2.3+, 2.3.3\n                        PyCrust 0.9.3, Idle 1.0.2\nposix     linux2        Red Hat 9 Linux, Python 2.1.3, 2.2.2, 2.3.2\nposix     sunos5        SunOS 5.9, Python 2.2, 2.3.2\nposix     darwin        Darwin 7.2.0, Python 2.3\nnt        win32         Windows Me\n                        Python 2.3(EE), Idle 1.0, PyCrust 0.7.2\n                        Python 2.1.1 Idle 0.8\nnt        win32         Windows 98, Python 2.1.1. Idle 0.8\nnt        win32         Cygwin 98-4.10, Python 2.1.1(MSC) - echo tests\n                        fail i.e. redefining environment variables may\n                        not work. FIXED: don't use cygwin echo!\n                        Comment: also `cmd /c echo` will not work\n                        but redefining environment variables do work.\nposix     cygwin        Cygwin 98-4.10, Python 2.3.3(cygming special)\nnt        win32         Windows XP, Python 2.3.3\n========  ============  =================================================\n\nKnown bugs:\n\n* Tests, that send messages to stderr, fail when executed from MSYS prompt\n  because the messages are lost at some point.\n\n")

# Assigning a List to a Name (line 55):

# Assigning a List to a Name (line 55):
__all__ = ['exec_command', 'find_executable']
module_type_store.set_exportable_members(['exec_command', 'find_executable'])

# Obtaining an instance of the builtin type 'list' (line 55)
list_32626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 55)
# Adding element type (line 55)
str_32627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 11), 'str', 'exec_command')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 10), list_32626, str_32627)
# Adding element type (line 55)
str_32628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 27), 'str', 'find_executable')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 10), list_32626, str_32628)

# Assigning a type to the variable '__all__' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), '__all__', list_32626)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 57, 0))

# 'import os' statement (line 57)
import os

import_module(stypy.reporting.localization.Localization(__file__, 57, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 58, 0))

# 'import sys' statement (line 58)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 58, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 59, 0))

# 'import shlex' statement (line 59)
import shlex

import_module(stypy.reporting.localization.Localization(__file__, 59, 0), 'shlex', shlex, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 61, 0))

# 'from numpy.distutils.misc_util import is_sequence, make_temp_file' statement (line 61)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_32629 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 61, 0), 'numpy.distutils.misc_util')

if (type(import_32629) is not StypyTypeError):

    if (import_32629 != 'pyd_module'):
        __import__(import_32629)
        sys_modules_32630 = sys.modules[import_32629]
        import_from_module(stypy.reporting.localization.Localization(__file__, 61, 0), 'numpy.distutils.misc_util', sys_modules_32630.module_type_store, module_type_store, ['is_sequence', 'make_temp_file'])
        nest_module(stypy.reporting.localization.Localization(__file__, 61, 0), __file__, sys_modules_32630, sys_modules_32630.module_type_store, module_type_store)
    else:
        from numpy.distutils.misc_util import is_sequence, make_temp_file

        import_from_module(stypy.reporting.localization.Localization(__file__, 61, 0), 'numpy.distutils.misc_util', None, module_type_store, ['is_sequence', 'make_temp_file'], [is_sequence, make_temp_file])

else:
    # Assigning a type to the variable 'numpy.distutils.misc_util' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'numpy.distutils.misc_util', import_32629)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 62, 0))

# 'from numpy.distutils import log' statement (line 62)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_32631 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 62, 0), 'numpy.distutils')

if (type(import_32631) is not StypyTypeError):

    if (import_32631 != 'pyd_module'):
        __import__(import_32631)
        sys_modules_32632 = sys.modules[import_32631]
        import_from_module(stypy.reporting.localization.Localization(__file__, 62, 0), 'numpy.distutils', sys_modules_32632.module_type_store, module_type_store, ['log'])
        nest_module(stypy.reporting.localization.Localization(__file__, 62, 0), __file__, sys_modules_32632, sys_modules_32632.module_type_store, module_type_store)
    else:
        from numpy.distutils import log

        import_from_module(stypy.reporting.localization.Localization(__file__, 62, 0), 'numpy.distutils', None, module_type_store, ['log'], [log])

else:
    # Assigning a type to the variable 'numpy.distutils' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'numpy.distutils', import_32631)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 63, 0))

# 'from numpy.distutils.compat import get_exception' statement (line 63)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_32633 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 63, 0), 'numpy.distutils.compat')

if (type(import_32633) is not StypyTypeError):

    if (import_32633 != 'pyd_module'):
        __import__(import_32633)
        sys_modules_32634 = sys.modules[import_32633]
        import_from_module(stypy.reporting.localization.Localization(__file__, 63, 0), 'numpy.distutils.compat', sys_modules_32634.module_type_store, module_type_store, ['get_exception'])
        nest_module(stypy.reporting.localization.Localization(__file__, 63, 0), __file__, sys_modules_32634, sys_modules_32634.module_type_store, module_type_store)
    else:
        from numpy.distutils.compat import get_exception

        import_from_module(stypy.reporting.localization.Localization(__file__, 63, 0), 'numpy.distutils.compat', None, module_type_store, ['get_exception'], [get_exception])

else:
    # Assigning a type to the variable 'numpy.distutils.compat' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'numpy.distutils.compat', import_32633)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 65, 0))

# 'from numpy.compat import open_latin1' statement (line 65)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_32635 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 65, 0), 'numpy.compat')

if (type(import_32635) is not StypyTypeError):

    if (import_32635 != 'pyd_module'):
        __import__(import_32635)
        sys_modules_32636 = sys.modules[import_32635]
        import_from_module(stypy.reporting.localization.Localization(__file__, 65, 0), 'numpy.compat', sys_modules_32636.module_type_store, module_type_store, ['open_latin1'])
        nest_module(stypy.reporting.localization.Localization(__file__, 65, 0), __file__, sys_modules_32636, sys_modules_32636.module_type_store, module_type_store)
    else:
        from numpy.compat import open_latin1

        import_from_module(stypy.reporting.localization.Localization(__file__, 65, 0), 'numpy.compat', None, module_type_store, ['open_latin1'], [open_latin1])

else:
    # Assigning a type to the variable 'numpy.compat' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'numpy.compat', import_32635)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')


@norecursion
def temp_file_name(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'temp_file_name'
    module_type_store = module_type_store.open_function_context('temp_file_name', 67, 0, False)
    
    # Passed parameters checking function
    temp_file_name.stypy_localization = localization
    temp_file_name.stypy_type_of_self = None
    temp_file_name.stypy_type_store = module_type_store
    temp_file_name.stypy_function_name = 'temp_file_name'
    temp_file_name.stypy_param_names_list = []
    temp_file_name.stypy_varargs_param_name = None
    temp_file_name.stypy_kwargs_param_name = None
    temp_file_name.stypy_call_defaults = defaults
    temp_file_name.stypy_call_varargs = varargs
    temp_file_name.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'temp_file_name', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'temp_file_name', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'temp_file_name(...)' code ##################

    
    # Assigning a Call to a Tuple (line 68):
    
    # Assigning a Call to a Name:
    
    # Call to make_temp_file(...): (line 68)
    # Processing the call keyword arguments (line 68)
    kwargs_32638 = {}
    # Getting the type of 'make_temp_file' (line 68)
    make_temp_file_32637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 15), 'make_temp_file', False)
    # Calling make_temp_file(args, kwargs) (line 68)
    make_temp_file_call_result_32639 = invoke(stypy.reporting.localization.Localization(__file__, 68, 15), make_temp_file_32637, *[], **kwargs_32638)
    
    # Assigning a type to the variable 'call_assignment_32505' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'call_assignment_32505', make_temp_file_call_result_32639)
    
    # Assigning a Call to a Name (line 68):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_32642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 4), 'int')
    # Processing the call keyword arguments
    kwargs_32643 = {}
    # Getting the type of 'call_assignment_32505' (line 68)
    call_assignment_32505_32640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'call_assignment_32505', False)
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___32641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 4), call_assignment_32505_32640, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_32644 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___32641, *[int_32642], **kwargs_32643)
    
    # Assigning a type to the variable 'call_assignment_32506' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'call_assignment_32506', getitem___call_result_32644)
    
    # Assigning a Name to a Name (line 68):
    # Getting the type of 'call_assignment_32506' (line 68)
    call_assignment_32506_32645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'call_assignment_32506')
    # Assigning a type to the variable 'fo' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'fo', call_assignment_32506_32645)
    
    # Assigning a Call to a Name (line 68):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_32648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 4), 'int')
    # Processing the call keyword arguments
    kwargs_32649 = {}
    # Getting the type of 'call_assignment_32505' (line 68)
    call_assignment_32505_32646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'call_assignment_32505', False)
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___32647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 4), call_assignment_32505_32646, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_32650 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___32647, *[int_32648], **kwargs_32649)
    
    # Assigning a type to the variable 'call_assignment_32507' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'call_assignment_32507', getitem___call_result_32650)
    
    # Assigning a Name to a Name (line 68):
    # Getting the type of 'call_assignment_32507' (line 68)
    call_assignment_32507_32651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'call_assignment_32507')
    # Assigning a type to the variable 'name' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'name', call_assignment_32507_32651)
    
    # Call to close(...): (line 69)
    # Processing the call keyword arguments (line 69)
    kwargs_32654 = {}
    # Getting the type of 'fo' (line 69)
    fo_32652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'fo', False)
    # Obtaining the member 'close' of a type (line 69)
    close_32653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 4), fo_32652, 'close')
    # Calling close(args, kwargs) (line 69)
    close_call_result_32655 = invoke(stypy.reporting.localization.Localization(__file__, 69, 4), close_32653, *[], **kwargs_32654)
    
    # Getting the type of 'name' (line 70)
    name_32656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 11), 'name')
    # Assigning a type to the variable 'stypy_return_type' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type', name_32656)
    
    # ################# End of 'temp_file_name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'temp_file_name' in the type store
    # Getting the type of 'stypy_return_type' (line 67)
    stypy_return_type_32657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32657)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'temp_file_name'
    return stypy_return_type_32657

# Assigning a type to the variable 'temp_file_name' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'temp_file_name', temp_file_name)

@norecursion
def get_pythonexe(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_pythonexe'
    module_type_store = module_type_store.open_function_context('get_pythonexe', 72, 0, False)
    
    # Passed parameters checking function
    get_pythonexe.stypy_localization = localization
    get_pythonexe.stypy_type_of_self = None
    get_pythonexe.stypy_type_store = module_type_store
    get_pythonexe.stypy_function_name = 'get_pythonexe'
    get_pythonexe.stypy_param_names_list = []
    get_pythonexe.stypy_varargs_param_name = None
    get_pythonexe.stypy_kwargs_param_name = None
    get_pythonexe.stypy_call_defaults = defaults
    get_pythonexe.stypy_call_varargs = varargs
    get_pythonexe.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_pythonexe', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_pythonexe', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_pythonexe(...)' code ##################

    
    # Assigning a Attribute to a Name (line 73):
    
    # Assigning a Attribute to a Name (line 73):
    # Getting the type of 'sys' (line 73)
    sys_32658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'sys')
    # Obtaining the member 'executable' of a type (line 73)
    executable_32659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 16), sys_32658, 'executable')
    # Assigning a type to the variable 'pythonexe' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'pythonexe', executable_32659)
    
    
    # Getting the type of 'os' (line 74)
    os_32660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 7), 'os')
    # Obtaining the member 'name' of a type (line 74)
    name_32661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 7), os_32660, 'name')
    
    # Obtaining an instance of the builtin type 'list' (line 74)
    list_32662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 74)
    # Adding element type (line 74)
    str_32663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 19), 'str', 'nt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 18), list_32662, str_32663)
    # Adding element type (line 74)
    str_32664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 25), 'str', 'dos')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 18), list_32662, str_32664)
    
    # Applying the binary operator 'in' (line 74)
    result_contains_32665 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 7), 'in', name_32661, list_32662)
    
    # Testing the type of an if condition (line 74)
    if_condition_32666 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 4), result_contains_32665)
    # Assigning a type to the variable 'if_condition_32666' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'if_condition_32666', if_condition_32666)
    # SSA begins for if statement (line 74)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 75):
    
    # Assigning a Call to a Name:
    
    # Call to split(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'pythonexe' (line 75)
    pythonexe_32670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 33), 'pythonexe', False)
    # Processing the call keyword arguments (line 75)
    kwargs_32671 = {}
    # Getting the type of 'os' (line 75)
    os_32667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 19), 'os', False)
    # Obtaining the member 'path' of a type (line 75)
    path_32668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 19), os_32667, 'path')
    # Obtaining the member 'split' of a type (line 75)
    split_32669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 19), path_32668, 'split')
    # Calling split(args, kwargs) (line 75)
    split_call_result_32672 = invoke(stypy.reporting.localization.Localization(__file__, 75, 19), split_32669, *[pythonexe_32670], **kwargs_32671)
    
    # Assigning a type to the variable 'call_assignment_32508' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'call_assignment_32508', split_call_result_32672)
    
    # Assigning a Call to a Name (line 75):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_32675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 8), 'int')
    # Processing the call keyword arguments
    kwargs_32676 = {}
    # Getting the type of 'call_assignment_32508' (line 75)
    call_assignment_32508_32673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'call_assignment_32508', False)
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___32674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), call_assignment_32508_32673, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_32677 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___32674, *[int_32675], **kwargs_32676)
    
    # Assigning a type to the variable 'call_assignment_32509' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'call_assignment_32509', getitem___call_result_32677)
    
    # Assigning a Name to a Name (line 75):
    # Getting the type of 'call_assignment_32509' (line 75)
    call_assignment_32509_32678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'call_assignment_32509')
    # Assigning a type to the variable 'fdir' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'fdir', call_assignment_32509_32678)
    
    # Assigning a Call to a Name (line 75):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_32681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 8), 'int')
    # Processing the call keyword arguments
    kwargs_32682 = {}
    # Getting the type of 'call_assignment_32508' (line 75)
    call_assignment_32508_32679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'call_assignment_32508', False)
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___32680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), call_assignment_32508_32679, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_32683 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___32680, *[int_32681], **kwargs_32682)
    
    # Assigning a type to the variable 'call_assignment_32510' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'call_assignment_32510', getitem___call_result_32683)
    
    # Assigning a Name to a Name (line 75):
    # Getting the type of 'call_assignment_32510' (line 75)
    call_assignment_32510_32684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'call_assignment_32510')
    # Assigning a type to the variable 'fn' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 14), 'fn', call_assignment_32510_32684)
    
    # Assigning a Call to a Name (line 76):
    
    # Assigning a Call to a Name (line 76):
    
    # Call to replace(...): (line 76)
    # Processing the call arguments (line 76)
    str_32690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 32), 'str', 'PYTHONW')
    str_32691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 43), 'str', 'PYTHON')
    # Processing the call keyword arguments (line 76)
    kwargs_32692 = {}
    
    # Call to upper(...): (line 76)
    # Processing the call keyword arguments (line 76)
    kwargs_32687 = {}
    # Getting the type of 'fn' (line 76)
    fn_32685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 13), 'fn', False)
    # Obtaining the member 'upper' of a type (line 76)
    upper_32686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 13), fn_32685, 'upper')
    # Calling upper(args, kwargs) (line 76)
    upper_call_result_32688 = invoke(stypy.reporting.localization.Localization(__file__, 76, 13), upper_32686, *[], **kwargs_32687)
    
    # Obtaining the member 'replace' of a type (line 76)
    replace_32689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 13), upper_call_result_32688, 'replace')
    # Calling replace(args, kwargs) (line 76)
    replace_call_result_32693 = invoke(stypy.reporting.localization.Localization(__file__, 76, 13), replace_32689, *[str_32690, str_32691], **kwargs_32692)
    
    # Assigning a type to the variable 'fn' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'fn', replace_call_result_32693)
    
    # Assigning a Call to a Name (line 77):
    
    # Assigning a Call to a Name (line 77):
    
    # Call to join(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'fdir' (line 77)
    fdir_32697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 33), 'fdir', False)
    # Getting the type of 'fn' (line 77)
    fn_32698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 39), 'fn', False)
    # Processing the call keyword arguments (line 77)
    kwargs_32699 = {}
    # Getting the type of 'os' (line 77)
    os_32694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 20), 'os', False)
    # Obtaining the member 'path' of a type (line 77)
    path_32695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 20), os_32694, 'path')
    # Obtaining the member 'join' of a type (line 77)
    join_32696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 20), path_32695, 'join')
    # Calling join(args, kwargs) (line 77)
    join_call_result_32700 = invoke(stypy.reporting.localization.Localization(__file__, 77, 20), join_32696, *[fdir_32697, fn_32698], **kwargs_32699)
    
    # Assigning a type to the variable 'pythonexe' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'pythonexe', join_call_result_32700)
    # Evaluating assert statement condition
    
    # Call to isfile(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'pythonexe' (line 78)
    pythonexe_32704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 30), 'pythonexe', False)
    # Processing the call keyword arguments (line 78)
    kwargs_32705 = {}
    # Getting the type of 'os' (line 78)
    os_32701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 78)
    path_32702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 15), os_32701, 'path')
    # Obtaining the member 'isfile' of a type (line 78)
    isfile_32703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 15), path_32702, 'isfile')
    # Calling isfile(args, kwargs) (line 78)
    isfile_call_result_32706 = invoke(stypy.reporting.localization.Localization(__file__, 78, 15), isfile_32703, *[pythonexe_32704], **kwargs_32705)
    
    # SSA join for if statement (line 74)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'pythonexe' (line 79)
    pythonexe_32707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), 'pythonexe')
    # Assigning a type to the variable 'stypy_return_type' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'stypy_return_type', pythonexe_32707)
    
    # ################# End of 'get_pythonexe(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_pythonexe' in the type store
    # Getting the type of 'stypy_return_type' (line 72)
    stypy_return_type_32708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32708)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_pythonexe'
    return stypy_return_type_32708

# Assigning a type to the variable 'get_pythonexe' (line 72)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'get_pythonexe', get_pythonexe)

@norecursion
def find_executable(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 81)
    None_32709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 30), 'None')
    
    # Obtaining an instance of the builtin type 'dict' (line 81)
    dict_32710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 43), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 81)
    
    defaults = [None_32709, dict_32710]
    # Create a new context for function 'find_executable'
    module_type_store = module_type_store.open_function_context('find_executable', 81, 0, False)
    
    # Passed parameters checking function
    find_executable.stypy_localization = localization
    find_executable.stypy_type_of_self = None
    find_executable.stypy_type_store = module_type_store
    find_executable.stypy_function_name = 'find_executable'
    find_executable.stypy_param_names_list = ['exe', 'path', '_cache']
    find_executable.stypy_varargs_param_name = None
    find_executable.stypy_kwargs_param_name = None
    find_executable.stypy_call_defaults = defaults
    find_executable.stypy_call_varargs = varargs
    find_executable.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'find_executable', ['exe', 'path', '_cache'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'find_executable', localization, ['exe', 'path', '_cache'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'find_executable(...)' code ##################

    str_32711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, (-1)), 'str', 'Return full path of a executable or None.\n\n    Symbolic links are not followed.\n    ')
    
    # Assigning a Tuple to a Name (line 86):
    
    # Assigning a Tuple to a Name (line 86):
    
    # Obtaining an instance of the builtin type 'tuple' (line 86)
    tuple_32712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 86)
    # Adding element type (line 86)
    # Getting the type of 'exe' (line 86)
    exe_32713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 10), 'exe')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 10), tuple_32712, exe_32713)
    # Adding element type (line 86)
    # Getting the type of 'path' (line 86)
    path_32714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'path')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 10), tuple_32712, path_32714)
    
    # Assigning a type to the variable 'key' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'key', tuple_32712)
    
    
    # SSA begins for try-except statement (line 87)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Obtaining the type of the subscript
    # Getting the type of 'key' (line 88)
    key_32715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 22), 'key')
    # Getting the type of '_cache' (line 88)
    _cache_32716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), '_cache')
    # Obtaining the member '__getitem__' of a type (line 88)
    getitem___32717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 15), _cache_32716, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 88)
    subscript_call_result_32718 = invoke(stypy.reporting.localization.Localization(__file__, 88, 15), getitem___32717, key_32715)
    
    # Assigning a type to the variable 'stypy_return_type' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'stypy_return_type', subscript_call_result_32718)
    # SSA branch for the except part of a try statement (line 87)
    # SSA branch for the except 'KeyError' branch of a try statement (line 87)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 87)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to debug(...): (line 91)
    # Processing the call arguments (line 91)
    str_32721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 14), 'str', 'find_executable(%r)')
    # Getting the type of 'exe' (line 91)
    exe_32722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 38), 'exe', False)
    # Applying the binary operator '%' (line 91)
    result_mod_32723 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 14), '%', str_32721, exe_32722)
    
    # Processing the call keyword arguments (line 91)
    kwargs_32724 = {}
    # Getting the type of 'log' (line 91)
    log_32719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'log', False)
    # Obtaining the member 'debug' of a type (line 91)
    debug_32720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 4), log_32719, 'debug')
    # Calling debug(args, kwargs) (line 91)
    debug_call_result_32725 = invoke(stypy.reporting.localization.Localization(__file__, 91, 4), debug_32720, *[result_mod_32723], **kwargs_32724)
    
    
    # Assigning a Name to a Name (line 92):
    
    # Assigning a Name to a Name (line 92):
    # Getting the type of 'exe' (line 92)
    exe_32726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 15), 'exe')
    # Assigning a type to the variable 'orig_exe' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'orig_exe', exe_32726)
    
    # Type idiom detected: calculating its left and rigth part (line 94)
    # Getting the type of 'path' (line 94)
    path_32727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 7), 'path')
    # Getting the type of 'None' (line 94)
    None_32728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 15), 'None')
    
    (may_be_32729, more_types_in_union_32730) = may_be_none(path_32727, None_32728)

    if may_be_32729:

        if more_types_in_union_32730:
            # Runtime conditional SSA (line 94)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 95):
        
        # Assigning a Call to a Name (line 95):
        
        # Call to get(...): (line 95)
        # Processing the call arguments (line 95)
        str_32734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 30), 'str', 'PATH')
        # Getting the type of 'os' (line 95)
        os_32735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 38), 'os', False)
        # Obtaining the member 'defpath' of a type (line 95)
        defpath_32736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 38), os_32735, 'defpath')
        # Processing the call keyword arguments (line 95)
        kwargs_32737 = {}
        # Getting the type of 'os' (line 95)
        os_32731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'os', False)
        # Obtaining the member 'environ' of a type (line 95)
        environ_32732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 15), os_32731, 'environ')
        # Obtaining the member 'get' of a type (line 95)
        get_32733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 15), environ_32732, 'get')
        # Calling get(args, kwargs) (line 95)
        get_call_result_32738 = invoke(stypy.reporting.localization.Localization(__file__, 95, 15), get_32733, *[str_32734, defpath_32736], **kwargs_32737)
        
        # Assigning a type to the variable 'path' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'path', get_call_result_32738)

        if more_types_in_union_32730:
            # SSA join for if statement (line 94)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'os' (line 96)
    os_32739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 7), 'os')
    # Obtaining the member 'name' of a type (line 96)
    name_32740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 7), os_32739, 'name')
    str_32741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 16), 'str', 'posix')
    # Applying the binary operator '==' (line 96)
    result_eq_32742 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 7), '==', name_32740, str_32741)
    
    # Testing the type of an if condition (line 96)
    if_condition_32743 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 4), result_eq_32742)
    # Assigning a type to the variable 'if_condition_32743' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'if_condition_32743', if_condition_32743)
    # SSA begins for if statement (line 96)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 97):
    
    # Assigning a Attribute to a Name (line 97):
    # Getting the type of 'os' (line 97)
    os_32744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 19), 'os')
    # Obtaining the member 'path' of a type (line 97)
    path_32745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 19), os_32744, 'path')
    # Obtaining the member 'realpath' of a type (line 97)
    realpath_32746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 19), path_32745, 'realpath')
    # Assigning a type to the variable 'realpath' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'realpath', realpath_32746)
    # SSA branch for the else part of an if statement (line 96)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Lambda to a Name (line 99):
    
    # Assigning a Lambda to a Name (line 99):

    @norecursion
    def _stypy_temp_lambda_18(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_18'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_18', 99, 19, True)
        # Passed parameters checking function
        _stypy_temp_lambda_18.stypy_localization = localization
        _stypy_temp_lambda_18.stypy_type_of_self = None
        _stypy_temp_lambda_18.stypy_type_store = module_type_store
        _stypy_temp_lambda_18.stypy_function_name = '_stypy_temp_lambda_18'
        _stypy_temp_lambda_18.stypy_param_names_list = ['a']
        _stypy_temp_lambda_18.stypy_varargs_param_name = None
        _stypy_temp_lambda_18.stypy_kwargs_param_name = None
        _stypy_temp_lambda_18.stypy_call_defaults = defaults
        _stypy_temp_lambda_18.stypy_call_varargs = varargs
        _stypy_temp_lambda_18.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_18', ['a'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_18', ['a'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        # Getting the type of 'a' (line 99)
        a_32747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 28), 'a')
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 19), 'stypy_return_type', a_32747)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_18' in the type store
        # Getting the type of 'stypy_return_type' (line 99)
        stypy_return_type_32748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 19), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_32748)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_18'
        return stypy_return_type_32748

    # Assigning a type to the variable '_stypy_temp_lambda_18' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 19), '_stypy_temp_lambda_18', _stypy_temp_lambda_18)
    # Getting the type of '_stypy_temp_lambda_18' (line 99)
    _stypy_temp_lambda_18_32749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 19), '_stypy_temp_lambda_18')
    # Assigning a type to the variable 'realpath' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'realpath', _stypy_temp_lambda_18_32749)
    # SSA join for if statement (line 96)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to startswith(...): (line 101)
    # Processing the call arguments (line 101)
    str_32752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 22), 'str', '"')
    # Processing the call keyword arguments (line 101)
    kwargs_32753 = {}
    # Getting the type of 'exe' (line 101)
    exe_32750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 7), 'exe', False)
    # Obtaining the member 'startswith' of a type (line 101)
    startswith_32751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 7), exe_32750, 'startswith')
    # Calling startswith(args, kwargs) (line 101)
    startswith_call_result_32754 = invoke(stypy.reporting.localization.Localization(__file__, 101, 7), startswith_32751, *[str_32752], **kwargs_32753)
    
    # Testing the type of an if condition (line 101)
    if_condition_32755 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 4), startswith_call_result_32754)
    # Assigning a type to the variable 'if_condition_32755' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'if_condition_32755', if_condition_32755)
    # SSA begins for if statement (line 101)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 102):
    
    # Assigning a Subscript to a Name (line 102):
    
    # Obtaining the type of the subscript
    int_32756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 18), 'int')
    int_32757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 20), 'int')
    slice_32758 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 102, 14), int_32756, int_32757, None)
    # Getting the type of 'exe' (line 102)
    exe_32759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 14), 'exe')
    # Obtaining the member '__getitem__' of a type (line 102)
    getitem___32760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 14), exe_32759, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 102)
    subscript_call_result_32761 = invoke(stypy.reporting.localization.Localization(__file__, 102, 14), getitem___32760, slice_32758)
    
    # Assigning a type to the variable 'exe' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'exe', subscript_call_result_32761)
    # SSA join for if statement (line 101)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 104):
    
    # Assigning a List to a Name (line 104):
    
    # Obtaining an instance of the builtin type 'list' (line 104)
    list_32762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 104)
    # Adding element type (line 104)
    str_32763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 16), 'str', '')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 15), list_32762, str_32763)
    
    # Assigning a type to the variable 'suffixes' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'suffixes', list_32762)
    
    
    # Getting the type of 'os' (line 105)
    os_32764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 7), 'os')
    # Obtaining the member 'name' of a type (line 105)
    name_32765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 7), os_32764, 'name')
    
    # Obtaining an instance of the builtin type 'list' (line 105)
    list_32766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 105)
    # Adding element type (line 105)
    str_32767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 19), 'str', 'nt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 18), list_32766, str_32767)
    # Adding element type (line 105)
    str_32768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 25), 'str', 'dos')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 18), list_32766, str_32768)
    # Adding element type (line 105)
    str_32769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 32), 'str', 'os2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 18), list_32766, str_32769)
    
    # Applying the binary operator 'in' (line 105)
    result_contains_32770 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 7), 'in', name_32765, list_32766)
    
    # Testing the type of an if condition (line 105)
    if_condition_32771 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 4), result_contains_32770)
    # Assigning a type to the variable 'if_condition_32771' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'if_condition_32771', if_condition_32771)
    # SSA begins for if statement (line 105)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 106):
    
    # Assigning a Call to a Name:
    
    # Call to splitext(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'exe' (line 106)
    exe_32775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 35), 'exe', False)
    # Processing the call keyword arguments (line 106)
    kwargs_32776 = {}
    # Getting the type of 'os' (line 106)
    os_32772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 18), 'os', False)
    # Obtaining the member 'path' of a type (line 106)
    path_32773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 18), os_32772, 'path')
    # Obtaining the member 'splitext' of a type (line 106)
    splitext_32774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 18), path_32773, 'splitext')
    # Calling splitext(args, kwargs) (line 106)
    splitext_call_result_32777 = invoke(stypy.reporting.localization.Localization(__file__, 106, 18), splitext_32774, *[exe_32775], **kwargs_32776)
    
    # Assigning a type to the variable 'call_assignment_32511' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'call_assignment_32511', splitext_call_result_32777)
    
    # Assigning a Call to a Name (line 106):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_32780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'int')
    # Processing the call keyword arguments
    kwargs_32781 = {}
    # Getting the type of 'call_assignment_32511' (line 106)
    call_assignment_32511_32778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'call_assignment_32511', False)
    # Obtaining the member '__getitem__' of a type (line 106)
    getitem___32779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), call_assignment_32511_32778, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_32782 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___32779, *[int_32780], **kwargs_32781)
    
    # Assigning a type to the variable 'call_assignment_32512' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'call_assignment_32512', getitem___call_result_32782)
    
    # Assigning a Name to a Name (line 106):
    # Getting the type of 'call_assignment_32512' (line 106)
    call_assignment_32512_32783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'call_assignment_32512')
    # Assigning a type to the variable 'fn' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'fn', call_assignment_32512_32783)
    
    # Assigning a Call to a Name (line 106):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_32786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'int')
    # Processing the call keyword arguments
    kwargs_32787 = {}
    # Getting the type of 'call_assignment_32511' (line 106)
    call_assignment_32511_32784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'call_assignment_32511', False)
    # Obtaining the member '__getitem__' of a type (line 106)
    getitem___32785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), call_assignment_32511_32784, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_32788 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___32785, *[int_32786], **kwargs_32787)
    
    # Assigning a type to the variable 'call_assignment_32513' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'call_assignment_32513', getitem___call_result_32788)
    
    # Assigning a Name to a Name (line 106):
    # Getting the type of 'call_assignment_32513' (line 106)
    call_assignment_32513_32789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'call_assignment_32513')
    # Assigning a type to the variable 'ext' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'ext', call_assignment_32513_32789)
    
    # Assigning a List to a Name (line 107):
    
    # Assigning a List to a Name (line 107):
    
    # Obtaining an instance of the builtin type 'list' (line 107)
    list_32790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 107)
    # Adding element type (line 107)
    str_32791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 26), 'str', '.exe')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 25), list_32790, str_32791)
    # Adding element type (line 107)
    str_32792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 34), 'str', '.com')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 25), list_32790, str_32792)
    # Adding element type (line 107)
    str_32793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 42), 'str', '.bat')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 25), list_32790, str_32793)
    
    # Assigning a type to the variable 'extra_suffixes' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'extra_suffixes', list_32790)
    
    
    
    # Call to lower(...): (line 108)
    # Processing the call keyword arguments (line 108)
    kwargs_32796 = {}
    # Getting the type of 'ext' (line 108)
    ext_32794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 11), 'ext', False)
    # Obtaining the member 'lower' of a type (line 108)
    lower_32795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 11), ext_32794, 'lower')
    # Calling lower(args, kwargs) (line 108)
    lower_call_result_32797 = invoke(stypy.reporting.localization.Localization(__file__, 108, 11), lower_32795, *[], **kwargs_32796)
    
    # Getting the type of 'extra_suffixes' (line 108)
    extra_suffixes_32798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 30), 'extra_suffixes')
    # Applying the binary operator 'notin' (line 108)
    result_contains_32799 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 11), 'notin', lower_call_result_32797, extra_suffixes_32798)
    
    # Testing the type of an if condition (line 108)
    if_condition_32800 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 8), result_contains_32799)
    # Assigning a type to the variable 'if_condition_32800' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'if_condition_32800', if_condition_32800)
    # SSA begins for if statement (line 108)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 109):
    
    # Assigning a Name to a Name (line 109):
    # Getting the type of 'extra_suffixes' (line 109)
    extra_suffixes_32801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 23), 'extra_suffixes')
    # Assigning a type to the variable 'suffixes' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'suffixes', extra_suffixes_32801)
    # SSA join for if statement (line 108)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 105)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isabs(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'exe' (line 111)
    exe_32805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 21), 'exe', False)
    # Processing the call keyword arguments (line 111)
    kwargs_32806 = {}
    # Getting the type of 'os' (line 111)
    os_32802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 7), 'os', False)
    # Obtaining the member 'path' of a type (line 111)
    path_32803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 7), os_32802, 'path')
    # Obtaining the member 'isabs' of a type (line 111)
    isabs_32804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 7), path_32803, 'isabs')
    # Calling isabs(args, kwargs) (line 111)
    isabs_call_result_32807 = invoke(stypy.reporting.localization.Localization(__file__, 111, 7), isabs_32804, *[exe_32805], **kwargs_32806)
    
    # Testing the type of an if condition (line 111)
    if_condition_32808 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 4), isabs_call_result_32807)
    # Assigning a type to the variable 'if_condition_32808' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'if_condition_32808', if_condition_32808)
    # SSA begins for if statement (line 111)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 112):
    
    # Assigning a List to a Name (line 112):
    
    # Obtaining an instance of the builtin type 'list' (line 112)
    list_32809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 112)
    # Adding element type (line 112)
    str_32810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 17), 'str', '')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 16), list_32809, str_32810)
    
    # Assigning a type to the variable 'paths' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'paths', list_32809)
    # SSA branch for the else part of an if statement (line 111)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a ListComp to a Name (line 114):
    
    # Assigning a ListComp to a Name (line 114):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to split(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'os' (line 114)
    os_32819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 57), 'os', False)
    # Obtaining the member 'pathsep' of a type (line 114)
    pathsep_32820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 57), os_32819, 'pathsep')
    # Processing the call keyword arguments (line 114)
    kwargs_32821 = {}
    # Getting the type of 'path' (line 114)
    path_32817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 46), 'path', False)
    # Obtaining the member 'split' of a type (line 114)
    split_32818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 46), path_32817, 'split')
    # Calling split(args, kwargs) (line 114)
    split_call_result_32822 = invoke(stypy.reporting.localization.Localization(__file__, 114, 46), split_32818, *[pathsep_32820], **kwargs_32821)
    
    comprehension_32823 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 18), split_call_result_32822)
    # Assigning a type to the variable 'p' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 18), 'p', comprehension_32823)
    
    # Call to abspath(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'p' (line 114)
    p_32814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 34), 'p', False)
    # Processing the call keyword arguments (line 114)
    kwargs_32815 = {}
    # Getting the type of 'os' (line 114)
    os_32811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 18), 'os', False)
    # Obtaining the member 'path' of a type (line 114)
    path_32812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 18), os_32811, 'path')
    # Obtaining the member 'abspath' of a type (line 114)
    abspath_32813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 18), path_32812, 'abspath')
    # Calling abspath(args, kwargs) (line 114)
    abspath_call_result_32816 = invoke(stypy.reporting.localization.Localization(__file__, 114, 18), abspath_32813, *[p_32814], **kwargs_32815)
    
    list_32824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 18), list_32824, abspath_call_result_32816)
    # Assigning a type to the variable 'paths' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'paths', list_32824)
    # SSA join for if statement (line 111)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'paths' (line 116)
    paths_32825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 'paths')
    # Testing the type of a for loop iterable (line 116)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 116, 4), paths_32825)
    # Getting the type of the for loop variable (line 116)
    for_loop_var_32826 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 116, 4), paths_32825)
    # Assigning a type to the variable 'path' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'path', for_loop_var_32826)
    # SSA begins for a for statement (line 116)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 117):
    
    # Assigning a Call to a Name (line 117):
    
    # Call to join(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'path' (line 117)
    path_32830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 26), 'path', False)
    # Getting the type of 'exe' (line 117)
    exe_32831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 32), 'exe', False)
    # Processing the call keyword arguments (line 117)
    kwargs_32832 = {}
    # Getting the type of 'os' (line 117)
    os_32827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 13), 'os', False)
    # Obtaining the member 'path' of a type (line 117)
    path_32828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 13), os_32827, 'path')
    # Obtaining the member 'join' of a type (line 117)
    join_32829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 13), path_32828, 'join')
    # Calling join(args, kwargs) (line 117)
    join_call_result_32833 = invoke(stypy.reporting.localization.Localization(__file__, 117, 13), join_32829, *[path_32830, exe_32831], **kwargs_32832)
    
    # Assigning a type to the variable 'fn' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'fn', join_call_result_32833)
    
    # Getting the type of 'suffixes' (line 118)
    suffixes_32834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 17), 'suffixes')
    # Testing the type of a for loop iterable (line 118)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 118, 8), suffixes_32834)
    # Getting the type of the for loop variable (line 118)
    for_loop_var_32835 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 118, 8), suffixes_32834)
    # Assigning a type to the variable 's' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 's', for_loop_var_32835)
    # SSA begins for a for statement (line 118)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 119):
    
    # Assigning a BinOp to a Name (line 119):
    # Getting the type of 'fn' (line 119)
    fn_32836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 20), 'fn')
    # Getting the type of 's' (line 119)
    s_32837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 23), 's')
    # Applying the binary operator '+' (line 119)
    result_add_32838 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 20), '+', fn_32836, s_32837)
    
    # Assigning a type to the variable 'f_ext' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'f_ext', result_add_32838)
    
    
    
    # Call to islink(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'f_ext' (line 120)
    f_ext_32842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 34), 'f_ext', False)
    # Processing the call keyword arguments (line 120)
    kwargs_32843 = {}
    # Getting the type of 'os' (line 120)
    os_32839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 19), 'os', False)
    # Obtaining the member 'path' of a type (line 120)
    path_32840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 19), os_32839, 'path')
    # Obtaining the member 'islink' of a type (line 120)
    islink_32841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 19), path_32840, 'islink')
    # Calling islink(args, kwargs) (line 120)
    islink_call_result_32844 = invoke(stypy.reporting.localization.Localization(__file__, 120, 19), islink_32841, *[f_ext_32842], **kwargs_32843)
    
    # Applying the 'not' unary operator (line 120)
    result_not__32845 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 15), 'not', islink_call_result_32844)
    
    # Testing the type of an if condition (line 120)
    if_condition_32846 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 120, 12), result_not__32845)
    # Assigning a type to the variable 'if_condition_32846' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'if_condition_32846', if_condition_32846)
    # SSA begins for if statement (line 120)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 121):
    
    # Assigning a Call to a Name (line 121):
    
    # Call to realpath(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'f_ext' (line 121)
    f_ext_32848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 33), 'f_ext', False)
    # Processing the call keyword arguments (line 121)
    kwargs_32849 = {}
    # Getting the type of 'realpath' (line 121)
    realpath_32847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 24), 'realpath', False)
    # Calling realpath(args, kwargs) (line 121)
    realpath_call_result_32850 = invoke(stypy.reporting.localization.Localization(__file__, 121, 24), realpath_32847, *[f_ext_32848], **kwargs_32849)
    
    # Assigning a type to the variable 'f_ext' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'f_ext', realpath_call_result_32850)
    # SSA join for if statement (line 120)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Call to isfile(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'f_ext' (line 122)
    f_ext_32854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 30), 'f_ext', False)
    # Processing the call keyword arguments (line 122)
    kwargs_32855 = {}
    # Getting the type of 'os' (line 122)
    os_32851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 122)
    path_32852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 15), os_32851, 'path')
    # Obtaining the member 'isfile' of a type (line 122)
    isfile_32853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 15), path_32852, 'isfile')
    # Calling isfile(args, kwargs) (line 122)
    isfile_call_result_32856 = invoke(stypy.reporting.localization.Localization(__file__, 122, 15), isfile_32853, *[f_ext_32854], **kwargs_32855)
    
    
    # Call to access(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'f_ext' (line 122)
    f_ext_32859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 51), 'f_ext', False)
    # Getting the type of 'os' (line 122)
    os_32860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 58), 'os', False)
    # Obtaining the member 'X_OK' of a type (line 122)
    X_OK_32861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 58), os_32860, 'X_OK')
    # Processing the call keyword arguments (line 122)
    kwargs_32862 = {}
    # Getting the type of 'os' (line 122)
    os_32857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 41), 'os', False)
    # Obtaining the member 'access' of a type (line 122)
    access_32858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 41), os_32857, 'access')
    # Calling access(args, kwargs) (line 122)
    access_call_result_32863 = invoke(stypy.reporting.localization.Localization(__file__, 122, 41), access_32858, *[f_ext_32859, X_OK_32861], **kwargs_32862)
    
    # Applying the binary operator 'and' (line 122)
    result_and_keyword_32864 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 15), 'and', isfile_call_result_32856, access_call_result_32863)
    
    # Testing the type of an if condition (line 122)
    if_condition_32865 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 12), result_and_keyword_32864)
    # Assigning a type to the variable 'if_condition_32865' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'if_condition_32865', if_condition_32865)
    # SSA begins for if statement (line 122)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to info(...): (line 123)
    # Processing the call arguments (line 123)
    str_32868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 25), 'str', 'Found executable %s')
    # Getting the type of 'f_ext' (line 123)
    f_ext_32869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 49), 'f_ext', False)
    # Applying the binary operator '%' (line 123)
    result_mod_32870 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 25), '%', str_32868, f_ext_32869)
    
    # Processing the call keyword arguments (line 123)
    kwargs_32871 = {}
    # Getting the type of 'log' (line 123)
    log_32866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'log', False)
    # Obtaining the member 'info' of a type (line 123)
    info_32867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 16), log_32866, 'info')
    # Calling info(args, kwargs) (line 123)
    info_call_result_32872 = invoke(stypy.reporting.localization.Localization(__file__, 123, 16), info_32867, *[result_mod_32870], **kwargs_32871)
    
    
    # Assigning a Name to a Subscript (line 124):
    
    # Assigning a Name to a Subscript (line 124):
    # Getting the type of 'f_ext' (line 124)
    f_ext_32873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 30), 'f_ext')
    # Getting the type of '_cache' (line 124)
    _cache_32874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), '_cache')
    # Getting the type of 'key' (line 124)
    key_32875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 23), 'key')
    # Storing an element on a container (line 124)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 16), _cache_32874, (key_32875, f_ext_32873))
    # Getting the type of 'f_ext' (line 125)
    f_ext_32876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 23), 'f_ext')
    # Assigning a type to the variable 'stypy_return_type' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'stypy_return_type', f_ext_32876)
    # SSA join for if statement (line 122)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to warn(...): (line 127)
    # Processing the call arguments (line 127)
    str_32879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 13), 'str', 'Could not locate executable %s')
    # Getting the type of 'orig_exe' (line 127)
    orig_exe_32880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 48), 'orig_exe', False)
    # Applying the binary operator '%' (line 127)
    result_mod_32881 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 13), '%', str_32879, orig_exe_32880)
    
    # Processing the call keyword arguments (line 127)
    kwargs_32882 = {}
    # Getting the type of 'log' (line 127)
    log_32877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'log', False)
    # Obtaining the member 'warn' of a type (line 127)
    warn_32878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 4), log_32877, 'warn')
    # Calling warn(args, kwargs) (line 127)
    warn_call_result_32883 = invoke(stypy.reporting.localization.Localization(__file__, 127, 4), warn_32878, *[result_mod_32881], **kwargs_32882)
    
    # Getting the type of 'None' (line 128)
    None_32884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 11), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'stypy_return_type', None_32884)
    
    # ################# End of 'find_executable(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'find_executable' in the type store
    # Getting the type of 'stypy_return_type' (line 81)
    stypy_return_type_32885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32885)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'find_executable'
    return stypy_return_type_32885

# Assigning a type to the variable 'find_executable' (line 81)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'find_executable', find_executable)

@norecursion
def _preserve_environment(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_preserve_environment'
    module_type_store = module_type_store.open_function_context('_preserve_environment', 132, 0, False)
    
    # Passed parameters checking function
    _preserve_environment.stypy_localization = localization
    _preserve_environment.stypy_type_of_self = None
    _preserve_environment.stypy_type_store = module_type_store
    _preserve_environment.stypy_function_name = '_preserve_environment'
    _preserve_environment.stypy_param_names_list = ['names']
    _preserve_environment.stypy_varargs_param_name = None
    _preserve_environment.stypy_kwargs_param_name = None
    _preserve_environment.stypy_call_defaults = defaults
    _preserve_environment.stypy_call_varargs = varargs
    _preserve_environment.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_preserve_environment', ['names'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_preserve_environment', localization, ['names'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_preserve_environment(...)' code ##################

    
    # Call to debug(...): (line 133)
    # Processing the call arguments (line 133)
    str_32888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 14), 'str', '_preserve_environment(%r)')
    # Getting the type of 'names' (line 133)
    names_32889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 45), 'names', False)
    # Applying the binary operator '%' (line 133)
    result_mod_32890 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 14), '%', str_32888, names_32889)
    
    # Processing the call keyword arguments (line 133)
    kwargs_32891 = {}
    # Getting the type of 'log' (line 133)
    log_32886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'log', False)
    # Obtaining the member 'debug' of a type (line 133)
    debug_32887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 4), log_32886, 'debug')
    # Calling debug(args, kwargs) (line 133)
    debug_call_result_32892 = invoke(stypy.reporting.localization.Localization(__file__, 133, 4), debug_32887, *[result_mod_32890], **kwargs_32891)
    
    
    # Assigning a Dict to a Name (line 134):
    
    # Assigning a Dict to a Name (line 134):
    
    # Obtaining an instance of the builtin type 'dict' (line 134)
    dict_32893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 10), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 134)
    
    # Assigning a type to the variable 'env' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'env', dict_32893)
    
    # Getting the type of 'names' (line 135)
    names_32894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'names')
    # Testing the type of a for loop iterable (line 135)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 135, 4), names_32894)
    # Getting the type of the for loop variable (line 135)
    for_loop_var_32895 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 135, 4), names_32894)
    # Assigning a type to the variable 'name' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'name', for_loop_var_32895)
    # SSA begins for a for statement (line 135)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 136):
    
    # Assigning a Call to a Subscript (line 136):
    
    # Call to get(...): (line 136)
    # Processing the call arguments (line 136)
    # Getting the type of 'name' (line 136)
    name_32899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 35), 'name', False)
    # Processing the call keyword arguments (line 136)
    kwargs_32900 = {}
    # Getting the type of 'os' (line 136)
    os_32896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 20), 'os', False)
    # Obtaining the member 'environ' of a type (line 136)
    environ_32897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 20), os_32896, 'environ')
    # Obtaining the member 'get' of a type (line 136)
    get_32898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 20), environ_32897, 'get')
    # Calling get(args, kwargs) (line 136)
    get_call_result_32901 = invoke(stypy.reporting.localization.Localization(__file__, 136, 20), get_32898, *[name_32899], **kwargs_32900)
    
    # Getting the type of 'env' (line 136)
    env_32902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'env')
    # Getting the type of 'name' (line 136)
    name_32903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'name')
    # Storing an element on a container (line 136)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 8), env_32902, (name_32903, get_call_result_32901))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'env' (line 137)
    env_32904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 11), 'env')
    # Assigning a type to the variable 'stypy_return_type' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'stypy_return_type', env_32904)
    
    # ################# End of '_preserve_environment(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_preserve_environment' in the type store
    # Getting the type of 'stypy_return_type' (line 132)
    stypy_return_type_32905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32905)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_preserve_environment'
    return stypy_return_type_32905

# Assigning a type to the variable '_preserve_environment' (line 132)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 0), '_preserve_environment', _preserve_environment)

@norecursion
def _update_environment(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_update_environment'
    module_type_store = module_type_store.open_function_context('_update_environment', 139, 0, False)
    
    # Passed parameters checking function
    _update_environment.stypy_localization = localization
    _update_environment.stypy_type_of_self = None
    _update_environment.stypy_type_store = module_type_store
    _update_environment.stypy_function_name = '_update_environment'
    _update_environment.stypy_param_names_list = []
    _update_environment.stypy_varargs_param_name = None
    _update_environment.stypy_kwargs_param_name = 'env'
    _update_environment.stypy_call_defaults = defaults
    _update_environment.stypy_call_varargs = varargs
    _update_environment.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_update_environment', [], None, 'env', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_update_environment', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_update_environment(...)' code ##################

    
    # Call to debug(...): (line 140)
    # Processing the call arguments (line 140)
    str_32908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 14), 'str', '_update_environment(...)')
    # Processing the call keyword arguments (line 140)
    kwargs_32909 = {}
    # Getting the type of 'log' (line 140)
    log_32906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'log', False)
    # Obtaining the member 'debug' of a type (line 140)
    debug_32907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 4), log_32906, 'debug')
    # Calling debug(args, kwargs) (line 140)
    debug_call_result_32910 = invoke(stypy.reporting.localization.Localization(__file__, 140, 4), debug_32907, *[str_32908], **kwargs_32909)
    
    
    
    # Call to items(...): (line 141)
    # Processing the call keyword arguments (line 141)
    kwargs_32913 = {}
    # Getting the type of 'env' (line 141)
    env_32911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 23), 'env', False)
    # Obtaining the member 'items' of a type (line 141)
    items_32912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 23), env_32911, 'items')
    # Calling items(args, kwargs) (line 141)
    items_call_result_32914 = invoke(stypy.reporting.localization.Localization(__file__, 141, 23), items_32912, *[], **kwargs_32913)
    
    # Testing the type of a for loop iterable (line 141)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 141, 4), items_call_result_32914)
    # Getting the type of the for loop variable (line 141)
    for_loop_var_32915 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 141, 4), items_call_result_32914)
    # Assigning a type to the variable 'name' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 4), for_loop_var_32915))
    # Assigning a type to the variable 'value' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 4), for_loop_var_32915))
    # SSA begins for a for statement (line 141)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BoolOp to a Subscript (line 142):
    
    # Assigning a BoolOp to a Subscript (line 142):
    
    # Evaluating a boolean operation
    # Getting the type of 'value' (line 142)
    value_32916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 27), 'value')
    str_32917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 36), 'str', '')
    # Applying the binary operator 'or' (line 142)
    result_or_keyword_32918 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 27), 'or', value_32916, str_32917)
    
    # Getting the type of 'os' (line 142)
    os_32919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'os')
    # Obtaining the member 'environ' of a type (line 142)
    environ_32920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), os_32919, 'environ')
    # Getting the type of 'name' (line 142)
    name_32921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 19), 'name')
    # Storing an element on a container (line 142)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 8), environ_32920, (name_32921, result_or_keyword_32918))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_update_environment(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_update_environment' in the type store
    # Getting the type of 'stypy_return_type' (line 139)
    stypy_return_type_32922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32922)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_update_environment'
    return stypy_return_type_32922

# Assigning a type to the variable '_update_environment' (line 139)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 0), '_update_environment', _update_environment)

@norecursion
def _supports_fileno(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_supports_fileno'
    module_type_store = module_type_store.open_function_context('_supports_fileno', 144, 0, False)
    
    # Passed parameters checking function
    _supports_fileno.stypy_localization = localization
    _supports_fileno.stypy_type_of_self = None
    _supports_fileno.stypy_type_store = module_type_store
    _supports_fileno.stypy_function_name = '_supports_fileno'
    _supports_fileno.stypy_param_names_list = ['stream']
    _supports_fileno.stypy_varargs_param_name = None
    _supports_fileno.stypy_kwargs_param_name = None
    _supports_fileno.stypy_call_defaults = defaults
    _supports_fileno.stypy_call_varargs = varargs
    _supports_fileno.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_supports_fileno', ['stream'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_supports_fileno', localization, ['stream'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_supports_fileno(...)' code ##################

    str_32923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, (-1)), 'str', "\n    Returns True if 'stream' supports the file descriptor and allows fileno().\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 148)
    str_32924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 23), 'str', 'fileno')
    # Getting the type of 'stream' (line 148)
    stream_32925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 15), 'stream')
    
    (may_be_32926, more_types_in_union_32927) = may_provide_member(str_32924, stream_32925)

    if may_be_32926:

        if more_types_in_union_32927:
            # Runtime conditional SSA (line 148)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'stream' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'stream', remove_not_member_provider_from_union(stream_32925, 'fileno'))
        
        
        # SSA begins for try-except statement (line 149)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 150):
        
        # Assigning a Call to a Name (line 150):
        
        # Call to fileno(...): (line 150)
        # Processing the call keyword arguments (line 150)
        kwargs_32930 = {}
        # Getting the type of 'stream' (line 150)
        stream_32928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'stream', False)
        # Obtaining the member 'fileno' of a type (line 150)
        fileno_32929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 16), stream_32928, 'fileno')
        # Calling fileno(args, kwargs) (line 150)
        fileno_call_result_32931 = invoke(stypy.reporting.localization.Localization(__file__, 150, 16), fileno_32929, *[], **kwargs_32930)
        
        # Assigning a type to the variable 'r' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'r', fileno_call_result_32931)
        # Getting the type of 'True' (line 151)
        True_32932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 19), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'stypy_return_type', True_32932)
        # SSA branch for the except part of a try statement (line 149)
        # SSA branch for the except 'IOError' branch of a try statement (line 149)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'False' (line 153)
        False_32933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'stypy_return_type', False_32933)
        # SSA join for try-except statement (line 149)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_32927:
            # Runtime conditional SSA for else branch (line 148)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_32926) or more_types_in_union_32927):
        # Assigning a type to the variable 'stream' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'stream', remove_member_provider_from_union(stream_32925, 'fileno'))
        # Getting the type of 'False' (line 155)
        False_32934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'stypy_return_type', False_32934)

        if (may_be_32926 and more_types_in_union_32927):
            # SSA join for if statement (line 148)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of '_supports_fileno(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_supports_fileno' in the type store
    # Getting the type of 'stypy_return_type' (line 144)
    stypy_return_type_32935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32935)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_supports_fileno'
    return stypy_return_type_32935

# Assigning a type to the variable '_supports_fileno' (line 144)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 0), '_supports_fileno', _supports_fileno)

@norecursion
def exec_command(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_32936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 37), 'str', '')
    # Getting the type of 'None' (line 157)
    None_32937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 51), 'None')
    # Getting the type of 'None' (line 157)
    None_32938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 65), 'None')
    int_32939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 32), 'int')
    defaults = [str_32936, None_32937, None_32938, int_32939]
    # Create a new context for function 'exec_command'
    module_type_store = module_type_store.open_function_context('exec_command', 157, 0, False)
    
    # Passed parameters checking function
    exec_command.stypy_localization = localization
    exec_command.stypy_type_of_self = None
    exec_command.stypy_type_store = module_type_store
    exec_command.stypy_function_name = 'exec_command'
    exec_command.stypy_param_names_list = ['command', 'execute_in', 'use_shell', 'use_tee', '_with_python']
    exec_command.stypy_varargs_param_name = None
    exec_command.stypy_kwargs_param_name = 'env'
    exec_command.stypy_call_defaults = defaults
    exec_command.stypy_call_varargs = varargs
    exec_command.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'exec_command', ['command', 'execute_in', 'use_shell', 'use_tee', '_with_python'], None, 'env', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'exec_command', localization, ['command', 'execute_in', 'use_shell', 'use_tee', '_with_python'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'exec_command(...)' code ##################

    str_32940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, (-1)), 'str', '\n    Return (status,output) of executed command.\n\n    Parameters\n    ----------\n    command : str\n        A concatenated string of executable and arguments.\n    execute_in : str\n        Before running command ``cd execute_in`` and after ``cd -``.\n    use_shell : {bool, None}, optional\n        If True, execute ``sh -c command``. Default None (True)\n    use_tee : {bool, None}, optional\n        If True use tee. Default None (True)\n\n\n    Returns\n    -------\n    res : str\n        Both stdout and stderr messages.\n\n    Notes\n    -----\n    On NT, DOS systems the returned status is correct for external commands.\n    Wild cards will not work for non-posix systems or when use_shell=0.\n\n    ')
    
    # Call to debug(...): (line 185)
    # Processing the call arguments (line 185)
    str_32943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 14), 'str', 'exec_command(%r,%s)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 185)
    tuple_32944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 185)
    # Adding element type (line 185)
    # Getting the type of 'command' (line 185)
    command_32945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 39), 'command', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 39), tuple_32944, command_32945)
    # Adding element type (line 185)
    
    # Call to join(...): (line 186)
    # Processing the call arguments (line 186)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to items(...): (line 186)
    # Processing the call keyword arguments (line 186)
    kwargs_32953 = {}
    # Getting the type of 'env' (line 186)
    env_32951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 40), 'env', False)
    # Obtaining the member 'items' of a type (line 186)
    items_32952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 40), env_32951, 'items')
    # Calling items(args, kwargs) (line 186)
    items_call_result_32954 = invoke(stypy.reporting.localization.Localization(__file__, 186, 40), items_32952, *[], **kwargs_32953)
    
    comprehension_32955 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 19), items_call_result_32954)
    # Assigning a type to the variable 'kv' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 19), 'kv', comprehension_32955)
    str_32948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 19), 'str', '%s=%r')
    # Getting the type of 'kv' (line 186)
    kv_32949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 27), 'kv', False)
    # Applying the binary operator '%' (line 186)
    result_mod_32950 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 19), '%', str_32948, kv_32949)
    
    list_32956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 19), list_32956, result_mod_32950)
    # Processing the call keyword arguments (line 186)
    kwargs_32957 = {}
    str_32946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 9), 'str', ',')
    # Obtaining the member 'join' of a type (line 186)
    join_32947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 9), str_32946, 'join')
    # Calling join(args, kwargs) (line 186)
    join_call_result_32958 = invoke(stypy.reporting.localization.Localization(__file__, 186, 9), join_32947, *[list_32956], **kwargs_32957)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 39), tuple_32944, join_call_result_32958)
    
    # Applying the binary operator '%' (line 185)
    result_mod_32959 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 14), '%', str_32943, tuple_32944)
    
    # Processing the call keyword arguments (line 185)
    kwargs_32960 = {}
    # Getting the type of 'log' (line 185)
    log_32941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'log', False)
    # Obtaining the member 'debug' of a type (line 185)
    debug_32942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 4), log_32941, 'debug')
    # Calling debug(args, kwargs) (line 185)
    debug_call_result_32961 = invoke(stypy.reporting.localization.Localization(__file__, 185, 4), debug_32942, *[result_mod_32959], **kwargs_32960)
    
    
    # Type idiom detected: calculating its left and rigth part (line 188)
    # Getting the type of 'use_tee' (line 188)
    use_tee_32962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 7), 'use_tee')
    # Getting the type of 'None' (line 188)
    None_32963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 18), 'None')
    
    (may_be_32964, more_types_in_union_32965) = may_be_none(use_tee_32962, None_32963)

    if may_be_32964:

        if more_types_in_union_32965:
            # Runtime conditional SSA (line 188)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Compare to a Name (line 189):
        
        # Assigning a Compare to a Name (line 189):
        
        # Getting the type of 'os' (line 189)
        os_32966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 18), 'os')
        # Obtaining the member 'name' of a type (line 189)
        name_32967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 18), os_32966, 'name')
        str_32968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 27), 'str', 'posix')
        # Applying the binary operator '==' (line 189)
        result_eq_32969 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 18), '==', name_32967, str_32968)
        
        # Assigning a type to the variable 'use_tee' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'use_tee', result_eq_32969)

        if more_types_in_union_32965:
            # SSA join for if statement (line 188)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 190)
    # Getting the type of 'use_shell' (line 190)
    use_shell_32970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 7), 'use_shell')
    # Getting the type of 'None' (line 190)
    None_32971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 20), 'None')
    
    (may_be_32972, more_types_in_union_32973) = may_be_none(use_shell_32970, None_32971)

    if may_be_32972:

        if more_types_in_union_32973:
            # Runtime conditional SSA (line 190)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Compare to a Name (line 191):
        
        # Assigning a Compare to a Name (line 191):
        
        # Getting the type of 'os' (line 191)
        os_32974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 20), 'os')
        # Obtaining the member 'name' of a type (line 191)
        name_32975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 20), os_32974, 'name')
        str_32976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 29), 'str', 'posix')
        # Applying the binary operator '==' (line 191)
        result_eq_32977 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 20), '==', name_32975, str_32976)
        
        # Assigning a type to the variable 'use_shell' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'use_shell', result_eq_32977)

        if more_types_in_union_32973:
            # SSA join for if statement (line 190)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 192):
    
    # Assigning a Call to a Name (line 192):
    
    # Call to abspath(...): (line 192)
    # Processing the call arguments (line 192)
    # Getting the type of 'execute_in' (line 192)
    execute_in_32981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 33), 'execute_in', False)
    # Processing the call keyword arguments (line 192)
    kwargs_32982 = {}
    # Getting the type of 'os' (line 192)
    os_32978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 17), 'os', False)
    # Obtaining the member 'path' of a type (line 192)
    path_32979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 17), os_32978, 'path')
    # Obtaining the member 'abspath' of a type (line 192)
    abspath_32980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 17), path_32979, 'abspath')
    # Calling abspath(args, kwargs) (line 192)
    abspath_call_result_32983 = invoke(stypy.reporting.localization.Localization(__file__, 192, 17), abspath_32980, *[execute_in_32981], **kwargs_32982)
    
    # Assigning a type to the variable 'execute_in' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'execute_in', abspath_call_result_32983)
    
    # Assigning a Call to a Name (line 193):
    
    # Assigning a Call to a Name (line 193):
    
    # Call to abspath(...): (line 193)
    # Processing the call arguments (line 193)
    
    # Call to getcwd(...): (line 193)
    # Processing the call keyword arguments (line 193)
    kwargs_32989 = {}
    # Getting the type of 'os' (line 193)
    os_32987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 29), 'os', False)
    # Obtaining the member 'getcwd' of a type (line 193)
    getcwd_32988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 29), os_32987, 'getcwd')
    # Calling getcwd(args, kwargs) (line 193)
    getcwd_call_result_32990 = invoke(stypy.reporting.localization.Localization(__file__, 193, 29), getcwd_32988, *[], **kwargs_32989)
    
    # Processing the call keyword arguments (line 193)
    kwargs_32991 = {}
    # Getting the type of 'os' (line 193)
    os_32984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 13), 'os', False)
    # Obtaining the member 'path' of a type (line 193)
    path_32985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 13), os_32984, 'path')
    # Obtaining the member 'abspath' of a type (line 193)
    abspath_32986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 13), path_32985, 'abspath')
    # Calling abspath(args, kwargs) (line 193)
    abspath_call_result_32992 = invoke(stypy.reporting.localization.Localization(__file__, 193, 13), abspath_32986, *[getcwd_call_result_32990], **kwargs_32991)
    
    # Assigning a type to the variable 'oldcwd' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'oldcwd', abspath_call_result_32992)
    
    
    
    # Obtaining the type of the subscript
    int_32993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 16), 'int')
    slice_32994 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 195, 7), int_32993, None, None)
    # Getting the type of '__name__' (line 195)
    name___32995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 7), '__name__')
    # Obtaining the member '__getitem__' of a type (line 195)
    getitem___32996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 7), name___32995, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 195)
    subscript_call_result_32997 = invoke(stypy.reporting.localization.Localization(__file__, 195, 7), getitem___32996, slice_32994)
    
    str_32998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 25), 'str', 'exec_command')
    # Applying the binary operator '==' (line 195)
    result_eq_32999 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 7), '==', subscript_call_result_32997, str_32998)
    
    # Testing the type of an if condition (line 195)
    if_condition_33000 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 195, 4), result_eq_32999)
    # Assigning a type to the variable 'if_condition_33000' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'if_condition_33000', if_condition_33000)
    # SSA begins for if statement (line 195)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 196):
    
    # Assigning a Call to a Name (line 196):
    
    # Call to dirname(...): (line 196)
    # Processing the call arguments (line 196)
    
    # Call to abspath(...): (line 196)
    # Processing the call arguments (line 196)
    # Getting the type of '__file__' (line 196)
    file___33007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 51), '__file__', False)
    # Processing the call keyword arguments (line 196)
    kwargs_33008 = {}
    # Getting the type of 'os' (line 196)
    os_33004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 35), 'os', False)
    # Obtaining the member 'path' of a type (line 196)
    path_33005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 35), os_33004, 'path')
    # Obtaining the member 'abspath' of a type (line 196)
    abspath_33006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 35), path_33005, 'abspath')
    # Calling abspath(args, kwargs) (line 196)
    abspath_call_result_33009 = invoke(stypy.reporting.localization.Localization(__file__, 196, 35), abspath_33006, *[file___33007], **kwargs_33008)
    
    # Processing the call keyword arguments (line 196)
    kwargs_33010 = {}
    # Getting the type of 'os' (line 196)
    os_33001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 19), 'os', False)
    # Obtaining the member 'path' of a type (line 196)
    path_33002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 19), os_33001, 'path')
    # Obtaining the member 'dirname' of a type (line 196)
    dirname_33003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 19), path_33002, 'dirname')
    # Calling dirname(args, kwargs) (line 196)
    dirname_call_result_33011 = invoke(stypy.reporting.localization.Localization(__file__, 196, 19), dirname_33003, *[abspath_call_result_33009], **kwargs_33010)
    
    # Assigning a type to the variable 'exec_dir' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'exec_dir', dirname_call_result_33011)
    # SSA branch for the else part of an if statement (line 195)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isfile(...): (line 197)
    # Processing the call arguments (line 197)
    str_33015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 24), 'str', 'exec_command.py')
    # Processing the call keyword arguments (line 197)
    kwargs_33016 = {}
    # Getting the type of 'os' (line 197)
    os_33012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 9), 'os', False)
    # Obtaining the member 'path' of a type (line 197)
    path_33013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 9), os_33012, 'path')
    # Obtaining the member 'isfile' of a type (line 197)
    isfile_33014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 9), path_33013, 'isfile')
    # Calling isfile(args, kwargs) (line 197)
    isfile_call_result_33017 = invoke(stypy.reporting.localization.Localization(__file__, 197, 9), isfile_33014, *[str_33015], **kwargs_33016)
    
    # Testing the type of an if condition (line 197)
    if_condition_33018 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 197, 9), isfile_call_result_33017)
    # Assigning a type to the variable 'if_condition_33018' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 9), 'if_condition_33018', if_condition_33018)
    # SSA begins for if statement (line 197)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 198):
    
    # Assigning a Call to a Name (line 198):
    
    # Call to abspath(...): (line 198)
    # Processing the call arguments (line 198)
    str_33022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 35), 'str', '.')
    # Processing the call keyword arguments (line 198)
    kwargs_33023 = {}
    # Getting the type of 'os' (line 198)
    os_33019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 19), 'os', False)
    # Obtaining the member 'path' of a type (line 198)
    path_33020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 19), os_33019, 'path')
    # Obtaining the member 'abspath' of a type (line 198)
    abspath_33021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 19), path_33020, 'abspath')
    # Calling abspath(args, kwargs) (line 198)
    abspath_call_result_33024 = invoke(stypy.reporting.localization.Localization(__file__, 198, 19), abspath_33021, *[str_33022], **kwargs_33023)
    
    # Assigning a type to the variable 'exec_dir' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'exec_dir', abspath_call_result_33024)
    # SSA branch for the else part of an if statement (line 197)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 200):
    
    # Assigning a Call to a Name (line 200):
    
    # Call to abspath(...): (line 200)
    # Processing the call arguments (line 200)
    
    # Obtaining the type of the subscript
    int_33028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 44), 'int')
    # Getting the type of 'sys' (line 200)
    sys_33029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 35), 'sys', False)
    # Obtaining the member 'argv' of a type (line 200)
    argv_33030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 35), sys_33029, 'argv')
    # Obtaining the member '__getitem__' of a type (line 200)
    getitem___33031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 35), argv_33030, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 200)
    subscript_call_result_33032 = invoke(stypy.reporting.localization.Localization(__file__, 200, 35), getitem___33031, int_33028)
    
    # Processing the call keyword arguments (line 200)
    kwargs_33033 = {}
    # Getting the type of 'os' (line 200)
    os_33025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 19), 'os', False)
    # Obtaining the member 'path' of a type (line 200)
    path_33026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 19), os_33025, 'path')
    # Obtaining the member 'abspath' of a type (line 200)
    abspath_33027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 19), path_33026, 'abspath')
    # Calling abspath(args, kwargs) (line 200)
    abspath_call_result_33034 = invoke(stypy.reporting.localization.Localization(__file__, 200, 19), abspath_33027, *[subscript_call_result_33032], **kwargs_33033)
    
    # Assigning a type to the variable 'exec_dir' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'exec_dir', abspath_call_result_33034)
    
    
    # Call to isfile(...): (line 201)
    # Processing the call arguments (line 201)
    # Getting the type of 'exec_dir' (line 201)
    exec_dir_33038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 26), 'exec_dir', False)
    # Processing the call keyword arguments (line 201)
    kwargs_33039 = {}
    # Getting the type of 'os' (line 201)
    os_33035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 201)
    path_33036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 11), os_33035, 'path')
    # Obtaining the member 'isfile' of a type (line 201)
    isfile_33037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 11), path_33036, 'isfile')
    # Calling isfile(args, kwargs) (line 201)
    isfile_call_result_33040 = invoke(stypy.reporting.localization.Localization(__file__, 201, 11), isfile_33037, *[exec_dir_33038], **kwargs_33039)
    
    # Testing the type of an if condition (line 201)
    if_condition_33041 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 8), isfile_call_result_33040)
    # Assigning a type to the variable 'if_condition_33041' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'if_condition_33041', if_condition_33041)
    # SSA begins for if statement (line 201)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 202):
    
    # Assigning a Call to a Name (line 202):
    
    # Call to dirname(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'exec_dir' (line 202)
    exec_dir_33045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 39), 'exec_dir', False)
    # Processing the call keyword arguments (line 202)
    kwargs_33046 = {}
    # Getting the type of 'os' (line 202)
    os_33042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 23), 'os', False)
    # Obtaining the member 'path' of a type (line 202)
    path_33043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 23), os_33042, 'path')
    # Obtaining the member 'dirname' of a type (line 202)
    dirname_33044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 23), path_33043, 'dirname')
    # Calling dirname(args, kwargs) (line 202)
    dirname_call_result_33047 = invoke(stypy.reporting.localization.Localization(__file__, 202, 23), dirname_33044, *[exec_dir_33045], **kwargs_33046)
    
    # Assigning a type to the variable 'exec_dir' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'exec_dir', dirname_call_result_33047)
    # SSA join for if statement (line 201)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 197)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 195)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'oldcwd' (line 204)
    oldcwd_33048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 7), 'oldcwd')
    # Getting the type of 'execute_in' (line 204)
    execute_in_33049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 15), 'execute_in')
    # Applying the binary operator '!=' (line 204)
    result_ne_33050 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 7), '!=', oldcwd_33048, execute_in_33049)
    
    # Testing the type of an if condition (line 204)
    if_condition_33051 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 204, 4), result_ne_33050)
    # Assigning a type to the variable 'if_condition_33051' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'if_condition_33051', if_condition_33051)
    # SSA begins for if statement (line 204)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to chdir(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'execute_in' (line 205)
    execute_in_33054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 17), 'execute_in', False)
    # Processing the call keyword arguments (line 205)
    kwargs_33055 = {}
    # Getting the type of 'os' (line 205)
    os_33052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'os', False)
    # Obtaining the member 'chdir' of a type (line 205)
    chdir_33053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), os_33052, 'chdir')
    # Calling chdir(args, kwargs) (line 205)
    chdir_call_result_33056 = invoke(stypy.reporting.localization.Localization(__file__, 205, 8), chdir_33053, *[execute_in_33054], **kwargs_33055)
    
    
    # Call to debug(...): (line 206)
    # Processing the call arguments (line 206)
    str_33059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 18), 'str', 'New cwd: %s')
    # Getting the type of 'execute_in' (line 206)
    execute_in_33060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 34), 'execute_in', False)
    # Applying the binary operator '%' (line 206)
    result_mod_33061 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 18), '%', str_33059, execute_in_33060)
    
    # Processing the call keyword arguments (line 206)
    kwargs_33062 = {}
    # Getting the type of 'log' (line 206)
    log_33057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'log', False)
    # Obtaining the member 'debug' of a type (line 206)
    debug_33058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), log_33057, 'debug')
    # Calling debug(args, kwargs) (line 206)
    debug_call_result_33063 = invoke(stypy.reporting.localization.Localization(__file__, 206, 8), debug_33058, *[result_mod_33061], **kwargs_33062)
    
    # SSA branch for the else part of an if statement (line 204)
    module_type_store.open_ssa_branch('else')
    
    # Call to debug(...): (line 208)
    # Processing the call arguments (line 208)
    str_33066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 18), 'str', 'Retaining cwd: %s')
    # Getting the type of 'oldcwd' (line 208)
    oldcwd_33067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 40), 'oldcwd', False)
    # Applying the binary operator '%' (line 208)
    result_mod_33068 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 18), '%', str_33066, oldcwd_33067)
    
    # Processing the call keyword arguments (line 208)
    kwargs_33069 = {}
    # Getting the type of 'log' (line 208)
    log_33064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'log', False)
    # Obtaining the member 'debug' of a type (line 208)
    debug_33065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 8), log_33064, 'debug')
    # Calling debug(args, kwargs) (line 208)
    debug_call_result_33070 = invoke(stypy.reporting.localization.Localization(__file__, 208, 8), debug_33065, *[result_mod_33068], **kwargs_33069)
    
    # SSA join for if statement (line 204)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 210):
    
    # Assigning a Call to a Name (line 210):
    
    # Call to _preserve_environment(...): (line 210)
    # Processing the call arguments (line 210)
    
    # Call to list(...): (line 210)
    # Processing the call arguments (line 210)
    
    # Call to keys(...): (line 210)
    # Processing the call keyword arguments (line 210)
    kwargs_33075 = {}
    # Getting the type of 'env' (line 210)
    env_33073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 41), 'env', False)
    # Obtaining the member 'keys' of a type (line 210)
    keys_33074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 41), env_33073, 'keys')
    # Calling keys(args, kwargs) (line 210)
    keys_call_result_33076 = invoke(stypy.reporting.localization.Localization(__file__, 210, 41), keys_33074, *[], **kwargs_33075)
    
    # Processing the call keyword arguments (line 210)
    kwargs_33077 = {}
    # Getting the type of 'list' (line 210)
    list_33072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 36), 'list', False)
    # Calling list(args, kwargs) (line 210)
    list_call_result_33078 = invoke(stypy.reporting.localization.Localization(__file__, 210, 36), list_33072, *[keys_call_result_33076], **kwargs_33077)
    
    # Processing the call keyword arguments (line 210)
    kwargs_33079 = {}
    # Getting the type of '_preserve_environment' (line 210)
    _preserve_environment_33071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 13), '_preserve_environment', False)
    # Calling _preserve_environment(args, kwargs) (line 210)
    _preserve_environment_call_result_33080 = invoke(stypy.reporting.localization.Localization(__file__, 210, 13), _preserve_environment_33071, *[list_call_result_33078], **kwargs_33079)
    
    # Assigning a type to the variable 'oldenv' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'oldenv', _preserve_environment_call_result_33080)
    
    # Call to _update_environment(...): (line 211)
    # Processing the call keyword arguments (line 211)
    # Getting the type of 'env' (line 211)
    env_33082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 27), 'env', False)
    kwargs_33083 = {'env_33082': env_33082}
    # Getting the type of '_update_environment' (line 211)
    _update_environment_33081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), '_update_environment', False)
    # Calling _update_environment(args, kwargs) (line 211)
    _update_environment_call_result_33084 = invoke(stypy.reporting.localization.Localization(__file__, 211, 4), _update_environment_33081, *[], **kwargs_33083)
    
    
    # Try-finally block (line 213)
    
    
    # Evaluating a boolean operation
    # Getting the type of '_with_python' (line 223)
    _with_python_33085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), '_with_python')
    
    # Call to _supports_fileno(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 'sys' (line 223)
    sys_33087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 46), 'sys', False)
    # Obtaining the member 'stdout' of a type (line 223)
    stdout_33088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 46), sys_33087, 'stdout')
    # Processing the call keyword arguments (line 223)
    kwargs_33089 = {}
    # Getting the type of '_supports_fileno' (line 223)
    _supports_fileno_33086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 29), '_supports_fileno', False)
    # Calling _supports_fileno(args, kwargs) (line 223)
    _supports_fileno_call_result_33090 = invoke(stypy.reporting.localization.Localization(__file__, 223, 29), _supports_fileno_33086, *[stdout_33088], **kwargs_33089)
    
    # Applying the binary operator 'and' (line 223)
    result_and_keyword_33091 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 12), 'and', _with_python_33085, _supports_fileno_call_result_33090)
    
    
    # Call to fileno(...): (line 224)
    # Processing the call keyword arguments (line 224)
    kwargs_33095 = {}
    # Getting the type of 'sys' (line 224)
    sys_33092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 28), 'sys', False)
    # Obtaining the member 'stdout' of a type (line 224)
    stdout_33093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 28), sys_33092, 'stdout')
    # Obtaining the member 'fileno' of a type (line 224)
    fileno_33094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 28), stdout_33093, 'fileno')
    # Calling fileno(args, kwargs) (line 224)
    fileno_call_result_33096 = invoke(stypy.reporting.localization.Localization(__file__, 224, 28), fileno_33094, *[], **kwargs_33095)
    
    int_33097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 51), 'int')
    # Applying the binary operator '==' (line 224)
    result_eq_33098 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 28), '==', fileno_call_result_33096, int_33097)
    
    # Applying the binary operator 'and' (line 223)
    result_and_keyword_33099 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 12), 'and', result_and_keyword_33091, result_eq_33098)
    
    # Testing the type of an if condition (line 223)
    if_condition_33100 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 8), result_and_keyword_33099)
    # Assigning a type to the variable 'if_condition_33100' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'if_condition_33100', if_condition_33100)
    # SSA begins for if statement (line 223)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 225):
    
    # Assigning a Call to a Name (line 225):
    
    # Call to _exec_command_python(...): (line 225)
    # Processing the call arguments (line 225)
    # Getting the type of 'command' (line 225)
    command_33102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 38), 'command', False)
    # Processing the call keyword arguments (line 225)
    # Getting the type of 'exec_dir' (line 226)
    exec_dir_33103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 57), 'exec_dir', False)
    keyword_33104 = exec_dir_33103
    # Getting the type of 'env' (line 227)
    env_33105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 40), 'env', False)
    kwargs_33106 = {'exec_command_dir': keyword_33104, 'env_33105': env_33105}
    # Getting the type of '_exec_command_python' (line 225)
    _exec_command_python_33101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 17), '_exec_command_python', False)
    # Calling _exec_command_python(args, kwargs) (line 225)
    _exec_command_python_call_result_33107 = invoke(stypy.reporting.localization.Localization(__file__, 225, 17), _exec_command_python_33101, *[command_33102], **kwargs_33106)
    
    # Assigning a type to the variable 'st' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'st', _exec_command_python_call_result_33107)
    # SSA branch for the else part of an if statement (line 223)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'os' (line 228)
    os_33108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 13), 'os')
    # Obtaining the member 'name' of a type (line 228)
    name_33109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 13), os_33108, 'name')
    str_33110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 22), 'str', 'posix')
    # Applying the binary operator '==' (line 228)
    result_eq_33111 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 13), '==', name_33109, str_33110)
    
    # Testing the type of an if condition (line 228)
    if_condition_33112 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 228, 13), result_eq_33111)
    # Assigning a type to the variable 'if_condition_33112' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 13), 'if_condition_33112', if_condition_33112)
    # SSA begins for if statement (line 228)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 229):
    
    # Assigning a Call to a Name (line 229):
    
    # Call to _exec_command_posix(...): (line 229)
    # Processing the call arguments (line 229)
    # Getting the type of 'command' (line 229)
    command_33114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 37), 'command', False)
    # Processing the call keyword arguments (line 229)
    # Getting the type of 'use_shell' (line 230)
    use_shell_33115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 47), 'use_shell', False)
    keyword_33116 = use_shell_33115
    # Getting the type of 'use_tee' (line 231)
    use_tee_33117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 45), 'use_tee', False)
    keyword_33118 = use_tee_33117
    # Getting the type of 'env' (line 232)
    env_33119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 39), 'env', False)
    kwargs_33120 = {'use_shell': keyword_33116, 'env_33119': env_33119, 'use_tee': keyword_33118}
    # Getting the type of '_exec_command_posix' (line 229)
    _exec_command_posix_33113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 17), '_exec_command_posix', False)
    # Calling _exec_command_posix(args, kwargs) (line 229)
    _exec_command_posix_call_result_33121 = invoke(stypy.reporting.localization.Localization(__file__, 229, 17), _exec_command_posix_33113, *[command_33114], **kwargs_33120)
    
    # Assigning a type to the variable 'st' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'st', _exec_command_posix_call_result_33121)
    # SSA branch for the else part of an if statement (line 228)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 234):
    
    # Assigning a Call to a Name (line 234):
    
    # Call to _exec_command(...): (line 234)
    # Processing the call arguments (line 234)
    # Getting the type of 'command' (line 234)
    command_33123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 31), 'command', False)
    # Processing the call keyword arguments (line 234)
    # Getting the type of 'use_shell' (line 234)
    use_shell_33124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 50), 'use_shell', False)
    keyword_33125 = use_shell_33124
    # Getting the type of 'use_tee' (line 235)
    use_tee_33126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 39), 'use_tee', False)
    keyword_33127 = use_tee_33126
    # Getting the type of 'env' (line 235)
    env_33128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 49), 'env', False)
    kwargs_33129 = {'use_shell': keyword_33125, 'env_33128': env_33128, 'use_tee': keyword_33127}
    # Getting the type of '_exec_command' (line 234)
    _exec_command_33122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 17), '_exec_command', False)
    # Calling _exec_command(args, kwargs) (line 234)
    _exec_command_call_result_33130 = invoke(stypy.reporting.localization.Localization(__file__, 234, 17), _exec_command_33122, *[command_33123], **kwargs_33129)
    
    # Assigning a type to the variable 'st' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'st', _exec_command_call_result_33130)
    # SSA join for if statement (line 228)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 223)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # finally branch of the try-finally block (line 213)
    
    
    # Getting the type of 'oldcwd' (line 237)
    oldcwd_33131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 11), 'oldcwd')
    # Getting the type of 'execute_in' (line 237)
    execute_in_33132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 19), 'execute_in')
    # Applying the binary operator '!=' (line 237)
    result_ne_33133 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 11), '!=', oldcwd_33131, execute_in_33132)
    
    # Testing the type of an if condition (line 237)
    if_condition_33134 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 237, 8), result_ne_33133)
    # Assigning a type to the variable 'if_condition_33134' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'if_condition_33134', if_condition_33134)
    # SSA begins for if statement (line 237)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to chdir(...): (line 238)
    # Processing the call arguments (line 238)
    # Getting the type of 'oldcwd' (line 238)
    oldcwd_33137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 21), 'oldcwd', False)
    # Processing the call keyword arguments (line 238)
    kwargs_33138 = {}
    # Getting the type of 'os' (line 238)
    os_33135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'os', False)
    # Obtaining the member 'chdir' of a type (line 238)
    chdir_33136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 12), os_33135, 'chdir')
    # Calling chdir(args, kwargs) (line 238)
    chdir_call_result_33139 = invoke(stypy.reporting.localization.Localization(__file__, 238, 12), chdir_33136, *[oldcwd_33137], **kwargs_33138)
    
    
    # Call to debug(...): (line 239)
    # Processing the call arguments (line 239)
    str_33142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 22), 'str', 'Restored cwd to %s')
    # Getting the type of 'oldcwd' (line 239)
    oldcwd_33143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 45), 'oldcwd', False)
    # Applying the binary operator '%' (line 239)
    result_mod_33144 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 22), '%', str_33142, oldcwd_33143)
    
    # Processing the call keyword arguments (line 239)
    kwargs_33145 = {}
    # Getting the type of 'log' (line 239)
    log_33140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'log', False)
    # Obtaining the member 'debug' of a type (line 239)
    debug_33141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 12), log_33140, 'debug')
    # Calling debug(args, kwargs) (line 239)
    debug_call_result_33146 = invoke(stypy.reporting.localization.Localization(__file__, 239, 12), debug_33141, *[result_mod_33144], **kwargs_33145)
    
    # SSA join for if statement (line 237)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _update_environment(...): (line 240)
    # Processing the call keyword arguments (line 240)
    # Getting the type of 'oldenv' (line 240)
    oldenv_33148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 30), 'oldenv', False)
    kwargs_33149 = {'oldenv_33148': oldenv_33148}
    # Getting the type of '_update_environment' (line 240)
    _update_environment_33147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), '_update_environment', False)
    # Calling _update_environment(args, kwargs) (line 240)
    _update_environment_call_result_33150 = invoke(stypy.reporting.localization.Localization(__file__, 240, 8), _update_environment_33147, *[], **kwargs_33149)
    
    
    # Getting the type of 'st' (line 242)
    st_33151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 11), 'st')
    # Assigning a type to the variable 'stypy_return_type' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'stypy_return_type', st_33151)
    
    # ################# End of 'exec_command(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'exec_command' in the type store
    # Getting the type of 'stypy_return_type' (line 157)
    stypy_return_type_33152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_33152)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'exec_command'
    return stypy_return_type_33152

# Assigning a type to the variable 'exec_command' (line 157)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'exec_command', exec_command)

@norecursion
def _exec_command_posix(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 245)
    None_33153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 37), 'None')
    # Getting the type of 'None' (line 246)
    None_33154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 35), 'None')
    defaults = [None_33153, None_33154]
    # Create a new context for function '_exec_command_posix'
    module_type_store = module_type_store.open_function_context('_exec_command_posix', 244, 0, False)
    
    # Passed parameters checking function
    _exec_command_posix.stypy_localization = localization
    _exec_command_posix.stypy_type_of_self = None
    _exec_command_posix.stypy_type_store = module_type_store
    _exec_command_posix.stypy_function_name = '_exec_command_posix'
    _exec_command_posix.stypy_param_names_list = ['command', 'use_shell', 'use_tee']
    _exec_command_posix.stypy_varargs_param_name = None
    _exec_command_posix.stypy_kwargs_param_name = 'env'
    _exec_command_posix.stypy_call_defaults = defaults
    _exec_command_posix.stypy_call_varargs = varargs
    _exec_command_posix.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_exec_command_posix', ['command', 'use_shell', 'use_tee'], None, 'env', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_exec_command_posix', localization, ['command', 'use_shell', 'use_tee'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_exec_command_posix(...)' code ##################

    
    # Call to debug(...): (line 248)
    # Processing the call arguments (line 248)
    str_33157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 14), 'str', '_exec_command_posix(...)')
    # Processing the call keyword arguments (line 248)
    kwargs_33158 = {}
    # Getting the type of 'log' (line 248)
    log_33155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'log', False)
    # Obtaining the member 'debug' of a type (line 248)
    debug_33156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 4), log_33155, 'debug')
    # Calling debug(args, kwargs) (line 248)
    debug_call_result_33159 = invoke(stypy.reporting.localization.Localization(__file__, 248, 4), debug_33156, *[str_33157], **kwargs_33158)
    
    
    
    # Call to is_sequence(...): (line 250)
    # Processing the call arguments (line 250)
    # Getting the type of 'command' (line 250)
    command_33161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 19), 'command', False)
    # Processing the call keyword arguments (line 250)
    kwargs_33162 = {}
    # Getting the type of 'is_sequence' (line 250)
    is_sequence_33160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 7), 'is_sequence', False)
    # Calling is_sequence(args, kwargs) (line 250)
    is_sequence_call_result_33163 = invoke(stypy.reporting.localization.Localization(__file__, 250, 7), is_sequence_33160, *[command_33161], **kwargs_33162)
    
    # Testing the type of an if condition (line 250)
    if_condition_33164 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 250, 4), is_sequence_call_result_33163)
    # Assigning a type to the variable 'if_condition_33164' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'if_condition_33164', if_condition_33164)
    # SSA begins for if statement (line 250)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 251):
    
    # Assigning a Call to a Name (line 251):
    
    # Call to join(...): (line 251)
    # Processing the call arguments (line 251)
    
    # Call to list(...): (line 251)
    # Processing the call arguments (line 251)
    # Getting the type of 'command' (line 251)
    command_33168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 36), 'command', False)
    # Processing the call keyword arguments (line 251)
    kwargs_33169 = {}
    # Getting the type of 'list' (line 251)
    list_33167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 31), 'list', False)
    # Calling list(args, kwargs) (line 251)
    list_call_result_33170 = invoke(stypy.reporting.localization.Localization(__file__, 251, 31), list_33167, *[command_33168], **kwargs_33169)
    
    # Processing the call keyword arguments (line 251)
    kwargs_33171 = {}
    str_33165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 22), 'str', ' ')
    # Obtaining the member 'join' of a type (line 251)
    join_33166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 22), str_33165, 'join')
    # Calling join(args, kwargs) (line 251)
    join_call_result_33172 = invoke(stypy.reporting.localization.Localization(__file__, 251, 22), join_33166, *[list_call_result_33170], **kwargs_33171)
    
    # Assigning a type to the variable 'command_str' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'command_str', join_call_result_33172)
    # SSA branch for the else part of an if statement (line 250)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 253):
    
    # Assigning a Name to a Name (line 253):
    # Getting the type of 'command' (line 253)
    command_33173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 22), 'command')
    # Assigning a type to the variable 'command_str' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'command_str', command_33173)
    # SSA join for if statement (line 250)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 255):
    
    # Assigning a Call to a Name (line 255):
    
    # Call to temp_file_name(...): (line 255)
    # Processing the call keyword arguments (line 255)
    kwargs_33175 = {}
    # Getting the type of 'temp_file_name' (line 255)
    temp_file_name_33174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 14), 'temp_file_name', False)
    # Calling temp_file_name(args, kwargs) (line 255)
    temp_file_name_call_result_33176 = invoke(stypy.reporting.localization.Localization(__file__, 255, 14), temp_file_name_33174, *[], **kwargs_33175)
    
    # Assigning a type to the variable 'tmpfile' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'tmpfile', temp_file_name_call_result_33176)
    
    # Assigning a Name to a Name (line 256):
    
    # Assigning a Name to a Name (line 256):
    # Getting the type of 'None' (line 256)
    None_33177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 14), 'None')
    # Assigning a type to the variable 'stsfile' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'stsfile', None_33177)
    
    # Getting the type of 'use_tee' (line 257)
    use_tee_33178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 7), 'use_tee')
    # Testing the type of an if condition (line 257)
    if_condition_33179 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 257, 4), use_tee_33178)
    # Assigning a type to the variable 'if_condition_33179' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'if_condition_33179', if_condition_33179)
    # SSA begins for if statement (line 257)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 258):
    
    # Assigning a Call to a Name (line 258):
    
    # Call to temp_file_name(...): (line 258)
    # Processing the call keyword arguments (line 258)
    kwargs_33181 = {}
    # Getting the type of 'temp_file_name' (line 258)
    temp_file_name_33180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 18), 'temp_file_name', False)
    # Calling temp_file_name(args, kwargs) (line 258)
    temp_file_name_call_result_33182 = invoke(stypy.reporting.localization.Localization(__file__, 258, 18), temp_file_name_33180, *[], **kwargs_33181)
    
    # Assigning a type to the variable 'stsfile' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'stsfile', temp_file_name_call_result_33182)
    
    # Assigning a Str to a Name (line 259):
    
    # Assigning a Str to a Name (line 259):
    str_33183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 17), 'str', '')
    # Assigning a type to the variable 'filter' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'filter', str_33183)
    
    
    # Getting the type of 'use_tee' (line 260)
    use_tee_33184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 11), 'use_tee')
    int_33185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 22), 'int')
    # Applying the binary operator '==' (line 260)
    result_eq_33186 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 11), '==', use_tee_33184, int_33185)
    
    # Testing the type of an if condition (line 260)
    if_condition_33187 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 260, 8), result_eq_33186)
    # Assigning a type to the variable 'if_condition_33187' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'if_condition_33187', if_condition_33187)
    # SSA begins for if statement (line 260)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 261):
    
    # Assigning a Str to a Name (line 261):
    str_33188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 21), 'str', '| tr -cd "\\n" | tr "\\n" "."; echo')
    # Assigning a type to the variable 'filter' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 12), 'filter', str_33188)
    # SSA join for if statement (line 260)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 262):
    
    # Assigning a BinOp to a Name (line 262):
    str_33189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 24), 'str', '( %s ; echo $? > %s ) 2>&1 | tee %s %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 263)
    tuple_33190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 263)
    # Adding element type (line 263)
    # Getting the type of 'command_str' (line 263)
    command_str_33191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 25), 'command_str')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 25), tuple_33190, command_str_33191)
    # Adding element type (line 263)
    # Getting the type of 'stsfile' (line 263)
    stsfile_33192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 38), 'stsfile')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 25), tuple_33190, stsfile_33192)
    # Adding element type (line 263)
    # Getting the type of 'tmpfile' (line 263)
    tmpfile_33193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 47), 'tmpfile')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 25), tuple_33190, tmpfile_33193)
    # Adding element type (line 263)
    # Getting the type of 'filter' (line 263)
    filter_33194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 56), 'filter')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 25), tuple_33190, filter_33194)
    
    # Applying the binary operator '%' (line 262)
    result_mod_33195 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 24), '%', str_33189, tuple_33190)
    
    # Assigning a type to the variable 'command_posix' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'command_posix', result_mod_33195)
    # SSA branch for the else part of an if statement (line 257)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 265):
    
    # Assigning a Call to a Name (line 265):
    
    # Call to temp_file_name(...): (line 265)
    # Processing the call keyword arguments (line 265)
    kwargs_33197 = {}
    # Getting the type of 'temp_file_name' (line 265)
    temp_file_name_33196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 18), 'temp_file_name', False)
    # Calling temp_file_name(args, kwargs) (line 265)
    temp_file_name_call_result_33198 = invoke(stypy.reporting.localization.Localization(__file__, 265, 18), temp_file_name_33196, *[], **kwargs_33197)
    
    # Assigning a type to the variable 'stsfile' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'stsfile', temp_file_name_call_result_33198)
    
    # Assigning a BinOp to a Name (line 266):
    
    # Assigning a BinOp to a Name (line 266):
    str_33199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 24), 'str', '( %s ; echo $? > %s ) > %s 2>&1')
    
    # Obtaining an instance of the builtin type 'tuple' (line 267)
    tuple_33200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 267)
    # Adding element type (line 267)
    # Getting the type of 'command_str' (line 267)
    command_str_33201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 27), 'command_str')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 27), tuple_33200, command_str_33201)
    # Adding element type (line 267)
    # Getting the type of 'stsfile' (line 267)
    stsfile_33202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 40), 'stsfile')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 27), tuple_33200, stsfile_33202)
    # Adding element type (line 267)
    # Getting the type of 'tmpfile' (line 267)
    tmpfile_33203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 49), 'tmpfile')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 27), tuple_33200, tmpfile_33203)
    
    # Applying the binary operator '%' (line 266)
    result_mod_33204 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 24), '%', str_33199, tuple_33200)
    
    # Assigning a type to the variable 'command_posix' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'command_posix', result_mod_33204)
    # SSA join for if statement (line 257)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to debug(...): (line 270)
    # Processing the call arguments (line 270)
    str_33207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 14), 'str', 'Running os.system(%r)')
    # Getting the type of 'command_posix' (line 270)
    command_posix_33208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 41), 'command_posix', False)
    # Applying the binary operator '%' (line 270)
    result_mod_33209 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 14), '%', str_33207, command_posix_33208)
    
    # Processing the call keyword arguments (line 270)
    kwargs_33210 = {}
    # Getting the type of 'log' (line 270)
    log_33205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'log', False)
    # Obtaining the member 'debug' of a type (line 270)
    debug_33206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 4), log_33205, 'debug')
    # Calling debug(args, kwargs) (line 270)
    debug_call_result_33211 = invoke(stypy.reporting.localization.Localization(__file__, 270, 4), debug_33206, *[result_mod_33209], **kwargs_33210)
    
    
    # Assigning a Call to a Name (line 271):
    
    # Assigning a Call to a Name (line 271):
    
    # Call to system(...): (line 271)
    # Processing the call arguments (line 271)
    # Getting the type of 'command_posix' (line 271)
    command_posix_33214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 23), 'command_posix', False)
    # Processing the call keyword arguments (line 271)
    kwargs_33215 = {}
    # Getting the type of 'os' (line 271)
    os_33212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 13), 'os', False)
    # Obtaining the member 'system' of a type (line 271)
    system_33213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 13), os_33212, 'system')
    # Calling system(args, kwargs) (line 271)
    system_call_result_33216 = invoke(stypy.reporting.localization.Localization(__file__, 271, 13), system_33213, *[command_posix_33214], **kwargs_33215)
    
    # Assigning a type to the variable 'status' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'status', system_call_result_33216)
    
    # Getting the type of 'use_tee' (line 273)
    use_tee_33217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 7), 'use_tee')
    # Testing the type of an if condition (line 273)
    if_condition_33218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 4), use_tee_33217)
    # Assigning a type to the variable 'if_condition_33218' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'if_condition_33218', if_condition_33218)
    # SSA begins for if statement (line 273)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'status' (line 274)
    status_33219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 11), 'status')
    # Testing the type of an if condition (line 274)
    if_condition_33220 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 274, 8), status_33219)
    # Assigning a type to the variable 'if_condition_33220' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'if_condition_33220', if_condition_33220)
    # SSA begins for if statement (line 274)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 276)
    # Processing the call arguments (line 276)
    str_33223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 21), 'str', '_exec_command_posix failed (status=%s)')
    # Getting the type of 'status' (line 276)
    status_33224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 64), 'status', False)
    # Applying the binary operator '%' (line 276)
    result_mod_33225 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 21), '%', str_33223, status_33224)
    
    # Processing the call keyword arguments (line 276)
    kwargs_33226 = {}
    # Getting the type of 'log' (line 276)
    log_33221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'log', False)
    # Obtaining the member 'warn' of a type (line 276)
    warn_33222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 12), log_33221, 'warn')
    # Calling warn(args, kwargs) (line 276)
    warn_call_result_33227 = invoke(stypy.reporting.localization.Localization(__file__, 276, 12), warn_33222, *[result_mod_33225], **kwargs_33226)
    
    
    # Call to _exec_command(...): (line 277)
    # Processing the call arguments (line 277)
    # Getting the type of 'command' (line 277)
    command_33229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 33), 'command', False)
    # Processing the call keyword arguments (line 277)
    # Getting the type of 'use_shell' (line 277)
    use_shell_33230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 52), 'use_shell', False)
    keyword_33231 = use_shell_33230
    # Getting the type of 'env' (line 277)
    env_33232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 65), 'env', False)
    kwargs_33233 = {'use_shell': keyword_33231, 'env_33232': env_33232}
    # Getting the type of '_exec_command' (line 277)
    _exec_command_33228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 19), '_exec_command', False)
    # Calling _exec_command(args, kwargs) (line 277)
    _exec_command_call_result_33234 = invoke(stypy.reporting.localization.Localization(__file__, 277, 19), _exec_command_33228, *[command_33229], **kwargs_33233)
    
    # Assigning a type to the variable 'stypy_return_type' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'stypy_return_type', _exec_command_call_result_33234)
    # SSA join for if statement (line 274)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 273)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 279)
    # Getting the type of 'stsfile' (line 279)
    stsfile_33235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'stsfile')
    # Getting the type of 'None' (line 279)
    None_33236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 22), 'None')
    
    (may_be_33237, more_types_in_union_33238) = may_not_be_none(stsfile_33235, None_33236)

    if may_be_33237:

        if more_types_in_union_33238:
            # Runtime conditional SSA (line 279)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 280):
        
        # Assigning a Call to a Name (line 280):
        
        # Call to open_latin1(...): (line 280)
        # Processing the call arguments (line 280)
        # Getting the type of 'stsfile' (line 280)
        stsfile_33240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 24), 'stsfile', False)
        str_33241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 33), 'str', 'r')
        # Processing the call keyword arguments (line 280)
        kwargs_33242 = {}
        # Getting the type of 'open_latin1' (line 280)
        open_latin1_33239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'open_latin1', False)
        # Calling open_latin1(args, kwargs) (line 280)
        open_latin1_call_result_33243 = invoke(stypy.reporting.localization.Localization(__file__, 280, 12), open_latin1_33239, *[stsfile_33240, str_33241], **kwargs_33242)
        
        # Assigning a type to the variable 'f' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'f', open_latin1_call_result_33243)
        
        # Assigning a Call to a Name (line 281):
        
        # Assigning a Call to a Name (line 281):
        
        # Call to read(...): (line 281)
        # Processing the call keyword arguments (line 281)
        kwargs_33246 = {}
        # Getting the type of 'f' (line 281)
        f_33244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 22), 'f', False)
        # Obtaining the member 'read' of a type (line 281)
        read_33245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 22), f_33244, 'read')
        # Calling read(args, kwargs) (line 281)
        read_call_result_33247 = invoke(stypy.reporting.localization.Localization(__file__, 281, 22), read_33245, *[], **kwargs_33246)
        
        # Assigning a type to the variable 'status_text' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'status_text', read_call_result_33247)
        
        # Assigning a Call to a Name (line 282):
        
        # Assigning a Call to a Name (line 282):
        
        # Call to int(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'status_text' (line 282)
        status_text_33249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 21), 'status_text', False)
        # Processing the call keyword arguments (line 282)
        kwargs_33250 = {}
        # Getting the type of 'int' (line 282)
        int_33248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 17), 'int', False)
        # Calling int(args, kwargs) (line 282)
        int_call_result_33251 = invoke(stypy.reporting.localization.Localization(__file__, 282, 17), int_33248, *[status_text_33249], **kwargs_33250)
        
        # Assigning a type to the variable 'status' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'status', int_call_result_33251)
        
        # Call to close(...): (line 283)
        # Processing the call keyword arguments (line 283)
        kwargs_33254 = {}
        # Getting the type of 'f' (line 283)
        f_33252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'f', False)
        # Obtaining the member 'close' of a type (line 283)
        close_33253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 8), f_33252, 'close')
        # Calling close(args, kwargs) (line 283)
        close_call_result_33255 = invoke(stypy.reporting.localization.Localization(__file__, 283, 8), close_33253, *[], **kwargs_33254)
        
        
        # Call to remove(...): (line 284)
        # Processing the call arguments (line 284)
        # Getting the type of 'stsfile' (line 284)
        stsfile_33258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 18), 'stsfile', False)
        # Processing the call keyword arguments (line 284)
        kwargs_33259 = {}
        # Getting the type of 'os' (line 284)
        os_33256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'os', False)
        # Obtaining the member 'remove' of a type (line 284)
        remove_33257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 8), os_33256, 'remove')
        # Calling remove(args, kwargs) (line 284)
        remove_call_result_33260 = invoke(stypy.reporting.localization.Localization(__file__, 284, 8), remove_33257, *[stsfile_33258], **kwargs_33259)
        

        if more_types_in_union_33238:
            # SSA join for if statement (line 279)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 286):
    
    # Assigning a Call to a Name (line 286):
    
    # Call to open_latin1(...): (line 286)
    # Processing the call arguments (line 286)
    # Getting the type of 'tmpfile' (line 286)
    tmpfile_33262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 20), 'tmpfile', False)
    str_33263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 29), 'str', 'r')
    # Processing the call keyword arguments (line 286)
    kwargs_33264 = {}
    # Getting the type of 'open_latin1' (line 286)
    open_latin1_33261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'open_latin1', False)
    # Calling open_latin1(args, kwargs) (line 286)
    open_latin1_call_result_33265 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), open_latin1_33261, *[tmpfile_33262, str_33263], **kwargs_33264)
    
    # Assigning a type to the variable 'f' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'f', open_latin1_call_result_33265)
    
    # Assigning a Call to a Name (line 287):
    
    # Assigning a Call to a Name (line 287):
    
    # Call to read(...): (line 287)
    # Processing the call keyword arguments (line 287)
    kwargs_33268 = {}
    # Getting the type of 'f' (line 287)
    f_33266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 11), 'f', False)
    # Obtaining the member 'read' of a type (line 287)
    read_33267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 11), f_33266, 'read')
    # Calling read(args, kwargs) (line 287)
    read_call_result_33269 = invoke(stypy.reporting.localization.Localization(__file__, 287, 11), read_33267, *[], **kwargs_33268)
    
    # Assigning a type to the variable 'text' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'text', read_call_result_33269)
    
    # Call to close(...): (line 288)
    # Processing the call keyword arguments (line 288)
    kwargs_33272 = {}
    # Getting the type of 'f' (line 288)
    f_33270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'f', False)
    # Obtaining the member 'close' of a type (line 288)
    close_33271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 4), f_33270, 'close')
    # Calling close(args, kwargs) (line 288)
    close_call_result_33273 = invoke(stypy.reporting.localization.Localization(__file__, 288, 4), close_33271, *[], **kwargs_33272)
    
    
    # Call to remove(...): (line 289)
    # Processing the call arguments (line 289)
    # Getting the type of 'tmpfile' (line 289)
    tmpfile_33276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 14), 'tmpfile', False)
    # Processing the call keyword arguments (line 289)
    kwargs_33277 = {}
    # Getting the type of 'os' (line 289)
    os_33274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'os', False)
    # Obtaining the member 'remove' of a type (line 289)
    remove_33275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 4), os_33274, 'remove')
    # Calling remove(args, kwargs) (line 289)
    remove_call_result_33278 = invoke(stypy.reporting.localization.Localization(__file__, 289, 4), remove_33275, *[tmpfile_33276], **kwargs_33277)
    
    
    
    
    # Obtaining the type of the subscript
    int_33279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 12), 'int')
    slice_33280 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 291, 7), int_33279, None, None)
    # Getting the type of 'text' (line 291)
    text_33281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 7), 'text')
    # Obtaining the member '__getitem__' of a type (line 291)
    getitem___33282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 7), text_33281, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 291)
    subscript_call_result_33283 = invoke(stypy.reporting.localization.Localization(__file__, 291, 7), getitem___33282, slice_33280)
    
    str_33284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 18), 'str', '\n')
    # Applying the binary operator '==' (line 291)
    result_eq_33285 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 7), '==', subscript_call_result_33283, str_33284)
    
    # Testing the type of an if condition (line 291)
    if_condition_33286 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 291, 4), result_eq_33285)
    # Assigning a type to the variable 'if_condition_33286' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'if_condition_33286', if_condition_33286)
    # SSA begins for if statement (line 291)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 292):
    
    # Assigning a Subscript to a Name (line 292):
    
    # Obtaining the type of the subscript
    int_33287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 21), 'int')
    slice_33288 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 292, 15), None, int_33287, None)
    # Getting the type of 'text' (line 292)
    text_33289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 15), 'text')
    # Obtaining the member '__getitem__' of a type (line 292)
    getitem___33290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 15), text_33289, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 292)
    subscript_call_result_33291 = invoke(stypy.reporting.localization.Localization(__file__, 292, 15), getitem___33290, slice_33288)
    
    # Assigning a type to the variable 'text' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'text', subscript_call_result_33291)
    # SSA join for if statement (line 291)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 294)
    tuple_33292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 294)
    # Adding element type (line 294)
    # Getting the type of 'status' (line 294)
    status_33293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 11), 'status')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 11), tuple_33292, status_33293)
    # Adding element type (line 294)
    # Getting the type of 'text' (line 294)
    text_33294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 19), 'text')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 11), tuple_33292, text_33294)
    
    # Assigning a type to the variable 'stypy_return_type' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'stypy_return_type', tuple_33292)
    
    # ################# End of '_exec_command_posix(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_exec_command_posix' in the type store
    # Getting the type of 'stypy_return_type' (line 244)
    stypy_return_type_33295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_33295)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_exec_command_posix'
    return stypy_return_type_33295

# Assigning a type to the variable '_exec_command_posix' (line 244)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 0), '_exec_command_posix', _exec_command_posix)

@norecursion
def _exec_command_python(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_33296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 42), 'str', '')
    defaults = [str_33296]
    # Create a new context for function '_exec_command_python'
    module_type_store = module_type_store.open_function_context('_exec_command_python', 297, 0, False)
    
    # Passed parameters checking function
    _exec_command_python.stypy_localization = localization
    _exec_command_python.stypy_type_of_self = None
    _exec_command_python.stypy_type_store = module_type_store
    _exec_command_python.stypy_function_name = '_exec_command_python'
    _exec_command_python.stypy_param_names_list = ['command', 'exec_command_dir']
    _exec_command_python.stypy_varargs_param_name = None
    _exec_command_python.stypy_kwargs_param_name = 'env'
    _exec_command_python.stypy_call_defaults = defaults
    _exec_command_python.stypy_call_varargs = varargs
    _exec_command_python.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_exec_command_python', ['command', 'exec_command_dir'], None, 'env', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_exec_command_python', localization, ['command', 'exec_command_dir'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_exec_command_python(...)' code ##################

    
    # Call to debug(...): (line 299)
    # Processing the call arguments (line 299)
    str_33299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 14), 'str', '_exec_command_python(...)')
    # Processing the call keyword arguments (line 299)
    kwargs_33300 = {}
    # Getting the type of 'log' (line 299)
    log_33297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'log', False)
    # Obtaining the member 'debug' of a type (line 299)
    debug_33298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 4), log_33297, 'debug')
    # Calling debug(args, kwargs) (line 299)
    debug_call_result_33301 = invoke(stypy.reporting.localization.Localization(__file__, 299, 4), debug_33298, *[str_33299], **kwargs_33300)
    
    
    # Assigning a Call to a Name (line 301):
    
    # Assigning a Call to a Name (line 301):
    
    # Call to get_pythonexe(...): (line 301)
    # Processing the call keyword arguments (line 301)
    kwargs_33303 = {}
    # Getting the type of 'get_pythonexe' (line 301)
    get_pythonexe_33302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 17), 'get_pythonexe', False)
    # Calling get_pythonexe(args, kwargs) (line 301)
    get_pythonexe_call_result_33304 = invoke(stypy.reporting.localization.Localization(__file__, 301, 17), get_pythonexe_33302, *[], **kwargs_33303)
    
    # Assigning a type to the variable 'python_exe' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'python_exe', get_pythonexe_call_result_33304)
    
    # Assigning a Call to a Name (line 302):
    
    # Assigning a Call to a Name (line 302):
    
    # Call to temp_file_name(...): (line 302)
    # Processing the call keyword arguments (line 302)
    kwargs_33306 = {}
    # Getting the type of 'temp_file_name' (line 302)
    temp_file_name_33305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 14), 'temp_file_name', False)
    # Calling temp_file_name(args, kwargs) (line 302)
    temp_file_name_call_result_33307 = invoke(stypy.reporting.localization.Localization(__file__, 302, 14), temp_file_name_33305, *[], **kwargs_33306)
    
    # Assigning a type to the variable 'cmdfile' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'cmdfile', temp_file_name_call_result_33307)
    
    # Assigning a Call to a Name (line 303):
    
    # Assigning a Call to a Name (line 303):
    
    # Call to temp_file_name(...): (line 303)
    # Processing the call keyword arguments (line 303)
    kwargs_33309 = {}
    # Getting the type of 'temp_file_name' (line 303)
    temp_file_name_33308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 14), 'temp_file_name', False)
    # Calling temp_file_name(args, kwargs) (line 303)
    temp_file_name_call_result_33310 = invoke(stypy.reporting.localization.Localization(__file__, 303, 14), temp_file_name_33308, *[], **kwargs_33309)
    
    # Assigning a type to the variable 'stsfile' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'stsfile', temp_file_name_call_result_33310)
    
    # Assigning a Call to a Name (line 304):
    
    # Assigning a Call to a Name (line 304):
    
    # Call to temp_file_name(...): (line 304)
    # Processing the call keyword arguments (line 304)
    kwargs_33312 = {}
    # Getting the type of 'temp_file_name' (line 304)
    temp_file_name_33311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 14), 'temp_file_name', False)
    # Calling temp_file_name(args, kwargs) (line 304)
    temp_file_name_call_result_33313 = invoke(stypy.reporting.localization.Localization(__file__, 304, 14), temp_file_name_33311, *[], **kwargs_33312)
    
    # Assigning a type to the variable 'outfile' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'outfile', temp_file_name_call_result_33313)
    
    # Assigning a Call to a Name (line 306):
    
    # Assigning a Call to a Name (line 306):
    
    # Call to open(...): (line 306)
    # Processing the call arguments (line 306)
    # Getting the type of 'cmdfile' (line 306)
    cmdfile_33315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 13), 'cmdfile', False)
    str_33316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 22), 'str', 'w')
    # Processing the call keyword arguments (line 306)
    kwargs_33317 = {}
    # Getting the type of 'open' (line 306)
    open_33314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'open', False)
    # Calling open(args, kwargs) (line 306)
    open_call_result_33318 = invoke(stypy.reporting.localization.Localization(__file__, 306, 8), open_33314, *[cmdfile_33315, str_33316], **kwargs_33317)
    
    # Assigning a type to the variable 'f' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'f', open_call_result_33318)
    
    # Call to write(...): (line 307)
    # Processing the call arguments (line 307)
    str_33321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 12), 'str', 'import os\n')
    # Processing the call keyword arguments (line 307)
    kwargs_33322 = {}
    # Getting the type of 'f' (line 307)
    f_33319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'f', False)
    # Obtaining the member 'write' of a type (line 307)
    write_33320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 4), f_33319, 'write')
    # Calling write(args, kwargs) (line 307)
    write_call_result_33323 = invoke(stypy.reporting.localization.Localization(__file__, 307, 4), write_33320, *[str_33321], **kwargs_33322)
    
    
    # Call to write(...): (line 308)
    # Processing the call arguments (line 308)
    str_33326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 12), 'str', 'import sys\n')
    # Processing the call keyword arguments (line 308)
    kwargs_33327 = {}
    # Getting the type of 'f' (line 308)
    f_33324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'f', False)
    # Obtaining the member 'write' of a type (line 308)
    write_33325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 4), f_33324, 'write')
    # Calling write(args, kwargs) (line 308)
    write_call_result_33328 = invoke(stypy.reporting.localization.Localization(__file__, 308, 4), write_33325, *[str_33326], **kwargs_33327)
    
    
    # Call to write(...): (line 309)
    # Processing the call arguments (line 309)
    str_33331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 12), 'str', 'sys.path.insert(0,%r)\n')
    # Getting the type of 'exec_command_dir' (line 309)
    exec_command_dir_33332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 41), 'exec_command_dir', False)
    # Applying the binary operator '%' (line 309)
    result_mod_33333 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 12), '%', str_33331, exec_command_dir_33332)
    
    # Processing the call keyword arguments (line 309)
    kwargs_33334 = {}
    # Getting the type of 'f' (line 309)
    f_33329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'f', False)
    # Obtaining the member 'write' of a type (line 309)
    write_33330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 4), f_33329, 'write')
    # Calling write(args, kwargs) (line 309)
    write_call_result_33335 = invoke(stypy.reporting.localization.Localization(__file__, 309, 4), write_33330, *[result_mod_33333], **kwargs_33334)
    
    
    # Call to write(...): (line 310)
    # Processing the call arguments (line 310)
    str_33338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 12), 'str', 'from exec_command import exec_command\n')
    # Processing the call keyword arguments (line 310)
    kwargs_33339 = {}
    # Getting the type of 'f' (line 310)
    f_33336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'f', False)
    # Obtaining the member 'write' of a type (line 310)
    write_33337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 4), f_33336, 'write')
    # Calling write(args, kwargs) (line 310)
    write_call_result_33340 = invoke(stypy.reporting.localization.Localization(__file__, 310, 4), write_33337, *[str_33338], **kwargs_33339)
    
    
    # Call to write(...): (line 311)
    # Processing the call arguments (line 311)
    str_33343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 12), 'str', 'del sys.path[0]\n')
    # Processing the call keyword arguments (line 311)
    kwargs_33344 = {}
    # Getting the type of 'f' (line 311)
    f_33341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'f', False)
    # Obtaining the member 'write' of a type (line 311)
    write_33342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 4), f_33341, 'write')
    # Calling write(args, kwargs) (line 311)
    write_call_result_33345 = invoke(stypy.reporting.localization.Localization(__file__, 311, 4), write_33342, *[str_33343], **kwargs_33344)
    
    
    # Call to write(...): (line 312)
    # Processing the call arguments (line 312)
    str_33348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 12), 'str', 'cmd = %r\n')
    # Getting the type of 'command' (line 312)
    command_33349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 27), 'command', False)
    # Applying the binary operator '%' (line 312)
    result_mod_33350 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 12), '%', str_33348, command_33349)
    
    # Processing the call keyword arguments (line 312)
    kwargs_33351 = {}
    # Getting the type of 'f' (line 312)
    f_33346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'f', False)
    # Obtaining the member 'write' of a type (line 312)
    write_33347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 4), f_33346, 'write')
    # Calling write(args, kwargs) (line 312)
    write_call_result_33352 = invoke(stypy.reporting.localization.Localization(__file__, 312, 4), write_33347, *[result_mod_33350], **kwargs_33351)
    
    
    # Call to write(...): (line 313)
    # Processing the call arguments (line 313)
    str_33355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 12), 'str', 'os.environ = %r\n')
    # Getting the type of 'os' (line 313)
    os_33356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 35), 'os', False)
    # Obtaining the member 'environ' of a type (line 313)
    environ_33357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 35), os_33356, 'environ')
    # Applying the binary operator '%' (line 313)
    result_mod_33358 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 12), '%', str_33355, environ_33357)
    
    # Processing the call keyword arguments (line 313)
    kwargs_33359 = {}
    # Getting the type of 'f' (line 313)
    f_33353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'f', False)
    # Obtaining the member 'write' of a type (line 313)
    write_33354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 4), f_33353, 'write')
    # Calling write(args, kwargs) (line 313)
    write_call_result_33360 = invoke(stypy.reporting.localization.Localization(__file__, 313, 4), write_33354, *[result_mod_33358], **kwargs_33359)
    
    
    # Call to write(...): (line 314)
    # Processing the call arguments (line 314)
    str_33363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 12), 'str', 's,o = exec_command(cmd, _with_python=0, **%r)\n')
    # Getting the type of 'env' (line 314)
    env_33364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 65), 'env', False)
    # Applying the binary operator '%' (line 314)
    result_mod_33365 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 12), '%', str_33363, env_33364)
    
    # Processing the call keyword arguments (line 314)
    kwargs_33366 = {}
    # Getting the type of 'f' (line 314)
    f_33361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'f', False)
    # Obtaining the member 'write' of a type (line 314)
    write_33362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 4), f_33361, 'write')
    # Calling write(args, kwargs) (line 314)
    write_call_result_33367 = invoke(stypy.reporting.localization.Localization(__file__, 314, 4), write_33362, *[result_mod_33365], **kwargs_33366)
    
    
    # Call to write(...): (line 315)
    # Processing the call arguments (line 315)
    str_33370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 12), 'str', 'f=open(%r,"w")\nf.write(str(s))\nf.close()\n')
    # Getting the type of 'stsfile' (line 315)
    stsfile_33371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 62), 'stsfile', False)
    # Applying the binary operator '%' (line 315)
    result_mod_33372 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 12), '%', str_33370, stsfile_33371)
    
    # Processing the call keyword arguments (line 315)
    kwargs_33373 = {}
    # Getting the type of 'f' (line 315)
    f_33368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'f', False)
    # Obtaining the member 'write' of a type (line 315)
    write_33369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 4), f_33368, 'write')
    # Calling write(args, kwargs) (line 315)
    write_call_result_33374 = invoke(stypy.reporting.localization.Localization(__file__, 315, 4), write_33369, *[result_mod_33372], **kwargs_33373)
    
    
    # Call to write(...): (line 316)
    # Processing the call arguments (line 316)
    str_33377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 12), 'str', 'f=open(%r,"w")\nf.write(o)\nf.close()\n')
    # Getting the type of 'outfile' (line 316)
    outfile_33378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 57), 'outfile', False)
    # Applying the binary operator '%' (line 316)
    result_mod_33379 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 12), '%', str_33377, outfile_33378)
    
    # Processing the call keyword arguments (line 316)
    kwargs_33380 = {}
    # Getting the type of 'f' (line 316)
    f_33375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'f', False)
    # Obtaining the member 'write' of a type (line 316)
    write_33376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 4), f_33375, 'write')
    # Calling write(args, kwargs) (line 316)
    write_call_result_33381 = invoke(stypy.reporting.localization.Localization(__file__, 316, 4), write_33376, *[result_mod_33379], **kwargs_33380)
    
    
    # Call to close(...): (line 317)
    # Processing the call keyword arguments (line 317)
    kwargs_33384 = {}
    # Getting the type of 'f' (line 317)
    f_33382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'f', False)
    # Obtaining the member 'close' of a type (line 317)
    close_33383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 4), f_33382, 'close')
    # Calling close(args, kwargs) (line 317)
    close_call_result_33385 = invoke(stypy.reporting.localization.Localization(__file__, 317, 4), close_33383, *[], **kwargs_33384)
    
    
    # Assigning a BinOp to a Name (line 319):
    
    # Assigning a BinOp to a Name (line 319):
    str_33386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 10), 'str', '%s %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 319)
    tuple_33387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 319)
    # Adding element type (line 319)
    # Getting the type of 'python_exe' (line 319)
    python_exe_33388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 21), 'python_exe')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 21), tuple_33387, python_exe_33388)
    # Adding element type (line 319)
    # Getting the type of 'cmdfile' (line 319)
    cmdfile_33389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 33), 'cmdfile')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 21), tuple_33387, cmdfile_33389)
    
    # Applying the binary operator '%' (line 319)
    result_mod_33390 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 10), '%', str_33386, tuple_33387)
    
    # Assigning a type to the variable 'cmd' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'cmd', result_mod_33390)
    
    # Assigning a Call to a Name (line 320):
    
    # Assigning a Call to a Name (line 320):
    
    # Call to system(...): (line 320)
    # Processing the call arguments (line 320)
    # Getting the type of 'cmd' (line 320)
    cmd_33393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 23), 'cmd', False)
    # Processing the call keyword arguments (line 320)
    kwargs_33394 = {}
    # Getting the type of 'os' (line 320)
    os_33391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 13), 'os', False)
    # Obtaining the member 'system' of a type (line 320)
    system_33392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 13), os_33391, 'system')
    # Calling system(args, kwargs) (line 320)
    system_call_result_33395 = invoke(stypy.reporting.localization.Localization(__file__, 320, 13), system_33392, *[cmd_33393], **kwargs_33394)
    
    # Assigning a type to the variable 'status' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'status', system_call_result_33395)
    
    # Getting the type of 'status' (line 321)
    status_33396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 7), 'status')
    # Testing the type of an if condition (line 321)
    if_condition_33397 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 321, 4), status_33396)
    # Assigning a type to the variable 'if_condition_33397' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'if_condition_33397', if_condition_33397)
    # SSA begins for if statement (line 321)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 322)
    # Processing the call arguments (line 322)
    str_33399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 27), 'str', '%r failed')
    
    # Obtaining an instance of the builtin type 'tuple' (line 322)
    tuple_33400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 322)
    # Adding element type (line 322)
    # Getting the type of 'cmd' (line 322)
    cmd_33401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 42), 'cmd', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 42), tuple_33400, cmd_33401)
    
    # Applying the binary operator '%' (line 322)
    result_mod_33402 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 27), '%', str_33399, tuple_33400)
    
    # Processing the call keyword arguments (line 322)
    kwargs_33403 = {}
    # Getting the type of 'RuntimeError' (line 322)
    RuntimeError_33398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 322)
    RuntimeError_call_result_33404 = invoke(stypy.reporting.localization.Localization(__file__, 322, 14), RuntimeError_33398, *[result_mod_33402], **kwargs_33403)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 322, 8), RuntimeError_call_result_33404, 'raise parameter', BaseException)
    # SSA join for if statement (line 321)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to remove(...): (line 323)
    # Processing the call arguments (line 323)
    # Getting the type of 'cmdfile' (line 323)
    cmdfile_33407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 14), 'cmdfile', False)
    # Processing the call keyword arguments (line 323)
    kwargs_33408 = {}
    # Getting the type of 'os' (line 323)
    os_33405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'os', False)
    # Obtaining the member 'remove' of a type (line 323)
    remove_33406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 4), os_33405, 'remove')
    # Calling remove(args, kwargs) (line 323)
    remove_call_result_33409 = invoke(stypy.reporting.localization.Localization(__file__, 323, 4), remove_33406, *[cmdfile_33407], **kwargs_33408)
    
    
    # Assigning a Call to a Name (line 325):
    
    # Assigning a Call to a Name (line 325):
    
    # Call to open_latin1(...): (line 325)
    # Processing the call arguments (line 325)
    # Getting the type of 'stsfile' (line 325)
    stsfile_33411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 20), 'stsfile', False)
    str_33412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 29), 'str', 'r')
    # Processing the call keyword arguments (line 325)
    kwargs_33413 = {}
    # Getting the type of 'open_latin1' (line 325)
    open_latin1_33410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'open_latin1', False)
    # Calling open_latin1(args, kwargs) (line 325)
    open_latin1_call_result_33414 = invoke(stypy.reporting.localization.Localization(__file__, 325, 8), open_latin1_33410, *[stsfile_33411, str_33412], **kwargs_33413)
    
    # Assigning a type to the variable 'f' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'f', open_latin1_call_result_33414)
    
    # Assigning a Call to a Name (line 326):
    
    # Assigning a Call to a Name (line 326):
    
    # Call to int(...): (line 326)
    # Processing the call arguments (line 326)
    
    # Call to read(...): (line 326)
    # Processing the call keyword arguments (line 326)
    kwargs_33418 = {}
    # Getting the type of 'f' (line 326)
    f_33416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 17), 'f', False)
    # Obtaining the member 'read' of a type (line 326)
    read_33417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 17), f_33416, 'read')
    # Calling read(args, kwargs) (line 326)
    read_call_result_33419 = invoke(stypy.reporting.localization.Localization(__file__, 326, 17), read_33417, *[], **kwargs_33418)
    
    # Processing the call keyword arguments (line 326)
    kwargs_33420 = {}
    # Getting the type of 'int' (line 326)
    int_33415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 13), 'int', False)
    # Calling int(args, kwargs) (line 326)
    int_call_result_33421 = invoke(stypy.reporting.localization.Localization(__file__, 326, 13), int_33415, *[read_call_result_33419], **kwargs_33420)
    
    # Assigning a type to the variable 'status' (line 326)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'status', int_call_result_33421)
    
    # Call to close(...): (line 327)
    # Processing the call keyword arguments (line 327)
    kwargs_33424 = {}
    # Getting the type of 'f' (line 327)
    f_33422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'f', False)
    # Obtaining the member 'close' of a type (line 327)
    close_33423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 4), f_33422, 'close')
    # Calling close(args, kwargs) (line 327)
    close_call_result_33425 = invoke(stypy.reporting.localization.Localization(__file__, 327, 4), close_33423, *[], **kwargs_33424)
    
    
    # Call to remove(...): (line 328)
    # Processing the call arguments (line 328)
    # Getting the type of 'stsfile' (line 328)
    stsfile_33428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 14), 'stsfile', False)
    # Processing the call keyword arguments (line 328)
    kwargs_33429 = {}
    # Getting the type of 'os' (line 328)
    os_33426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'os', False)
    # Obtaining the member 'remove' of a type (line 328)
    remove_33427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 4), os_33426, 'remove')
    # Calling remove(args, kwargs) (line 328)
    remove_call_result_33430 = invoke(stypy.reporting.localization.Localization(__file__, 328, 4), remove_33427, *[stsfile_33428], **kwargs_33429)
    
    
    # Assigning a Call to a Name (line 330):
    
    # Assigning a Call to a Name (line 330):
    
    # Call to open_latin1(...): (line 330)
    # Processing the call arguments (line 330)
    # Getting the type of 'outfile' (line 330)
    outfile_33432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 20), 'outfile', False)
    str_33433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 29), 'str', 'r')
    # Processing the call keyword arguments (line 330)
    kwargs_33434 = {}
    # Getting the type of 'open_latin1' (line 330)
    open_latin1_33431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'open_latin1', False)
    # Calling open_latin1(args, kwargs) (line 330)
    open_latin1_call_result_33435 = invoke(stypy.reporting.localization.Localization(__file__, 330, 8), open_latin1_33431, *[outfile_33432, str_33433], **kwargs_33434)
    
    # Assigning a type to the variable 'f' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'f', open_latin1_call_result_33435)
    
    # Assigning a Call to a Name (line 331):
    
    # Assigning a Call to a Name (line 331):
    
    # Call to read(...): (line 331)
    # Processing the call keyword arguments (line 331)
    kwargs_33438 = {}
    # Getting the type of 'f' (line 331)
    f_33436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 11), 'f', False)
    # Obtaining the member 'read' of a type (line 331)
    read_33437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 11), f_33436, 'read')
    # Calling read(args, kwargs) (line 331)
    read_call_result_33439 = invoke(stypy.reporting.localization.Localization(__file__, 331, 11), read_33437, *[], **kwargs_33438)
    
    # Assigning a type to the variable 'text' (line 331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'text', read_call_result_33439)
    
    # Call to close(...): (line 332)
    # Processing the call keyword arguments (line 332)
    kwargs_33442 = {}
    # Getting the type of 'f' (line 332)
    f_33440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'f', False)
    # Obtaining the member 'close' of a type (line 332)
    close_33441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 4), f_33440, 'close')
    # Calling close(args, kwargs) (line 332)
    close_call_result_33443 = invoke(stypy.reporting.localization.Localization(__file__, 332, 4), close_33441, *[], **kwargs_33442)
    
    
    # Call to remove(...): (line 333)
    # Processing the call arguments (line 333)
    # Getting the type of 'outfile' (line 333)
    outfile_33446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 14), 'outfile', False)
    # Processing the call keyword arguments (line 333)
    kwargs_33447 = {}
    # Getting the type of 'os' (line 333)
    os_33444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'os', False)
    # Obtaining the member 'remove' of a type (line 333)
    remove_33445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 4), os_33444, 'remove')
    # Calling remove(args, kwargs) (line 333)
    remove_call_result_33448 = invoke(stypy.reporting.localization.Localization(__file__, 333, 4), remove_33445, *[outfile_33446], **kwargs_33447)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 335)
    tuple_33449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 335)
    # Adding element type (line 335)
    # Getting the type of 'status' (line 335)
    status_33450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 11), 'status')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 11), tuple_33449, status_33450)
    # Adding element type (line 335)
    # Getting the type of 'text' (line 335)
    text_33451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 19), 'text')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 11), tuple_33449, text_33451)
    
    # Assigning a type to the variable 'stypy_return_type' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'stypy_return_type', tuple_33449)
    
    # ################# End of '_exec_command_python(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_exec_command_python' in the type store
    # Getting the type of 'stypy_return_type' (line 297)
    stypy_return_type_33452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_33452)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_exec_command_python'
    return stypy_return_type_33452

# Assigning a type to the variable '_exec_command_python' (line 297)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 0), '_exec_command_python', _exec_command_python)

@norecursion
def quote_arg(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'quote_arg'
    module_type_store = module_type_store.open_function_context('quote_arg', 337, 0, False)
    
    # Passed parameters checking function
    quote_arg.stypy_localization = localization
    quote_arg.stypy_type_of_self = None
    quote_arg.stypy_type_store = module_type_store
    quote_arg.stypy_function_name = 'quote_arg'
    quote_arg.stypy_param_names_list = ['arg']
    quote_arg.stypy_varargs_param_name = None
    quote_arg.stypy_kwargs_param_name = None
    quote_arg.stypy_call_defaults = defaults
    quote_arg.stypy_call_varargs = varargs
    quote_arg.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'quote_arg', ['arg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'quote_arg', localization, ['arg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'quote_arg(...)' code ##################

    
    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    int_33453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 11), 'int')
    # Getting the type of 'arg' (line 338)
    arg_33454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 7), 'arg')
    # Obtaining the member '__getitem__' of a type (line 338)
    getitem___33455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 7), arg_33454, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 338)
    subscript_call_result_33456 = invoke(stypy.reporting.localization.Localization(__file__, 338, 7), getitem___33455, int_33453)
    
    str_33457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 15), 'str', '"')
    # Applying the binary operator '!=' (line 338)
    result_ne_33458 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 7), '!=', subscript_call_result_33456, str_33457)
    
    
    str_33459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 23), 'str', ' ')
    # Getting the type of 'arg' (line 338)
    arg_33460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 30), 'arg')
    # Applying the binary operator 'in' (line 338)
    result_contains_33461 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 23), 'in', str_33459, arg_33460)
    
    # Applying the binary operator 'and' (line 338)
    result_and_keyword_33462 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 7), 'and', result_ne_33458, result_contains_33461)
    
    # Testing the type of an if condition (line 338)
    if_condition_33463 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 338, 4), result_and_keyword_33462)
    # Assigning a type to the variable 'if_condition_33463' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'if_condition_33463', if_condition_33463)
    # SSA begins for if statement (line 338)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_33464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 15), 'str', '"%s"')
    # Getting the type of 'arg' (line 339)
    arg_33465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 24), 'arg')
    # Applying the binary operator '%' (line 339)
    result_mod_33466 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 15), '%', str_33464, arg_33465)
    
    # Assigning a type to the variable 'stypy_return_type' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'stypy_return_type', result_mod_33466)
    # SSA join for if statement (line 338)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'arg' (line 340)
    arg_33467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 11), 'arg')
    # Assigning a type to the variable 'stypy_return_type' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'stypy_return_type', arg_33467)
    
    # ################# End of 'quote_arg(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'quote_arg' in the type store
    # Getting the type of 'stypy_return_type' (line 337)
    stypy_return_type_33468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_33468)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'quote_arg'
    return stypy_return_type_33468

# Assigning a type to the variable 'quote_arg' (line 337)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 0), 'quote_arg', quote_arg)

@norecursion
def _exec_command(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 342)
    None_33469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 38), 'None')
    # Getting the type of 'None' (line 342)
    None_33470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 54), 'None')
    defaults = [None_33469, None_33470]
    # Create a new context for function '_exec_command'
    module_type_store = module_type_store.open_function_context('_exec_command', 342, 0, False)
    
    # Passed parameters checking function
    _exec_command.stypy_localization = localization
    _exec_command.stypy_type_of_self = None
    _exec_command.stypy_type_store = module_type_store
    _exec_command.stypy_function_name = '_exec_command'
    _exec_command.stypy_param_names_list = ['command', 'use_shell', 'use_tee']
    _exec_command.stypy_varargs_param_name = None
    _exec_command.stypy_kwargs_param_name = 'env'
    _exec_command.stypy_call_defaults = defaults
    _exec_command.stypy_call_varargs = varargs
    _exec_command.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_exec_command', ['command', 'use_shell', 'use_tee'], None, 'env', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_exec_command', localization, ['command', 'use_shell', 'use_tee'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_exec_command(...)' code ##################

    
    # Call to debug(...): (line 343)
    # Processing the call arguments (line 343)
    str_33473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 14), 'str', '_exec_command(...)')
    # Processing the call keyword arguments (line 343)
    kwargs_33474 = {}
    # Getting the type of 'log' (line 343)
    log_33471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'log', False)
    # Obtaining the member 'debug' of a type (line 343)
    debug_33472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 4), log_33471, 'debug')
    # Calling debug(args, kwargs) (line 343)
    debug_call_result_33475 = invoke(stypy.reporting.localization.Localization(__file__, 343, 4), debug_33472, *[str_33473], **kwargs_33474)
    
    
    # Type idiom detected: calculating its left and rigth part (line 345)
    # Getting the type of 'use_shell' (line 345)
    use_shell_33476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 7), 'use_shell')
    # Getting the type of 'None' (line 345)
    None_33477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 20), 'None')
    
    (may_be_33478, more_types_in_union_33479) = may_be_none(use_shell_33476, None_33477)

    if may_be_33478:

        if more_types_in_union_33479:
            # Runtime conditional SSA (line 345)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Compare to a Name (line 346):
        
        # Assigning a Compare to a Name (line 346):
        
        # Getting the type of 'os' (line 346)
        os_33480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 20), 'os')
        # Obtaining the member 'name' of a type (line 346)
        name_33481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 20), os_33480, 'name')
        str_33482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 29), 'str', 'posix')
        # Applying the binary operator '==' (line 346)
        result_eq_33483 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 20), '==', name_33481, str_33482)
        
        # Assigning a type to the variable 'use_shell' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'use_shell', result_eq_33483)

        if more_types_in_union_33479:
            # SSA join for if statement (line 345)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 347)
    # Getting the type of 'use_tee' (line 347)
    use_tee_33484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 7), 'use_tee')
    # Getting the type of 'None' (line 347)
    None_33485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 18), 'None')
    
    (may_be_33486, more_types_in_union_33487) = may_be_none(use_tee_33484, None_33485)

    if may_be_33486:

        if more_types_in_union_33487:
            # Runtime conditional SSA (line 347)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Compare to a Name (line 348):
        
        # Assigning a Compare to a Name (line 348):
        
        # Getting the type of 'os' (line 348)
        os_33488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 18), 'os')
        # Obtaining the member 'name' of a type (line 348)
        name_33489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 18), os_33488, 'name')
        str_33490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 27), 'str', 'posix')
        # Applying the binary operator '==' (line 348)
        result_eq_33491 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 18), '==', name_33489, str_33490)
        
        # Assigning a type to the variable 'use_tee' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'use_tee', result_eq_33491)

        if more_types_in_union_33487:
            # SSA join for if statement (line 347)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Num to a Name (line 349):
    
    # Assigning a Num to a Name (line 349):
    int_33492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 20), 'int')
    # Assigning a type to the variable 'using_command' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'using_command', int_33492)
    
    # Getting the type of 'use_shell' (line 350)
    use_shell_33493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 7), 'use_shell')
    # Testing the type of an if condition (line 350)
    if_condition_33494 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 350, 4), use_shell_33493)
    # Assigning a type to the variable 'if_condition_33494' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'if_condition_33494', if_condition_33494)
    # SSA begins for if statement (line 350)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 353):
    
    # Assigning a Call to a Name (line 353):
    
    # Call to get(...): (line 353)
    # Processing the call arguments (line 353)
    str_33498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 28), 'str', 'SHELL')
    str_33499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 37), 'str', '/bin/sh')
    # Processing the call keyword arguments (line 353)
    kwargs_33500 = {}
    # Getting the type of 'os' (line 353)
    os_33495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 13), 'os', False)
    # Obtaining the member 'environ' of a type (line 353)
    environ_33496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 13), os_33495, 'environ')
    # Obtaining the member 'get' of a type (line 353)
    get_33497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 13), environ_33496, 'get')
    # Calling get(args, kwargs) (line 353)
    get_call_result_33501 = invoke(stypy.reporting.localization.Localization(__file__, 353, 13), get_33497, *[str_33498, str_33499], **kwargs_33500)
    
    # Assigning a type to the variable 'sh' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'sh', get_call_result_33501)
    
    
    # Call to is_sequence(...): (line 354)
    # Processing the call arguments (line 354)
    # Getting the type of 'command' (line 354)
    command_33503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 23), 'command', False)
    # Processing the call keyword arguments (line 354)
    kwargs_33504 = {}
    # Getting the type of 'is_sequence' (line 354)
    is_sequence_33502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 11), 'is_sequence', False)
    # Calling is_sequence(args, kwargs) (line 354)
    is_sequence_call_result_33505 = invoke(stypy.reporting.localization.Localization(__file__, 354, 11), is_sequence_33502, *[command_33503], **kwargs_33504)
    
    # Testing the type of an if condition (line 354)
    if_condition_33506 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 354, 8), is_sequence_call_result_33505)
    # Assigning a type to the variable 'if_condition_33506' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'if_condition_33506', if_condition_33506)
    # SSA begins for if statement (line 354)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 355):
    
    # Assigning a List to a Name (line 355):
    
    # Obtaining an instance of the builtin type 'list' (line 355)
    list_33507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 355)
    # Adding element type (line 355)
    # Getting the type of 'sh' (line 355)
    sh_33508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 20), 'sh')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 19), list_33507, sh_33508)
    # Adding element type (line 355)
    str_33509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 24), 'str', '-c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 19), list_33507, str_33509)
    # Adding element type (line 355)
    
    # Call to join(...): (line 355)
    # Processing the call arguments (line 355)
    
    # Call to list(...): (line 355)
    # Processing the call arguments (line 355)
    # Getting the type of 'command' (line 355)
    command_33513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 44), 'command', False)
    # Processing the call keyword arguments (line 355)
    kwargs_33514 = {}
    # Getting the type of 'list' (line 355)
    list_33512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 39), 'list', False)
    # Calling list(args, kwargs) (line 355)
    list_call_result_33515 = invoke(stypy.reporting.localization.Localization(__file__, 355, 39), list_33512, *[command_33513], **kwargs_33514)
    
    # Processing the call keyword arguments (line 355)
    kwargs_33516 = {}
    str_33510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 30), 'str', ' ')
    # Obtaining the member 'join' of a type (line 355)
    join_33511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 30), str_33510, 'join')
    # Calling join(args, kwargs) (line 355)
    join_call_result_33517 = invoke(stypy.reporting.localization.Localization(__file__, 355, 30), join_33511, *[list_call_result_33515], **kwargs_33516)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 19), list_33507, join_call_result_33517)
    
    # Assigning a type to the variable 'argv' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 12), 'argv', list_33507)
    # SSA branch for the else part of an if statement (line 354)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a List to a Name (line 357):
    
    # Assigning a List to a Name (line 357):
    
    # Obtaining an instance of the builtin type 'list' (line 357)
    list_33518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 357)
    # Adding element type (line 357)
    # Getting the type of 'sh' (line 357)
    sh_33519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 20), 'sh')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 19), list_33518, sh_33519)
    # Adding element type (line 357)
    str_33520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 24), 'str', '-c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 19), list_33518, str_33520)
    # Adding element type (line 357)
    # Getting the type of 'command' (line 357)
    command_33521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 30), 'command')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 19), list_33518, command_33521)
    
    # Assigning a type to the variable 'argv' (line 357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 12), 'argv', list_33518)
    # SSA join for if statement (line 354)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 350)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to is_sequence(...): (line 361)
    # Processing the call arguments (line 361)
    # Getting the type of 'command' (line 361)
    command_33523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 23), 'command', False)
    # Processing the call keyword arguments (line 361)
    kwargs_33524 = {}
    # Getting the type of 'is_sequence' (line 361)
    is_sequence_33522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 11), 'is_sequence', False)
    # Calling is_sequence(args, kwargs) (line 361)
    is_sequence_call_result_33525 = invoke(stypy.reporting.localization.Localization(__file__, 361, 11), is_sequence_33522, *[command_33523], **kwargs_33524)
    
    # Testing the type of an if condition (line 361)
    if_condition_33526 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 361, 8), is_sequence_call_result_33525)
    # Assigning a type to the variable 'if_condition_33526' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'if_condition_33526', if_condition_33526)
    # SSA begins for if statement (line 361)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 362):
    
    # Assigning a Subscript to a Name (line 362):
    
    # Obtaining the type of the subscript
    slice_33527 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 362, 19), None, None, None)
    # Getting the type of 'command' (line 362)
    command_33528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 19), 'command')
    # Obtaining the member '__getitem__' of a type (line 362)
    getitem___33529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 19), command_33528, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 362)
    subscript_call_result_33530 = invoke(stypy.reporting.localization.Localization(__file__, 362, 19), getitem___33529, slice_33527)
    
    # Assigning a type to the variable 'argv' (line 362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'argv', subscript_call_result_33530)
    # SSA branch for the else part of an if statement (line 361)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 364):
    
    # Assigning a Call to a Name (line 364):
    
    # Call to split(...): (line 364)
    # Processing the call arguments (line 364)
    # Getting the type of 'command' (line 364)
    command_33533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 31), 'command', False)
    # Processing the call keyword arguments (line 364)
    kwargs_33534 = {}
    # Getting the type of 'shlex' (line 364)
    shlex_33531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 19), 'shlex', False)
    # Obtaining the member 'split' of a type (line 364)
    split_33532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 19), shlex_33531, 'split')
    # Calling split(args, kwargs) (line 364)
    split_call_result_33535 = invoke(stypy.reporting.localization.Localization(__file__, 364, 19), split_33532, *[command_33533], **kwargs_33534)
    
    # Assigning a type to the variable 'argv' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 12), 'argv', split_call_result_33535)
    # SSA join for if statement (line 361)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 350)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 366)
    str_33536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 19), 'str', 'spawnvpe')
    # Getting the type of 'os' (line 366)
    os_33537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 15), 'os')
    
    (may_be_33538, more_types_in_union_33539) = may_provide_member(str_33536, os_33537)

    if may_be_33538:

        if more_types_in_union_33539:
            # Runtime conditional SSA (line 366)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'os' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'os', remove_not_member_provider_from_union(os_33537, 'spawnvpe'))
        
        # Assigning a Attribute to a Name (line 367):
        
        # Assigning a Attribute to a Name (line 367):
        # Getting the type of 'os' (line 367)
        os_33540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 24), 'os')
        # Obtaining the member 'spawnvpe' of a type (line 367)
        spawnvpe_33541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 24), os_33540, 'spawnvpe')
        # Assigning a type to the variable 'spawn_command' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'spawn_command', spawnvpe_33541)

        if more_types_in_union_33539:
            # Runtime conditional SSA for else branch (line 366)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_33538) or more_types_in_union_33539):
        # Assigning a type to the variable 'os' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'os', remove_member_provider_from_union(os_33537, 'spawnvpe'))
        
        # Assigning a Attribute to a Name (line 369):
        
        # Assigning a Attribute to a Name (line 369):
        # Getting the type of 'os' (line 369)
        os_33542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 24), 'os')
        # Obtaining the member 'spawnve' of a type (line 369)
        spawnve_33543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 24), os_33542, 'spawnve')
        # Assigning a type to the variable 'spawn_command' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'spawn_command', spawnve_33543)
        
        # Assigning a BoolOp to a Subscript (line 370):
        
        # Assigning a BoolOp to a Subscript (line 370):
        
        # Evaluating a boolean operation
        
        # Call to find_executable(...): (line 370)
        # Processing the call arguments (line 370)
        
        # Obtaining the type of the subscript
        int_33545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 39), 'int')
        # Getting the type of 'argv' (line 370)
        argv_33546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 34), 'argv', False)
        # Obtaining the member '__getitem__' of a type (line 370)
        getitem___33547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 34), argv_33546, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 370)
        subscript_call_result_33548 = invoke(stypy.reporting.localization.Localization(__file__, 370, 34), getitem___33547, int_33545)
        
        # Processing the call keyword arguments (line 370)
        kwargs_33549 = {}
        # Getting the type of 'find_executable' (line 370)
        find_executable_33544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 18), 'find_executable', False)
        # Calling find_executable(args, kwargs) (line 370)
        find_executable_call_result_33550 = invoke(stypy.reporting.localization.Localization(__file__, 370, 18), find_executable_33544, *[subscript_call_result_33548], **kwargs_33549)
        
        
        # Obtaining the type of the subscript
        int_33551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 51), 'int')
        # Getting the type of 'argv' (line 370)
        argv_33552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 46), 'argv')
        # Obtaining the member '__getitem__' of a type (line 370)
        getitem___33553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 46), argv_33552, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 370)
        subscript_call_result_33554 = invoke(stypy.reporting.localization.Localization(__file__, 370, 46), getitem___33553, int_33551)
        
        # Applying the binary operator 'or' (line 370)
        result_or_keyword_33555 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 18), 'or', find_executable_call_result_33550, subscript_call_result_33554)
        
        # Getting the type of 'argv' (line 370)
        argv_33556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'argv')
        int_33557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 13), 'int')
        # Storing an element on a container (line 370)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 8), argv_33556, (int_33557, result_or_keyword_33555))
        
        
        
        # Call to isfile(...): (line 371)
        # Processing the call arguments (line 371)
        
        # Obtaining the type of the subscript
        int_33561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 35), 'int')
        # Getting the type of 'argv' (line 371)
        argv_33562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 30), 'argv', False)
        # Obtaining the member '__getitem__' of a type (line 371)
        getitem___33563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 30), argv_33562, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 371)
        subscript_call_result_33564 = invoke(stypy.reporting.localization.Localization(__file__, 371, 30), getitem___33563, int_33561)
        
        # Processing the call keyword arguments (line 371)
        kwargs_33565 = {}
        # Getting the type of 'os' (line 371)
        os_33558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 371)
        path_33559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 15), os_33558, 'path')
        # Obtaining the member 'isfile' of a type (line 371)
        isfile_33560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 15), path_33559, 'isfile')
        # Calling isfile(args, kwargs) (line 371)
        isfile_call_result_33566 = invoke(stypy.reporting.localization.Localization(__file__, 371, 15), isfile_33560, *[subscript_call_result_33564], **kwargs_33565)
        
        # Applying the 'not' unary operator (line 371)
        result_not__33567 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 11), 'not', isfile_call_result_33566)
        
        # Testing the type of an if condition (line 371)
        if_condition_33568 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 371, 8), result_not__33567)
        # Assigning a type to the variable 'if_condition_33568' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'if_condition_33568', if_condition_33568)
        # SSA begins for if statement (line 371)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 372)
        # Processing the call arguments (line 372)
        str_33571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 21), 'str', 'Executable %s does not exist')
        
        # Obtaining the type of the subscript
        int_33572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 60), 'int')
        # Getting the type of 'argv' (line 372)
        argv_33573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 55), 'argv', False)
        # Obtaining the member '__getitem__' of a type (line 372)
        getitem___33574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 55), argv_33573, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 372)
        subscript_call_result_33575 = invoke(stypy.reporting.localization.Localization(__file__, 372, 55), getitem___33574, int_33572)
        
        # Applying the binary operator '%' (line 372)
        result_mod_33576 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 21), '%', str_33571, subscript_call_result_33575)
        
        # Processing the call keyword arguments (line 372)
        kwargs_33577 = {}
        # Getting the type of 'log' (line 372)
        log_33569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'log', False)
        # Obtaining the member 'warn' of a type (line 372)
        warn_33570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 12), log_33569, 'warn')
        # Calling warn(args, kwargs) (line 372)
        warn_call_result_33578 = invoke(stypy.reporting.localization.Localization(__file__, 372, 12), warn_33570, *[result_mod_33576], **kwargs_33577)
        
        
        
        # Getting the type of 'os' (line 373)
        os_33579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 15), 'os')
        # Obtaining the member 'name' of a type (line 373)
        name_33580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 15), os_33579, 'name')
        
        # Obtaining an instance of the builtin type 'list' (line 373)
        list_33581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 373)
        # Adding element type (line 373)
        str_33582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 27), 'str', 'nt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 26), list_33581, str_33582)
        # Adding element type (line 373)
        str_33583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 33), 'str', 'dos')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 26), list_33581, str_33583)
        
        # Applying the binary operator 'in' (line 373)
        result_contains_33584 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 15), 'in', name_33580, list_33581)
        
        # Testing the type of an if condition (line 373)
        if_condition_33585 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 373, 12), result_contains_33584)
        # Assigning a type to the variable 'if_condition_33585' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 12), 'if_condition_33585', if_condition_33585)
        # SSA begins for if statement (line 373)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 375):
        
        # Assigning a BinOp to a Name (line 375):
        
        # Obtaining an instance of the builtin type 'list' (line 375)
        list_33586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 375)
        # Adding element type (line 375)
        
        # Obtaining the type of the subscript
        str_33587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 35), 'str', 'COMSPEC')
        # Getting the type of 'os' (line 375)
        os_33588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 24), 'os')
        # Obtaining the member 'environ' of a type (line 375)
        environ_33589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 24), os_33588, 'environ')
        # Obtaining the member '__getitem__' of a type (line 375)
        getitem___33590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 24), environ_33589, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 375)
        subscript_call_result_33591 = invoke(stypy.reporting.localization.Localization(__file__, 375, 24), getitem___33590, str_33587)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 23), list_33586, subscript_call_result_33591)
        # Adding element type (line 375)
        str_33592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 47), 'str', '/C')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 23), list_33586, str_33592)
        
        # Getting the type of 'argv' (line 375)
        argv_33593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 55), 'argv')
        # Applying the binary operator '+' (line 375)
        result_add_33594 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 23), '+', list_33586, argv_33593)
        
        # Assigning a type to the variable 'argv' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 16), 'argv', result_add_33594)
        
        # Assigning a Num to a Name (line 376):
        
        # Assigning a Num to a Name (line 376):
        int_33595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 32), 'int')
        # Assigning a type to the variable 'using_command' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 16), 'using_command', int_33595)
        # SSA join for if statement (line 373)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 371)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_33538 and more_types_in_union_33539):
            # SSA join for if statement (line 366)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 378):
    
    # Assigning a Call to a Name (line 378):
    
    # Call to _supports_fileno(...): (line 378)
    # Processing the call arguments (line 378)
    # Getting the type of 'sys' (line 378)
    sys_33597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 38), 'sys', False)
    # Obtaining the member 'stdout' of a type (line 378)
    stdout_33598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 38), sys_33597, 'stdout')
    # Processing the call keyword arguments (line 378)
    kwargs_33599 = {}
    # Getting the type of '_supports_fileno' (line 378)
    _supports_fileno_33596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 21), '_supports_fileno', False)
    # Calling _supports_fileno(args, kwargs) (line 378)
    _supports_fileno_call_result_33600 = invoke(stypy.reporting.localization.Localization(__file__, 378, 21), _supports_fileno_33596, *[stdout_33598], **kwargs_33599)
    
    # Assigning a type to the variable '_so_has_fileno' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), '_so_has_fileno', _supports_fileno_call_result_33600)
    
    # Assigning a Call to a Name (line 379):
    
    # Assigning a Call to a Name (line 379):
    
    # Call to _supports_fileno(...): (line 379)
    # Processing the call arguments (line 379)
    # Getting the type of 'sys' (line 379)
    sys_33602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 38), 'sys', False)
    # Obtaining the member 'stderr' of a type (line 379)
    stderr_33603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 38), sys_33602, 'stderr')
    # Processing the call keyword arguments (line 379)
    kwargs_33604 = {}
    # Getting the type of '_supports_fileno' (line 379)
    _supports_fileno_33601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 21), '_supports_fileno', False)
    # Calling _supports_fileno(args, kwargs) (line 379)
    _supports_fileno_call_result_33605 = invoke(stypy.reporting.localization.Localization(__file__, 379, 21), _supports_fileno_33601, *[stderr_33603], **kwargs_33604)
    
    # Assigning a type to the variable '_se_has_fileno' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), '_se_has_fileno', _supports_fileno_call_result_33605)
    
    # Assigning a Attribute to a Name (line 380):
    
    # Assigning a Attribute to a Name (line 380):
    # Getting the type of 'sys' (line 380)
    sys_33606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 15), 'sys')
    # Obtaining the member 'stdout' of a type (line 380)
    stdout_33607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 15), sys_33606, 'stdout')
    # Obtaining the member 'flush' of a type (line 380)
    flush_33608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 15), stdout_33607, 'flush')
    # Assigning a type to the variable 'so_flush' (line 380)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 4), 'so_flush', flush_33608)
    
    # Assigning a Attribute to a Name (line 381):
    
    # Assigning a Attribute to a Name (line 381):
    # Getting the type of 'sys' (line 381)
    sys_33609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 15), 'sys')
    # Obtaining the member 'stderr' of a type (line 381)
    stderr_33610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 15), sys_33609, 'stderr')
    # Obtaining the member 'flush' of a type (line 381)
    flush_33611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 15), stderr_33610, 'flush')
    # Assigning a type to the variable 'se_flush' (line 381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'se_flush', flush_33611)
    
    # Getting the type of '_so_has_fileno' (line 382)
    _so_has_fileno_33612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 7), '_so_has_fileno')
    # Testing the type of an if condition (line 382)
    if_condition_33613 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 382, 4), _so_has_fileno_33612)
    # Assigning a type to the variable 'if_condition_33613' (line 382)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'if_condition_33613', if_condition_33613)
    # SSA begins for if statement (line 382)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 383):
    
    # Assigning a Call to a Name (line 383):
    
    # Call to fileno(...): (line 383)
    # Processing the call keyword arguments (line 383)
    kwargs_33617 = {}
    # Getting the type of 'sys' (line 383)
    sys_33614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 20), 'sys', False)
    # Obtaining the member 'stdout' of a type (line 383)
    stdout_33615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 20), sys_33614, 'stdout')
    # Obtaining the member 'fileno' of a type (line 383)
    fileno_33616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 20), stdout_33615, 'fileno')
    # Calling fileno(args, kwargs) (line 383)
    fileno_call_result_33618 = invoke(stypy.reporting.localization.Localization(__file__, 383, 20), fileno_33616, *[], **kwargs_33617)
    
    # Assigning a type to the variable 'so_fileno' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'so_fileno', fileno_call_result_33618)
    
    # Assigning a Call to a Name (line 384):
    
    # Assigning a Call to a Name (line 384):
    
    # Call to dup(...): (line 384)
    # Processing the call arguments (line 384)
    # Getting the type of 'so_fileno' (line 384)
    so_fileno_33621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 24), 'so_fileno', False)
    # Processing the call keyword arguments (line 384)
    kwargs_33622 = {}
    # Getting the type of 'os' (line 384)
    os_33619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 17), 'os', False)
    # Obtaining the member 'dup' of a type (line 384)
    dup_33620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 17), os_33619, 'dup')
    # Calling dup(args, kwargs) (line 384)
    dup_call_result_33623 = invoke(stypy.reporting.localization.Localization(__file__, 384, 17), dup_33620, *[so_fileno_33621], **kwargs_33622)
    
    # Assigning a type to the variable 'so_dup' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'so_dup', dup_call_result_33623)
    # SSA join for if statement (line 382)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of '_se_has_fileno' (line 385)
    _se_has_fileno_33624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 7), '_se_has_fileno')
    # Testing the type of an if condition (line 385)
    if_condition_33625 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 385, 4), _se_has_fileno_33624)
    # Assigning a type to the variable 'if_condition_33625' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'if_condition_33625', if_condition_33625)
    # SSA begins for if statement (line 385)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 386):
    
    # Assigning a Call to a Name (line 386):
    
    # Call to fileno(...): (line 386)
    # Processing the call keyword arguments (line 386)
    kwargs_33629 = {}
    # Getting the type of 'sys' (line 386)
    sys_33626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 20), 'sys', False)
    # Obtaining the member 'stderr' of a type (line 386)
    stderr_33627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 20), sys_33626, 'stderr')
    # Obtaining the member 'fileno' of a type (line 386)
    fileno_33628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 20), stderr_33627, 'fileno')
    # Calling fileno(args, kwargs) (line 386)
    fileno_call_result_33630 = invoke(stypy.reporting.localization.Localization(__file__, 386, 20), fileno_33628, *[], **kwargs_33629)
    
    # Assigning a type to the variable 'se_fileno' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'se_fileno', fileno_call_result_33630)
    
    # Assigning a Call to a Name (line 387):
    
    # Assigning a Call to a Name (line 387):
    
    # Call to dup(...): (line 387)
    # Processing the call arguments (line 387)
    # Getting the type of 'se_fileno' (line 387)
    se_fileno_33633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 24), 'se_fileno', False)
    # Processing the call keyword arguments (line 387)
    kwargs_33634 = {}
    # Getting the type of 'os' (line 387)
    os_33631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 17), 'os', False)
    # Obtaining the member 'dup' of a type (line 387)
    dup_33632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 17), os_33631, 'dup')
    # Calling dup(args, kwargs) (line 387)
    dup_call_result_33635 = invoke(stypy.reporting.localization.Localization(__file__, 387, 17), dup_33632, *[se_fileno_33633], **kwargs_33634)
    
    # Assigning a type to the variable 'se_dup' (line 387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'se_dup', dup_call_result_33635)
    # SSA join for if statement (line 385)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 389):
    
    # Assigning a Call to a Name (line 389):
    
    # Call to temp_file_name(...): (line 389)
    # Processing the call keyword arguments (line 389)
    kwargs_33637 = {}
    # Getting the type of 'temp_file_name' (line 389)
    temp_file_name_33636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 14), 'temp_file_name', False)
    # Calling temp_file_name(args, kwargs) (line 389)
    temp_file_name_call_result_33638 = invoke(stypy.reporting.localization.Localization(__file__, 389, 14), temp_file_name_33636, *[], **kwargs_33637)
    
    # Assigning a type to the variable 'outfile' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'outfile', temp_file_name_call_result_33638)
    
    # Assigning a Call to a Name (line 390):
    
    # Assigning a Call to a Name (line 390):
    
    # Call to open(...): (line 390)
    # Processing the call arguments (line 390)
    # Getting the type of 'outfile' (line 390)
    outfile_33640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 16), 'outfile', False)
    str_33641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 25), 'str', 'w')
    # Processing the call keyword arguments (line 390)
    kwargs_33642 = {}
    # Getting the type of 'open' (line 390)
    open_33639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 11), 'open', False)
    # Calling open(args, kwargs) (line 390)
    open_call_result_33643 = invoke(stypy.reporting.localization.Localization(__file__, 390, 11), open_33639, *[outfile_33640, str_33641], **kwargs_33642)
    
    # Assigning a type to the variable 'fout' (line 390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'fout', open_call_result_33643)
    
    # Getting the type of 'using_command' (line 391)
    using_command_33644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 7), 'using_command')
    # Testing the type of an if condition (line 391)
    if_condition_33645 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 391, 4), using_command_33644)
    # Assigning a type to the variable 'if_condition_33645' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'if_condition_33645', if_condition_33645)
    # SSA begins for if statement (line 391)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 392):
    
    # Assigning a Call to a Name (line 392):
    
    # Call to temp_file_name(...): (line 392)
    # Processing the call keyword arguments (line 392)
    kwargs_33647 = {}
    # Getting the type of 'temp_file_name' (line 392)
    temp_file_name_33646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 18), 'temp_file_name', False)
    # Calling temp_file_name(args, kwargs) (line 392)
    temp_file_name_call_result_33648 = invoke(stypy.reporting.localization.Localization(__file__, 392, 18), temp_file_name_33646, *[], **kwargs_33647)
    
    # Assigning a type to the variable 'errfile' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'errfile', temp_file_name_call_result_33648)
    
    # Assigning a Call to a Name (line 393):
    
    # Assigning a Call to a Name (line 393):
    
    # Call to open(...): (line 393)
    # Processing the call arguments (line 393)
    # Getting the type of 'errfile' (line 393)
    errfile_33650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 20), 'errfile', False)
    str_33651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 29), 'str', 'w')
    # Processing the call keyword arguments (line 393)
    kwargs_33652 = {}
    # Getting the type of 'open' (line 393)
    open_33649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 15), 'open', False)
    # Calling open(args, kwargs) (line 393)
    open_call_result_33653 = invoke(stypy.reporting.localization.Localization(__file__, 393, 15), open_33649, *[errfile_33650, str_33651], **kwargs_33652)
    
    # Assigning a type to the variable 'ferr' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'ferr', open_call_result_33653)
    # SSA join for if statement (line 391)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to debug(...): (line 395)
    # Processing the call arguments (line 395)
    str_33656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 14), 'str', 'Running %s(%s,%r,%r,os.environ)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 396)
    tuple_33657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 396)
    # Adding element type (line 396)
    # Getting the type of 'spawn_command' (line 396)
    spawn_command_33658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 17), 'spawn_command', False)
    # Obtaining the member '__name__' of a type (line 396)
    name___33659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 17), spawn_command_33658, '__name__')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 17), tuple_33657, name___33659)
    # Adding element type (line 396)
    # Getting the type of 'os' (line 396)
    os_33660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 41), 'os', False)
    # Obtaining the member 'P_WAIT' of a type (line 396)
    P_WAIT_33661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 41), os_33660, 'P_WAIT')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 17), tuple_33657, P_WAIT_33661)
    # Adding element type (line 396)
    
    # Obtaining the type of the subscript
    int_33662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 57), 'int')
    # Getting the type of 'argv' (line 396)
    argv_33663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 52), 'argv', False)
    # Obtaining the member '__getitem__' of a type (line 396)
    getitem___33664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 52), argv_33663, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 396)
    subscript_call_result_33665 = invoke(stypy.reporting.localization.Localization(__file__, 396, 52), getitem___33664, int_33662)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 17), tuple_33657, subscript_call_result_33665)
    # Adding element type (line 396)
    # Getting the type of 'argv' (line 396)
    argv_33666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 61), 'argv', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 17), tuple_33657, argv_33666)
    
    # Applying the binary operator '%' (line 395)
    result_mod_33667 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 14), '%', str_33656, tuple_33657)
    
    # Processing the call keyword arguments (line 395)
    kwargs_33668 = {}
    # Getting the type of 'log' (line 395)
    log_33654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 4), 'log', False)
    # Obtaining the member 'debug' of a type (line 395)
    debug_33655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 4), log_33654, 'debug')
    # Calling debug(args, kwargs) (line 395)
    debug_call_result_33669 = invoke(stypy.reporting.localization.Localization(__file__, 395, 4), debug_33655, *[result_mod_33667], **kwargs_33668)
    
    
    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    int_33670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 24), 'int')
    # Getting the type of 'sys' (line 398)
    sys_33671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 7), 'sys')
    # Obtaining the member 'version_info' of a type (line 398)
    version_info_33672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 7), sys_33671, 'version_info')
    # Obtaining the member '__getitem__' of a type (line 398)
    getitem___33673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 7), version_info_33672, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 398)
    subscript_call_result_33674 = invoke(stypy.reporting.localization.Localization(__file__, 398, 7), getitem___33673, int_33670)
    
    int_33675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 30), 'int')
    # Applying the binary operator '>=' (line 398)
    result_ge_33676 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 7), '>=', subscript_call_result_33674, int_33675)
    
    
    # Getting the type of 'os' (line 398)
    os_33677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 36), 'os')
    # Obtaining the member 'name' of a type (line 398)
    name_33678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 36), os_33677, 'name')
    str_33679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 47), 'str', 'nt')
    # Applying the binary operator '==' (line 398)
    result_eq_33680 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 36), '==', name_33678, str_33679)
    
    # Applying the binary operator 'and' (line 398)
    result_and_keyword_33681 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 7), 'and', result_ge_33676, result_eq_33680)
    
    # Testing the type of an if condition (line 398)
    if_condition_33682 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 398, 4), result_and_keyword_33681)
    # Assigning a type to the variable 'if_condition_33682' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'if_condition_33682', if_condition_33682)
    # SSA begins for if statement (line 398)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Dict to a Name (line 407):
    
    # Assigning a Dict to a Name (line 407):
    
    # Obtaining an instance of the builtin type 'dict' (line 407)
    dict_33683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 26), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 407)
    
    # Assigning a type to the variable 'encoded_environ' (line 407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'encoded_environ', dict_33683)
    
    
    # Call to items(...): (line 408)
    # Processing the call keyword arguments (line 408)
    kwargs_33687 = {}
    # Getting the type of 'os' (line 408)
    os_33684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 20), 'os', False)
    # Obtaining the member 'environ' of a type (line 408)
    environ_33685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 20), os_33684, 'environ')
    # Obtaining the member 'items' of a type (line 408)
    items_33686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 20), environ_33685, 'items')
    # Calling items(args, kwargs) (line 408)
    items_call_result_33688 = invoke(stypy.reporting.localization.Localization(__file__, 408, 20), items_33686, *[], **kwargs_33687)
    
    # Testing the type of a for loop iterable (line 408)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 408, 8), items_call_result_33688)
    # Getting the type of the for loop variable (line 408)
    for_loop_var_33689 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 408, 8), items_call_result_33688)
    # Assigning a type to the variable 'k' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 8), for_loop_var_33689))
    # Assigning a type to the variable 'v' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 8), for_loop_var_33689))
    # SSA begins for a for statement (line 408)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # SSA begins for try-except statement (line 409)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Subscript (line 410):
    
    # Assigning a Call to a Subscript (line 410):
    
    # Call to encode(...): (line 410)
    # Processing the call arguments (line 410)
    
    # Call to getfilesystemencoding(...): (line 411)
    # Processing the call keyword arguments (line 411)
    kwargs_33694 = {}
    # Getting the type of 'sys' (line 411)
    sys_33692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 20), 'sys', False)
    # Obtaining the member 'getfilesystemencoding' of a type (line 411)
    getfilesystemencoding_33693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 20), sys_33692, 'getfilesystemencoding')
    # Calling getfilesystemencoding(args, kwargs) (line 411)
    getfilesystemencoding_call_result_33695 = invoke(stypy.reporting.localization.Localization(__file__, 411, 20), getfilesystemencoding_33693, *[], **kwargs_33694)
    
    # Processing the call keyword arguments (line 410)
    kwargs_33696 = {}
    # Getting the type of 'v' (line 410)
    v_33690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 73), 'v', False)
    # Obtaining the member 'encode' of a type (line 410)
    encode_33691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 73), v_33690, 'encode')
    # Calling encode(args, kwargs) (line 410)
    encode_call_result_33697 = invoke(stypy.reporting.localization.Localization(__file__, 410, 73), encode_33691, *[getfilesystemencoding_call_result_33695], **kwargs_33696)
    
    # Getting the type of 'encoded_environ' (line 410)
    encoded_environ_33698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 16), 'encoded_environ')
    
    # Call to encode(...): (line 410)
    # Processing the call arguments (line 410)
    
    # Call to getfilesystemencoding(...): (line 410)
    # Processing the call keyword arguments (line 410)
    kwargs_33703 = {}
    # Getting the type of 'sys' (line 410)
    sys_33701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 41), 'sys', False)
    # Obtaining the member 'getfilesystemencoding' of a type (line 410)
    getfilesystemencoding_33702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 41), sys_33701, 'getfilesystemencoding')
    # Calling getfilesystemencoding(args, kwargs) (line 410)
    getfilesystemencoding_call_result_33704 = invoke(stypy.reporting.localization.Localization(__file__, 410, 41), getfilesystemencoding_33702, *[], **kwargs_33703)
    
    # Processing the call keyword arguments (line 410)
    kwargs_33705 = {}
    # Getting the type of 'k' (line 410)
    k_33699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 32), 'k', False)
    # Obtaining the member 'encode' of a type (line 410)
    encode_33700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 32), k_33699, 'encode')
    # Calling encode(args, kwargs) (line 410)
    encode_call_result_33706 = invoke(stypy.reporting.localization.Localization(__file__, 410, 32), encode_33700, *[getfilesystemencoding_call_result_33704], **kwargs_33705)
    
    # Storing an element on a container (line 410)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 16), encoded_environ_33698, (encode_call_result_33706, encode_call_result_33697))
    # SSA branch for the except part of a try statement (line 409)
    # SSA branch for the except 'UnicodeEncodeError' branch of a try statement (line 409)
    module_type_store.open_ssa_branch('except')
    
    # Call to debug(...): (line 413)
    # Processing the call arguments (line 413)
    str_33709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 26), 'str', 'ignoring un-encodable env entry %s')
    # Getting the type of 'k' (line 413)
    k_33710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 64), 'k', False)
    # Processing the call keyword arguments (line 413)
    kwargs_33711 = {}
    # Getting the type of 'log' (line 413)
    log_33707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 16), 'log', False)
    # Obtaining the member 'debug' of a type (line 413)
    debug_33708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 16), log_33707, 'debug')
    # Calling debug(args, kwargs) (line 413)
    debug_call_result_33712 = invoke(stypy.reporting.localization.Localization(__file__, 413, 16), debug_33708, *[str_33709, k_33710], **kwargs_33711)
    
    # SSA join for try-except statement (line 409)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 398)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Attribute to a Name (line 415):
    
    # Assigning a Attribute to a Name (line 415):
    # Getting the type of 'os' (line 415)
    os_33713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 26), 'os')
    # Obtaining the member 'environ' of a type (line 415)
    environ_33714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 26), os_33713, 'environ')
    # Assigning a type to the variable 'encoded_environ' (line 415)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'encoded_environ', environ_33714)
    # SSA join for if statement (line 398)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 417):
    
    # Assigning a Subscript to a Name (line 417):
    
    # Obtaining the type of the subscript
    int_33715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 17), 'int')
    # Getting the type of 'argv' (line 417)
    argv_33716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 12), 'argv')
    # Obtaining the member '__getitem__' of a type (line 417)
    getitem___33717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 12), argv_33716, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 417)
    subscript_call_result_33718 = invoke(stypy.reporting.localization.Localization(__file__, 417, 12), getitem___33717, int_33715)
    
    # Assigning a type to the variable 'argv0' (line 417)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'argv0', subscript_call_result_33718)
    
    
    # Getting the type of 'using_command' (line 418)
    using_command_33719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 11), 'using_command')
    # Applying the 'not' unary operator (line 418)
    result_not__33720 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 7), 'not', using_command_33719)
    
    # Testing the type of an if condition (line 418)
    if_condition_33721 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 418, 4), result_not__33720)
    # Assigning a type to the variable 'if_condition_33721' (line 418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'if_condition_33721', if_condition_33721)
    # SSA begins for if statement (line 418)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 419):
    
    # Assigning a Call to a Subscript (line 419):
    
    # Call to quote_arg(...): (line 419)
    # Processing the call arguments (line 419)
    # Getting the type of 'argv0' (line 419)
    argv0_33723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 28), 'argv0', False)
    # Processing the call keyword arguments (line 419)
    kwargs_33724 = {}
    # Getting the type of 'quote_arg' (line 419)
    quote_arg_33722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 18), 'quote_arg', False)
    # Calling quote_arg(args, kwargs) (line 419)
    quote_arg_call_result_33725 = invoke(stypy.reporting.localization.Localization(__file__, 419, 18), quote_arg_33722, *[argv0_33723], **kwargs_33724)
    
    # Getting the type of 'argv' (line 419)
    argv_33726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'argv')
    int_33727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 13), 'int')
    # Storing an element on a container (line 419)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 8), argv_33726, (int_33727, quote_arg_call_result_33725))
    # SSA join for if statement (line 418)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to so_flush(...): (line 421)
    # Processing the call keyword arguments (line 421)
    kwargs_33729 = {}
    # Getting the type of 'so_flush' (line 421)
    so_flush_33728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 4), 'so_flush', False)
    # Calling so_flush(args, kwargs) (line 421)
    so_flush_call_result_33730 = invoke(stypy.reporting.localization.Localization(__file__, 421, 4), so_flush_33728, *[], **kwargs_33729)
    
    
    # Call to se_flush(...): (line 422)
    # Processing the call keyword arguments (line 422)
    kwargs_33732 = {}
    # Getting the type of 'se_flush' (line 422)
    se_flush_33731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'se_flush', False)
    # Calling se_flush(args, kwargs) (line 422)
    se_flush_call_result_33733 = invoke(stypy.reporting.localization.Localization(__file__, 422, 4), se_flush_33731, *[], **kwargs_33732)
    
    
    # Getting the type of '_so_has_fileno' (line 423)
    _so_has_fileno_33734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 7), '_so_has_fileno')
    # Testing the type of an if condition (line 423)
    if_condition_33735 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 423, 4), _so_has_fileno_33734)
    # Assigning a type to the variable 'if_condition_33735' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'if_condition_33735', if_condition_33735)
    # SSA begins for if statement (line 423)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to dup2(...): (line 424)
    # Processing the call arguments (line 424)
    
    # Call to fileno(...): (line 424)
    # Processing the call keyword arguments (line 424)
    kwargs_33740 = {}
    # Getting the type of 'fout' (line 424)
    fout_33738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 16), 'fout', False)
    # Obtaining the member 'fileno' of a type (line 424)
    fileno_33739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 16), fout_33738, 'fileno')
    # Calling fileno(args, kwargs) (line 424)
    fileno_call_result_33741 = invoke(stypy.reporting.localization.Localization(__file__, 424, 16), fileno_33739, *[], **kwargs_33740)
    
    # Getting the type of 'so_fileno' (line 424)
    so_fileno_33742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 31), 'so_fileno', False)
    # Processing the call keyword arguments (line 424)
    kwargs_33743 = {}
    # Getting the type of 'os' (line 424)
    os_33736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'os', False)
    # Obtaining the member 'dup2' of a type (line 424)
    dup2_33737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 8), os_33736, 'dup2')
    # Calling dup2(args, kwargs) (line 424)
    dup2_call_result_33744 = invoke(stypy.reporting.localization.Localization(__file__, 424, 8), dup2_33737, *[fileno_call_result_33741, so_fileno_33742], **kwargs_33743)
    
    # SSA join for if statement (line 423)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of '_se_has_fileno' (line 426)
    _se_has_fileno_33745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 7), '_se_has_fileno')
    # Testing the type of an if condition (line 426)
    if_condition_33746 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 426, 4), _se_has_fileno_33745)
    # Assigning a type to the variable 'if_condition_33746' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'if_condition_33746', if_condition_33746)
    # SSA begins for if statement (line 426)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'using_command' (line 427)
    using_command_33747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 11), 'using_command')
    # Testing the type of an if condition (line 427)
    if_condition_33748 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 427, 8), using_command_33747)
    # Assigning a type to the variable 'if_condition_33748' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'if_condition_33748', if_condition_33748)
    # SSA begins for if statement (line 427)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to dup2(...): (line 430)
    # Processing the call arguments (line 430)
    
    # Call to fileno(...): (line 430)
    # Processing the call keyword arguments (line 430)
    kwargs_33753 = {}
    # Getting the type of 'ferr' (line 430)
    ferr_33751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 20), 'ferr', False)
    # Obtaining the member 'fileno' of a type (line 430)
    fileno_33752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 20), ferr_33751, 'fileno')
    # Calling fileno(args, kwargs) (line 430)
    fileno_call_result_33754 = invoke(stypy.reporting.localization.Localization(__file__, 430, 20), fileno_33752, *[], **kwargs_33753)
    
    # Getting the type of 'se_fileno' (line 430)
    se_fileno_33755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 35), 'se_fileno', False)
    # Processing the call keyword arguments (line 430)
    kwargs_33756 = {}
    # Getting the type of 'os' (line 430)
    os_33749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 12), 'os', False)
    # Obtaining the member 'dup2' of a type (line 430)
    dup2_33750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 12), os_33749, 'dup2')
    # Calling dup2(args, kwargs) (line 430)
    dup2_call_result_33757 = invoke(stypy.reporting.localization.Localization(__file__, 430, 12), dup2_33750, *[fileno_call_result_33754, se_fileno_33755], **kwargs_33756)
    
    # SSA branch for the else part of an if statement (line 427)
    module_type_store.open_ssa_branch('else')
    
    # Call to dup2(...): (line 432)
    # Processing the call arguments (line 432)
    
    # Call to fileno(...): (line 432)
    # Processing the call keyword arguments (line 432)
    kwargs_33762 = {}
    # Getting the type of 'fout' (line 432)
    fout_33760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 20), 'fout', False)
    # Obtaining the member 'fileno' of a type (line 432)
    fileno_33761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 20), fout_33760, 'fileno')
    # Calling fileno(args, kwargs) (line 432)
    fileno_call_result_33763 = invoke(stypy.reporting.localization.Localization(__file__, 432, 20), fileno_33761, *[], **kwargs_33762)
    
    # Getting the type of 'se_fileno' (line 432)
    se_fileno_33764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 35), 'se_fileno', False)
    # Processing the call keyword arguments (line 432)
    kwargs_33765 = {}
    # Getting the type of 'os' (line 432)
    os_33758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 12), 'os', False)
    # Obtaining the member 'dup2' of a type (line 432)
    dup2_33759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 12), os_33758, 'dup2')
    # Calling dup2(args, kwargs) (line 432)
    dup2_call_result_33766 = invoke(stypy.reporting.localization.Localization(__file__, 432, 12), dup2_33759, *[fileno_call_result_33763, se_fileno_33764], **kwargs_33765)
    
    # SSA join for if statement (line 427)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 426)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 433)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 434):
    
    # Assigning a Call to a Name (line 434):
    
    # Call to spawn_command(...): (line 434)
    # Processing the call arguments (line 434)
    # Getting the type of 'os' (line 434)
    os_33768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 31), 'os', False)
    # Obtaining the member 'P_WAIT' of a type (line 434)
    P_WAIT_33769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 31), os_33768, 'P_WAIT')
    # Getting the type of 'argv0' (line 434)
    argv0_33770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 42), 'argv0', False)
    # Getting the type of 'argv' (line 434)
    argv_33771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 49), 'argv', False)
    # Getting the type of 'encoded_environ' (line 434)
    encoded_environ_33772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 55), 'encoded_environ', False)
    # Processing the call keyword arguments (line 434)
    kwargs_33773 = {}
    # Getting the type of 'spawn_command' (line 434)
    spawn_command_33767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 17), 'spawn_command', False)
    # Calling spawn_command(args, kwargs) (line 434)
    spawn_command_call_result_33774 = invoke(stypy.reporting.localization.Localization(__file__, 434, 17), spawn_command_33767, *[P_WAIT_33769, argv0_33770, argv_33771, encoded_environ_33772], **kwargs_33773)
    
    # Assigning a type to the variable 'status' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'status', spawn_command_call_result_33774)
    # SSA branch for the except part of a try statement (line 433)
    # SSA branch for the except 'Exception' branch of a try statement (line 433)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Name (line 436):
    
    # Assigning a Call to a Name (line 436):
    
    # Call to str(...): (line 436)
    # Processing the call arguments (line 436)
    
    # Call to get_exception(...): (line 436)
    # Processing the call keyword arguments (line 436)
    kwargs_33777 = {}
    # Getting the type of 'get_exception' (line 436)
    get_exception_33776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 22), 'get_exception', False)
    # Calling get_exception(args, kwargs) (line 436)
    get_exception_call_result_33778 = invoke(stypy.reporting.localization.Localization(__file__, 436, 22), get_exception_33776, *[], **kwargs_33777)
    
    # Processing the call keyword arguments (line 436)
    kwargs_33779 = {}
    # Getting the type of 'str' (line 436)
    str_33775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 18), 'str', False)
    # Calling str(args, kwargs) (line 436)
    str_call_result_33780 = invoke(stypy.reporting.localization.Localization(__file__, 436, 18), str_33775, *[get_exception_call_result_33778], **kwargs_33779)
    
    # Assigning a type to the variable 'errmess' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'errmess', str_call_result_33780)
    
    # Assigning a Num to a Name (line 437):
    
    # Assigning a Num to a Name (line 437):
    int_33781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 17), 'int')
    # Assigning a type to the variable 'status' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'status', int_33781)
    
    # Call to write(...): (line 438)
    # Processing the call arguments (line 438)
    str_33785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 25), 'str', '%s: %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 438)
    tuple_33786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 438)
    # Adding element type (line 438)
    # Getting the type of 'errmess' (line 438)
    errmess_33787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 35), 'errmess', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 35), tuple_33786, errmess_33787)
    # Adding element type (line 438)
    
    # Obtaining the type of the subscript
    int_33788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 49), 'int')
    # Getting the type of 'argv' (line 438)
    argv_33789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 44), 'argv', False)
    # Obtaining the member '__getitem__' of a type (line 438)
    getitem___33790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 44), argv_33789, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 438)
    subscript_call_result_33791 = invoke(stypy.reporting.localization.Localization(__file__, 438, 44), getitem___33790, int_33788)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 35), tuple_33786, subscript_call_result_33791)
    
    # Applying the binary operator '%' (line 438)
    result_mod_33792 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 25), '%', str_33785, tuple_33786)
    
    # Processing the call keyword arguments (line 438)
    kwargs_33793 = {}
    # Getting the type of 'sys' (line 438)
    sys_33782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'sys', False)
    # Obtaining the member 'stderr' of a type (line 438)
    stderr_33783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 8), sys_33782, 'stderr')
    # Obtaining the member 'write' of a type (line 438)
    write_33784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 8), stderr_33783, 'write')
    # Calling write(args, kwargs) (line 438)
    write_call_result_33794 = invoke(stypy.reporting.localization.Localization(__file__, 438, 8), write_33784, *[result_mod_33792], **kwargs_33793)
    
    # SSA join for try-except statement (line 433)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to so_flush(...): (line 440)
    # Processing the call keyword arguments (line 440)
    kwargs_33796 = {}
    # Getting the type of 'so_flush' (line 440)
    so_flush_33795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'so_flush', False)
    # Calling so_flush(args, kwargs) (line 440)
    so_flush_call_result_33797 = invoke(stypy.reporting.localization.Localization(__file__, 440, 4), so_flush_33795, *[], **kwargs_33796)
    
    
    # Call to se_flush(...): (line 441)
    # Processing the call keyword arguments (line 441)
    kwargs_33799 = {}
    # Getting the type of 'se_flush' (line 441)
    se_flush_33798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'se_flush', False)
    # Calling se_flush(args, kwargs) (line 441)
    se_flush_call_result_33800 = invoke(stypy.reporting.localization.Localization(__file__, 441, 4), se_flush_33798, *[], **kwargs_33799)
    
    
    # Getting the type of '_so_has_fileno' (line 442)
    _so_has_fileno_33801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 7), '_so_has_fileno')
    # Testing the type of an if condition (line 442)
    if_condition_33802 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 442, 4), _so_has_fileno_33801)
    # Assigning a type to the variable 'if_condition_33802' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'if_condition_33802', if_condition_33802)
    # SSA begins for if statement (line 442)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to dup2(...): (line 443)
    # Processing the call arguments (line 443)
    # Getting the type of 'so_dup' (line 443)
    so_dup_33805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 16), 'so_dup', False)
    # Getting the type of 'so_fileno' (line 443)
    so_fileno_33806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 24), 'so_fileno', False)
    # Processing the call keyword arguments (line 443)
    kwargs_33807 = {}
    # Getting the type of 'os' (line 443)
    os_33803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'os', False)
    # Obtaining the member 'dup2' of a type (line 443)
    dup2_33804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 8), os_33803, 'dup2')
    # Calling dup2(args, kwargs) (line 443)
    dup2_call_result_33808 = invoke(stypy.reporting.localization.Localization(__file__, 443, 8), dup2_33804, *[so_dup_33805, so_fileno_33806], **kwargs_33807)
    
    
    # Call to close(...): (line 444)
    # Processing the call arguments (line 444)
    # Getting the type of 'so_dup' (line 444)
    so_dup_33811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 17), 'so_dup', False)
    # Processing the call keyword arguments (line 444)
    kwargs_33812 = {}
    # Getting the type of 'os' (line 444)
    os_33809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'os', False)
    # Obtaining the member 'close' of a type (line 444)
    close_33810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 8), os_33809, 'close')
    # Calling close(args, kwargs) (line 444)
    close_call_result_33813 = invoke(stypy.reporting.localization.Localization(__file__, 444, 8), close_33810, *[so_dup_33811], **kwargs_33812)
    
    # SSA join for if statement (line 442)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of '_se_has_fileno' (line 445)
    _se_has_fileno_33814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 7), '_se_has_fileno')
    # Testing the type of an if condition (line 445)
    if_condition_33815 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 445, 4), _se_has_fileno_33814)
    # Assigning a type to the variable 'if_condition_33815' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'if_condition_33815', if_condition_33815)
    # SSA begins for if statement (line 445)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to dup2(...): (line 446)
    # Processing the call arguments (line 446)
    # Getting the type of 'se_dup' (line 446)
    se_dup_33818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 16), 'se_dup', False)
    # Getting the type of 'se_fileno' (line 446)
    se_fileno_33819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 24), 'se_fileno', False)
    # Processing the call keyword arguments (line 446)
    kwargs_33820 = {}
    # Getting the type of 'os' (line 446)
    os_33816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'os', False)
    # Obtaining the member 'dup2' of a type (line 446)
    dup2_33817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 8), os_33816, 'dup2')
    # Calling dup2(args, kwargs) (line 446)
    dup2_call_result_33821 = invoke(stypy.reporting.localization.Localization(__file__, 446, 8), dup2_33817, *[se_dup_33818, se_fileno_33819], **kwargs_33820)
    
    
    # Call to close(...): (line 447)
    # Processing the call arguments (line 447)
    # Getting the type of 'se_dup' (line 447)
    se_dup_33824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 17), 'se_dup', False)
    # Processing the call keyword arguments (line 447)
    kwargs_33825 = {}
    # Getting the type of 'os' (line 447)
    os_33822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'os', False)
    # Obtaining the member 'close' of a type (line 447)
    close_33823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 8), os_33822, 'close')
    # Calling close(args, kwargs) (line 447)
    close_call_result_33826 = invoke(stypy.reporting.localization.Localization(__file__, 447, 8), close_33823, *[se_dup_33824], **kwargs_33825)
    
    # SSA join for if statement (line 445)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to close(...): (line 449)
    # Processing the call keyword arguments (line 449)
    kwargs_33829 = {}
    # Getting the type of 'fout' (line 449)
    fout_33827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 4), 'fout', False)
    # Obtaining the member 'close' of a type (line 449)
    close_33828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 4), fout_33827, 'close')
    # Calling close(args, kwargs) (line 449)
    close_call_result_33830 = invoke(stypy.reporting.localization.Localization(__file__, 449, 4), close_33828, *[], **kwargs_33829)
    
    
    # Assigning a Call to a Name (line 450):
    
    # Assigning a Call to a Name (line 450):
    
    # Call to open_latin1(...): (line 450)
    # Processing the call arguments (line 450)
    # Getting the type of 'outfile' (line 450)
    outfile_33832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 23), 'outfile', False)
    str_33833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 32), 'str', 'r')
    # Processing the call keyword arguments (line 450)
    kwargs_33834 = {}
    # Getting the type of 'open_latin1' (line 450)
    open_latin1_33831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 11), 'open_latin1', False)
    # Calling open_latin1(args, kwargs) (line 450)
    open_latin1_call_result_33835 = invoke(stypy.reporting.localization.Localization(__file__, 450, 11), open_latin1_33831, *[outfile_33832, str_33833], **kwargs_33834)
    
    # Assigning a type to the variable 'fout' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'fout', open_latin1_call_result_33835)
    
    # Assigning a Call to a Name (line 451):
    
    # Assigning a Call to a Name (line 451):
    
    # Call to read(...): (line 451)
    # Processing the call keyword arguments (line 451)
    kwargs_33838 = {}
    # Getting the type of 'fout' (line 451)
    fout_33836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 11), 'fout', False)
    # Obtaining the member 'read' of a type (line 451)
    read_33837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 11), fout_33836, 'read')
    # Calling read(args, kwargs) (line 451)
    read_call_result_33839 = invoke(stypy.reporting.localization.Localization(__file__, 451, 11), read_33837, *[], **kwargs_33838)
    
    # Assigning a type to the variable 'text' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'text', read_call_result_33839)
    
    # Call to close(...): (line 452)
    # Processing the call keyword arguments (line 452)
    kwargs_33842 = {}
    # Getting the type of 'fout' (line 452)
    fout_33840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'fout', False)
    # Obtaining the member 'close' of a type (line 452)
    close_33841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 4), fout_33840, 'close')
    # Calling close(args, kwargs) (line 452)
    close_call_result_33843 = invoke(stypy.reporting.localization.Localization(__file__, 452, 4), close_33841, *[], **kwargs_33842)
    
    
    # Call to remove(...): (line 453)
    # Processing the call arguments (line 453)
    # Getting the type of 'outfile' (line 453)
    outfile_33846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 14), 'outfile', False)
    # Processing the call keyword arguments (line 453)
    kwargs_33847 = {}
    # Getting the type of 'os' (line 453)
    os_33844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'os', False)
    # Obtaining the member 'remove' of a type (line 453)
    remove_33845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 4), os_33844, 'remove')
    # Calling remove(args, kwargs) (line 453)
    remove_call_result_33848 = invoke(stypy.reporting.localization.Localization(__file__, 453, 4), remove_33845, *[outfile_33846], **kwargs_33847)
    
    
    # Getting the type of 'using_command' (line 455)
    using_command_33849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 7), 'using_command')
    # Testing the type of an if condition (line 455)
    if_condition_33850 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 455, 4), using_command_33849)
    # Assigning a type to the variable 'if_condition_33850' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'if_condition_33850', if_condition_33850)
    # SSA begins for if statement (line 455)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to close(...): (line 456)
    # Processing the call keyword arguments (line 456)
    kwargs_33853 = {}
    # Getting the type of 'ferr' (line 456)
    ferr_33851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'ferr', False)
    # Obtaining the member 'close' of a type (line 456)
    close_33852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 8), ferr_33851, 'close')
    # Calling close(args, kwargs) (line 456)
    close_call_result_33854 = invoke(stypy.reporting.localization.Localization(__file__, 456, 8), close_33852, *[], **kwargs_33853)
    
    
    # Assigning a Call to a Name (line 457):
    
    # Assigning a Call to a Name (line 457):
    
    # Call to open_latin1(...): (line 457)
    # Processing the call arguments (line 457)
    # Getting the type of 'errfile' (line 457)
    errfile_33856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 27), 'errfile', False)
    str_33857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 36), 'str', 'r')
    # Processing the call keyword arguments (line 457)
    kwargs_33858 = {}
    # Getting the type of 'open_latin1' (line 457)
    open_latin1_33855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 15), 'open_latin1', False)
    # Calling open_latin1(args, kwargs) (line 457)
    open_latin1_call_result_33859 = invoke(stypy.reporting.localization.Localization(__file__, 457, 15), open_latin1_33855, *[errfile_33856, str_33857], **kwargs_33858)
    
    # Assigning a type to the variable 'ferr' (line 457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'ferr', open_latin1_call_result_33859)
    
    # Assigning a Call to a Name (line 458):
    
    # Assigning a Call to a Name (line 458):
    
    # Call to read(...): (line 458)
    # Processing the call keyword arguments (line 458)
    kwargs_33862 = {}
    # Getting the type of 'ferr' (line 458)
    ferr_33860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 18), 'ferr', False)
    # Obtaining the member 'read' of a type (line 458)
    read_33861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 18), ferr_33860, 'read')
    # Calling read(args, kwargs) (line 458)
    read_call_result_33863 = invoke(stypy.reporting.localization.Localization(__file__, 458, 18), read_33861, *[], **kwargs_33862)
    
    # Assigning a type to the variable 'errmess' (line 458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'errmess', read_call_result_33863)
    
    # Call to close(...): (line 459)
    # Processing the call keyword arguments (line 459)
    kwargs_33866 = {}
    # Getting the type of 'ferr' (line 459)
    ferr_33864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'ferr', False)
    # Obtaining the member 'close' of a type (line 459)
    close_33865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 8), ferr_33864, 'close')
    # Calling close(args, kwargs) (line 459)
    close_call_result_33867 = invoke(stypy.reporting.localization.Localization(__file__, 459, 8), close_33865, *[], **kwargs_33866)
    
    
    # Call to remove(...): (line 460)
    # Processing the call arguments (line 460)
    # Getting the type of 'errfile' (line 460)
    errfile_33870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 18), 'errfile', False)
    # Processing the call keyword arguments (line 460)
    kwargs_33871 = {}
    # Getting the type of 'os' (line 460)
    os_33868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'os', False)
    # Obtaining the member 'remove' of a type (line 460)
    remove_33869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 8), os_33868, 'remove')
    # Calling remove(args, kwargs) (line 460)
    remove_call_result_33872 = invoke(stypy.reporting.localization.Localization(__file__, 460, 8), remove_33869, *[errfile_33870], **kwargs_33871)
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'errmess' (line 461)
    errmess_33873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 11), 'errmess')
    
    # Getting the type of 'status' (line 461)
    status_33874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 27), 'status')
    # Applying the 'not' unary operator (line 461)
    result_not__33875 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 23), 'not', status_33874)
    
    # Applying the binary operator 'and' (line 461)
    result_and_keyword_33876 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 11), 'and', errmess_33873, result_not__33875)
    
    # Testing the type of an if condition (line 461)
    if_condition_33877 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 461, 8), result_and_keyword_33876)
    # Assigning a type to the variable 'if_condition_33877' (line 461)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'if_condition_33877', if_condition_33877)
    # SSA begins for if statement (line 461)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'text' (line 466)
    text_33878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 15), 'text')
    # Testing the type of an if condition (line 466)
    if_condition_33879 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 466, 12), text_33878)
    # Assigning a type to the variable 'if_condition_33879' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'if_condition_33879', if_condition_33879)
    # SSA begins for if statement (line 466)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 467):
    
    # Assigning a BinOp to a Name (line 467):
    # Getting the type of 'text' (line 467)
    text_33880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 23), 'text')
    str_33881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 30), 'str', '\n')
    # Applying the binary operator '+' (line 467)
    result_add_33882 = python_operator(stypy.reporting.localization.Localization(__file__, 467, 23), '+', text_33880, str_33881)
    
    # Assigning a type to the variable 'text' (line 467)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 16), 'text', result_add_33882)
    # SSA join for if statement (line 466)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 469):
    
    # Assigning a BinOp to a Name (line 469):
    # Getting the type of 'text' (line 469)
    text_33883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 19), 'text')
    # Getting the type of 'errmess' (line 469)
    errmess_33884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 26), 'errmess')
    # Applying the binary operator '+' (line 469)
    result_add_33885 = python_operator(stypy.reporting.localization.Localization(__file__, 469, 19), '+', text_33883, errmess_33884)
    
    # Assigning a type to the variable 'text' (line 469)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 12), 'text', result_add_33885)
    
    # Call to print(...): (line 470)
    # Processing the call arguments (line 470)
    # Getting the type of 'errmess' (line 470)
    errmess_33887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 19), 'errmess', False)
    # Processing the call keyword arguments (line 470)
    kwargs_33888 = {}
    # Getting the type of 'print' (line 470)
    print_33886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'print', False)
    # Calling print(args, kwargs) (line 470)
    print_call_result_33889 = invoke(stypy.reporting.localization.Localization(__file__, 470, 12), print_33886, *[errmess_33887], **kwargs_33888)
    
    # SSA join for if statement (line 461)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 455)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_33890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 12), 'int')
    slice_33891 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 471, 7), int_33890, None, None)
    # Getting the type of 'text' (line 471)
    text_33892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 7), 'text')
    # Obtaining the member '__getitem__' of a type (line 471)
    getitem___33893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 7), text_33892, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 471)
    subscript_call_result_33894 = invoke(stypy.reporting.localization.Localization(__file__, 471, 7), getitem___33893, slice_33891)
    
    str_33895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 18), 'str', '\n')
    # Applying the binary operator '==' (line 471)
    result_eq_33896 = python_operator(stypy.reporting.localization.Localization(__file__, 471, 7), '==', subscript_call_result_33894, str_33895)
    
    # Testing the type of an if condition (line 471)
    if_condition_33897 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 471, 4), result_eq_33896)
    # Assigning a type to the variable 'if_condition_33897' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'if_condition_33897', if_condition_33897)
    # SSA begins for if statement (line 471)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 472):
    
    # Assigning a Subscript to a Name (line 472):
    
    # Obtaining the type of the subscript
    int_33898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 21), 'int')
    slice_33899 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 472, 15), None, int_33898, None)
    # Getting the type of 'text' (line 472)
    text_33900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 15), 'text')
    # Obtaining the member '__getitem__' of a type (line 472)
    getitem___33901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 15), text_33900, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 472)
    subscript_call_result_33902 = invoke(stypy.reporting.localization.Localization(__file__, 472, 15), getitem___33901, slice_33899)
    
    # Assigning a type to the variable 'text' (line 472)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'text', subscript_call_result_33902)
    # SSA join for if statement (line 471)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 473)
    # Getting the type of 'status' (line 473)
    status_33903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 7), 'status')
    # Getting the type of 'None' (line 473)
    None_33904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 17), 'None')
    
    (may_be_33905, more_types_in_union_33906) = may_be_none(status_33903, None_33904)

    if may_be_33905:

        if more_types_in_union_33906:
            # Runtime conditional SSA (line 473)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Num to a Name (line 474):
        
        # Assigning a Num to a Name (line 474):
        int_33907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 17), 'int')
        # Assigning a type to the variable 'status' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'status', int_33907)

        if more_types_in_union_33906:
            # SSA join for if statement (line 473)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'use_tee' (line 476)
    use_tee_33908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 7), 'use_tee')
    # Testing the type of an if condition (line 476)
    if_condition_33909 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 476, 4), use_tee_33908)
    # Assigning a type to the variable 'if_condition_33909' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'if_condition_33909', if_condition_33909)
    # SSA begins for if statement (line 476)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 477)
    # Processing the call arguments (line 477)
    # Getting the type of 'text' (line 477)
    text_33911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 15), 'text', False)
    # Processing the call keyword arguments (line 477)
    kwargs_33912 = {}
    # Getting the type of 'print' (line 477)
    print_33910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'print', False)
    # Calling print(args, kwargs) (line 477)
    print_call_result_33913 = invoke(stypy.reporting.localization.Localization(__file__, 477, 8), print_33910, *[text_33911], **kwargs_33912)
    
    # SSA join for if statement (line 476)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 479)
    tuple_33914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 479)
    # Adding element type (line 479)
    # Getting the type of 'status' (line 479)
    status_33915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 11), 'status')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 11), tuple_33914, status_33915)
    # Adding element type (line 479)
    # Getting the type of 'text' (line 479)
    text_33916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 19), 'text')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 11), tuple_33914, text_33916)
    
    # Assigning a type to the variable 'stypy_return_type' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 4), 'stypy_return_type', tuple_33914)
    
    # ################# End of '_exec_command(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_exec_command' in the type store
    # Getting the type of 'stypy_return_type' (line 342)
    stypy_return_type_33917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_33917)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_exec_command'
    return stypy_return_type_33917

# Assigning a type to the variable '_exec_command' (line 342)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 0), '_exec_command', _exec_command)

@norecursion
def test_nt(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_nt'
    module_type_store = module_type_store.open_function_context('test_nt', 482, 0, False)
    
    # Passed parameters checking function
    test_nt.stypy_localization = localization
    test_nt.stypy_type_of_self = None
    test_nt.stypy_type_store = module_type_store
    test_nt.stypy_function_name = 'test_nt'
    test_nt.stypy_param_names_list = []
    test_nt.stypy_varargs_param_name = None
    test_nt.stypy_kwargs_param_name = 'kws'
    test_nt.stypy_call_defaults = defaults
    test_nt.stypy_call_varargs = varargs
    test_nt.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_nt', [], None, 'kws', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_nt', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_nt(...)' code ##################

    
    # Assigning a Call to a Name (line 483):
    
    # Assigning a Call to a Name (line 483):
    
    # Call to get_pythonexe(...): (line 483)
    # Processing the call keyword arguments (line 483)
    kwargs_33919 = {}
    # Getting the type of 'get_pythonexe' (line 483)
    get_pythonexe_33918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 16), 'get_pythonexe', False)
    # Calling get_pythonexe(args, kwargs) (line 483)
    get_pythonexe_call_result_33920 = invoke(stypy.reporting.localization.Localization(__file__, 483, 16), get_pythonexe_33918, *[], **kwargs_33919)
    
    # Assigning a type to the variable 'pythonexe' (line 483)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 4), 'pythonexe', get_pythonexe_call_result_33920)
    
    # Assigning a Call to a Name (line 484):
    
    # Assigning a Call to a Name (line 484):
    
    # Call to find_executable(...): (line 484)
    # Processing the call arguments (line 484)
    str_33922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 27), 'str', 'echo')
    # Processing the call keyword arguments (line 484)
    kwargs_33923 = {}
    # Getting the type of 'find_executable' (line 484)
    find_executable_33921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 11), 'find_executable', False)
    # Calling find_executable(args, kwargs) (line 484)
    find_executable_call_result_33924 = invoke(stypy.reporting.localization.Localization(__file__, 484, 11), find_executable_33921, *[str_33922], **kwargs_33923)
    
    # Assigning a type to the variable 'echo' (line 484)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 4), 'echo', find_executable_call_result_33924)
    
    # Assigning a Compare to a Name (line 485):
    
    # Assigning a Compare to a Name (line 485):
    
    # Getting the type of 'echo' (line 485)
    echo_33925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 24), 'echo')
    str_33926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 32), 'str', 'echo')
    # Applying the binary operator '!=' (line 485)
    result_ne_33927 = python_operator(stypy.reporting.localization.Localization(__file__, 485, 24), '!=', echo_33925, str_33926)
    
    # Assigning a type to the variable 'using_cygwin_echo' (line 485)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 4), 'using_cygwin_echo', result_ne_33927)
    
    # Getting the type of 'using_cygwin_echo' (line 486)
    using_cygwin_echo_33928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 7), 'using_cygwin_echo')
    # Testing the type of an if condition (line 486)
    if_condition_33929 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 486, 4), using_cygwin_echo_33928)
    # Assigning a type to the variable 'if_condition_33929' (line 486)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'if_condition_33929', if_condition_33929)
    # SSA begins for if statement (line 486)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 487)
    # Processing the call arguments (line 487)
    str_33932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 17), 'str', 'Using cygwin echo in win32 environment is not supported')
    # Processing the call keyword arguments (line 487)
    kwargs_33933 = {}
    # Getting the type of 'log' (line 487)
    log_33930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'log', False)
    # Obtaining the member 'warn' of a type (line 487)
    warn_33931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 8), log_33930, 'warn')
    # Calling warn(args, kwargs) (line 487)
    warn_call_result_33934 = invoke(stypy.reporting.localization.Localization(__file__, 487, 8), warn_33931, *[str_33932], **kwargs_33933)
    
    
    # Assigning a Call to a Tuple (line 489):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 489)
    # Processing the call arguments (line 489)
    # Getting the type of 'pythonexe' (line 489)
    pythonexe_33936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 26), 'pythonexe', False)
    str_33937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 26), 'str', ' -c "import os;print os.environ.get(\'AAA\',\'\')"')
    # Applying the binary operator '+' (line 489)
    result_add_33938 = python_operator(stypy.reporting.localization.Localization(__file__, 489, 26), '+', pythonexe_33936, str_33937)
    
    # Processing the call keyword arguments (line 489)
    kwargs_33939 = {}
    # Getting the type of 'exec_command' (line 489)
    exec_command_33935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 13), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 489)
    exec_command_call_result_33940 = invoke(stypy.reporting.localization.Localization(__file__, 489, 13), exec_command_33935, *[result_add_33938], **kwargs_33939)
    
    # Assigning a type to the variable 'call_assignment_32514' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'call_assignment_32514', exec_command_call_result_33940)
    
    # Assigning a Call to a Name (line 489):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_33943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 8), 'int')
    # Processing the call keyword arguments
    kwargs_33944 = {}
    # Getting the type of 'call_assignment_32514' (line 489)
    call_assignment_32514_33941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'call_assignment_32514', False)
    # Obtaining the member '__getitem__' of a type (line 489)
    getitem___33942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 8), call_assignment_32514_33941, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_33945 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___33942, *[int_33943], **kwargs_33944)
    
    # Assigning a type to the variable 'call_assignment_32515' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'call_assignment_32515', getitem___call_result_33945)
    
    # Assigning a Name to a Name (line 489):
    # Getting the type of 'call_assignment_32515' (line 489)
    call_assignment_32515_33946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'call_assignment_32515')
    # Assigning a type to the variable 's' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 's', call_assignment_32515_33946)
    
    # Assigning a Call to a Name (line 489):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_33949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 8), 'int')
    # Processing the call keyword arguments
    kwargs_33950 = {}
    # Getting the type of 'call_assignment_32514' (line 489)
    call_assignment_32514_33947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'call_assignment_32514', False)
    # Obtaining the member '__getitem__' of a type (line 489)
    getitem___33948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 8), call_assignment_32514_33947, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_33951 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___33948, *[int_33949], **kwargs_33950)
    
    # Assigning a type to the variable 'call_assignment_32516' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'call_assignment_32516', getitem___call_result_33951)
    
    # Assigning a Name to a Name (line 489):
    # Getting the type of 'call_assignment_32516' (line 489)
    call_assignment_32516_33952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'call_assignment_32516')
    # Assigning a type to the variable 'o' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 11), 'o', call_assignment_32516_33952)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 491)
    s_33953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 15), 's')
    int_33954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 18), 'int')
    # Applying the binary operator '==' (line 491)
    result_eq_33955 = python_operator(stypy.reporting.localization.Localization(__file__, 491, 15), '==', s_33953, int_33954)
    
    
    # Getting the type of 'o' (line 491)
    o_33956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 24), 'o')
    str_33957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 27), 'str', '')
    # Applying the binary operator '==' (line 491)
    result_eq_33958 = python_operator(stypy.reporting.localization.Localization(__file__, 491, 24), '==', o_33956, str_33957)
    
    # Applying the binary operator 'and' (line 491)
    result_and_keyword_33959 = python_operator(stypy.reporting.localization.Localization(__file__, 491, 15), 'and', result_eq_33955, result_eq_33958)
    
    
    # Assigning a Call to a Tuple (line 493):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 493)
    # Processing the call arguments (line 493)
    # Getting the type of 'pythonexe' (line 493)
    pythonexe_33961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 26), 'pythonexe', False)
    str_33962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 26), 'str', ' -c "import os;print os.environ.get(\'AAA\')"')
    # Applying the binary operator '+' (line 493)
    result_add_33963 = python_operator(stypy.reporting.localization.Localization(__file__, 493, 26), '+', pythonexe_33961, str_33962)
    
    # Processing the call keyword arguments (line 493)
    str_33964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 29), 'str', 'Tere')
    keyword_33965 = str_33964
    kwargs_33966 = {'AAA': keyword_33965}
    # Getting the type of 'exec_command' (line 493)
    exec_command_33960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 13), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 493)
    exec_command_call_result_33967 = invoke(stypy.reporting.localization.Localization(__file__, 493, 13), exec_command_33960, *[result_add_33963], **kwargs_33966)
    
    # Assigning a type to the variable 'call_assignment_32517' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'call_assignment_32517', exec_command_call_result_33967)
    
    # Assigning a Call to a Name (line 493):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_33970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 8), 'int')
    # Processing the call keyword arguments
    kwargs_33971 = {}
    # Getting the type of 'call_assignment_32517' (line 493)
    call_assignment_32517_33968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'call_assignment_32517', False)
    # Obtaining the member '__getitem__' of a type (line 493)
    getitem___33969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 8), call_assignment_32517_33968, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_33972 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___33969, *[int_33970], **kwargs_33971)
    
    # Assigning a type to the variable 'call_assignment_32518' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'call_assignment_32518', getitem___call_result_33972)
    
    # Assigning a Name to a Name (line 493):
    # Getting the type of 'call_assignment_32518' (line 493)
    call_assignment_32518_33973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'call_assignment_32518')
    # Assigning a type to the variable 's' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 's', call_assignment_32518_33973)
    
    # Assigning a Call to a Name (line 493):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_33976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 8), 'int')
    # Processing the call keyword arguments
    kwargs_33977 = {}
    # Getting the type of 'call_assignment_32517' (line 493)
    call_assignment_32517_33974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'call_assignment_32517', False)
    # Obtaining the member '__getitem__' of a type (line 493)
    getitem___33975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 8), call_assignment_32517_33974, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_33978 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___33975, *[int_33976], **kwargs_33977)
    
    # Assigning a type to the variable 'call_assignment_32519' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'call_assignment_32519', getitem___call_result_33978)
    
    # Assigning a Name to a Name (line 493):
    # Getting the type of 'call_assignment_32519' (line 493)
    call_assignment_32519_33979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'call_assignment_32519')
    # Assigning a type to the variable 'o' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 11), 'o', call_assignment_32519_33979)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 496)
    s_33980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 15), 's')
    int_33981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 18), 'int')
    # Applying the binary operator '==' (line 496)
    result_eq_33982 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 15), '==', s_33980, int_33981)
    
    
    # Getting the type of 'o' (line 496)
    o_33983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 24), 'o')
    str_33984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 27), 'str', 'Tere')
    # Applying the binary operator '==' (line 496)
    result_eq_33985 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 24), '==', o_33983, str_33984)
    
    # Applying the binary operator 'and' (line 496)
    result_and_keyword_33986 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 15), 'and', result_eq_33982, result_eq_33985)
    
    
    # Assigning a Str to a Subscript (line 498):
    
    # Assigning a Str to a Subscript (line 498):
    str_33987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 28), 'str', 'Hi')
    # Getting the type of 'os' (line 498)
    os_33988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'os')
    # Obtaining the member 'environ' of a type (line 498)
    environ_33989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 8), os_33988, 'environ')
    str_33990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 19), 'str', 'BBB')
    # Storing an element on a container (line 498)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 498, 8), environ_33989, (str_33990, str_33987))
    
    # Assigning a Call to a Tuple (line 499):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 499)
    # Processing the call arguments (line 499)
    # Getting the type of 'pythonexe' (line 499)
    pythonexe_33992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 26), 'pythonexe', False)
    str_33993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 26), 'str', ' -c "import os;print os.environ.get(\'BBB\',\'\')"')
    # Applying the binary operator '+' (line 499)
    result_add_33994 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 26), '+', pythonexe_33992, str_33993)
    
    # Processing the call keyword arguments (line 499)
    kwargs_33995 = {}
    # Getting the type of 'exec_command' (line 499)
    exec_command_33991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 13), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 499)
    exec_command_call_result_33996 = invoke(stypy.reporting.localization.Localization(__file__, 499, 13), exec_command_33991, *[result_add_33994], **kwargs_33995)
    
    # Assigning a type to the variable 'call_assignment_32520' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'call_assignment_32520', exec_command_call_result_33996)
    
    # Assigning a Call to a Name (line 499):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_33999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 8), 'int')
    # Processing the call keyword arguments
    kwargs_34000 = {}
    # Getting the type of 'call_assignment_32520' (line 499)
    call_assignment_32520_33997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'call_assignment_32520', False)
    # Obtaining the member '__getitem__' of a type (line 499)
    getitem___33998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 8), call_assignment_32520_33997, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34001 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___33998, *[int_33999], **kwargs_34000)
    
    # Assigning a type to the variable 'call_assignment_32521' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'call_assignment_32521', getitem___call_result_34001)
    
    # Assigning a Name to a Name (line 499):
    # Getting the type of 'call_assignment_32521' (line 499)
    call_assignment_32521_34002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'call_assignment_32521')
    # Assigning a type to the variable 's' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 's', call_assignment_32521_34002)
    
    # Assigning a Call to a Name (line 499):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 8), 'int')
    # Processing the call keyword arguments
    kwargs_34006 = {}
    # Getting the type of 'call_assignment_32520' (line 499)
    call_assignment_32520_34003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'call_assignment_32520', False)
    # Obtaining the member '__getitem__' of a type (line 499)
    getitem___34004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 8), call_assignment_32520_34003, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34007 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34004, *[int_34005], **kwargs_34006)
    
    # Assigning a type to the variable 'call_assignment_32522' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'call_assignment_32522', getitem___call_result_34007)
    
    # Assigning a Name to a Name (line 499):
    # Getting the type of 'call_assignment_32522' (line 499)
    call_assignment_32522_34008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'call_assignment_32522')
    # Assigning a type to the variable 'o' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 11), 'o', call_assignment_32522_34008)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 501)
    s_34009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 15), 's')
    int_34010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 18), 'int')
    # Applying the binary operator '==' (line 501)
    result_eq_34011 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 15), '==', s_34009, int_34010)
    
    
    # Getting the type of 'o' (line 501)
    o_34012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 24), 'o')
    str_34013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 27), 'str', 'Hi')
    # Applying the binary operator '==' (line 501)
    result_eq_34014 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 24), '==', o_34012, str_34013)
    
    # Applying the binary operator 'and' (line 501)
    result_and_keyword_34015 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 15), 'and', result_eq_34011, result_eq_34014)
    
    
    # Assigning a Call to a Tuple (line 503):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 503)
    # Processing the call arguments (line 503)
    # Getting the type of 'pythonexe' (line 503)
    pythonexe_34017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 26), 'pythonexe', False)
    str_34018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 26), 'str', ' -c "import os;print os.environ.get(\'BBB\',\'\')"')
    # Applying the binary operator '+' (line 503)
    result_add_34019 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 26), '+', pythonexe_34017, str_34018)
    
    # Processing the call keyword arguments (line 503)
    str_34020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 29), 'str', 'Hey')
    keyword_34021 = str_34020
    kwargs_34022 = {'BBB': keyword_34021}
    # Getting the type of 'exec_command' (line 503)
    exec_command_34016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 13), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 503)
    exec_command_call_result_34023 = invoke(stypy.reporting.localization.Localization(__file__, 503, 13), exec_command_34016, *[result_add_34019], **kwargs_34022)
    
    # Assigning a type to the variable 'call_assignment_32523' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'call_assignment_32523', exec_command_call_result_34023)
    
    # Assigning a Call to a Name (line 503):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 8), 'int')
    # Processing the call keyword arguments
    kwargs_34027 = {}
    # Getting the type of 'call_assignment_32523' (line 503)
    call_assignment_32523_34024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'call_assignment_32523', False)
    # Obtaining the member '__getitem__' of a type (line 503)
    getitem___34025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 8), call_assignment_32523_34024, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34028 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34025, *[int_34026], **kwargs_34027)
    
    # Assigning a type to the variable 'call_assignment_32524' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'call_assignment_32524', getitem___call_result_34028)
    
    # Assigning a Name to a Name (line 503):
    # Getting the type of 'call_assignment_32524' (line 503)
    call_assignment_32524_34029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'call_assignment_32524')
    # Assigning a type to the variable 's' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 's', call_assignment_32524_34029)
    
    # Assigning a Call to a Name (line 503):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 8), 'int')
    # Processing the call keyword arguments
    kwargs_34033 = {}
    # Getting the type of 'call_assignment_32523' (line 503)
    call_assignment_32523_34030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'call_assignment_32523', False)
    # Obtaining the member '__getitem__' of a type (line 503)
    getitem___34031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 8), call_assignment_32523_34030, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34034 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34031, *[int_34032], **kwargs_34033)
    
    # Assigning a type to the variable 'call_assignment_32525' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'call_assignment_32525', getitem___call_result_34034)
    
    # Assigning a Name to a Name (line 503):
    # Getting the type of 'call_assignment_32525' (line 503)
    call_assignment_32525_34035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'call_assignment_32525')
    # Assigning a type to the variable 'o' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 11), 'o', call_assignment_32525_34035)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 506)
    s_34036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 15), 's')
    int_34037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 18), 'int')
    # Applying the binary operator '==' (line 506)
    result_eq_34038 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 15), '==', s_34036, int_34037)
    
    
    # Getting the type of 'o' (line 506)
    o_34039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 24), 'o')
    str_34040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 27), 'str', 'Hey')
    # Applying the binary operator '==' (line 506)
    result_eq_34041 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 24), '==', o_34039, str_34040)
    
    # Applying the binary operator 'and' (line 506)
    result_and_keyword_34042 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 15), 'and', result_eq_34038, result_eq_34041)
    
    
    # Assigning a Call to a Tuple (line 508):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 508)
    # Processing the call arguments (line 508)
    # Getting the type of 'pythonexe' (line 508)
    pythonexe_34044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 26), 'pythonexe', False)
    str_34045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 26), 'str', ' -c "import os;print os.environ.get(\'BBB\',\'\')"')
    # Applying the binary operator '+' (line 508)
    result_add_34046 = python_operator(stypy.reporting.localization.Localization(__file__, 508, 26), '+', pythonexe_34044, str_34045)
    
    # Processing the call keyword arguments (line 508)
    kwargs_34047 = {}
    # Getting the type of 'exec_command' (line 508)
    exec_command_34043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 13), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 508)
    exec_command_call_result_34048 = invoke(stypy.reporting.localization.Localization(__file__, 508, 13), exec_command_34043, *[result_add_34046], **kwargs_34047)
    
    # Assigning a type to the variable 'call_assignment_32526' (line 508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'call_assignment_32526', exec_command_call_result_34048)
    
    # Assigning a Call to a Name (line 508):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 8), 'int')
    # Processing the call keyword arguments
    kwargs_34052 = {}
    # Getting the type of 'call_assignment_32526' (line 508)
    call_assignment_32526_34049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'call_assignment_32526', False)
    # Obtaining the member '__getitem__' of a type (line 508)
    getitem___34050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 8), call_assignment_32526_34049, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34053 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34050, *[int_34051], **kwargs_34052)
    
    # Assigning a type to the variable 'call_assignment_32527' (line 508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'call_assignment_32527', getitem___call_result_34053)
    
    # Assigning a Name to a Name (line 508):
    # Getting the type of 'call_assignment_32527' (line 508)
    call_assignment_32527_34054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'call_assignment_32527')
    # Assigning a type to the variable 's' (line 508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 's', call_assignment_32527_34054)
    
    # Assigning a Call to a Name (line 508):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 8), 'int')
    # Processing the call keyword arguments
    kwargs_34058 = {}
    # Getting the type of 'call_assignment_32526' (line 508)
    call_assignment_32526_34055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'call_assignment_32526', False)
    # Obtaining the member '__getitem__' of a type (line 508)
    getitem___34056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 8), call_assignment_32526_34055, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34059 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34056, *[int_34057], **kwargs_34058)
    
    # Assigning a type to the variable 'call_assignment_32528' (line 508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'call_assignment_32528', getitem___call_result_34059)
    
    # Assigning a Name to a Name (line 508):
    # Getting the type of 'call_assignment_32528' (line 508)
    call_assignment_32528_34060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'call_assignment_32528')
    # Assigning a type to the variable 'o' (line 508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 11), 'o', call_assignment_32528_34060)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 510)
    s_34061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 15), 's')
    int_34062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 18), 'int')
    # Applying the binary operator '==' (line 510)
    result_eq_34063 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 15), '==', s_34061, int_34062)
    
    
    # Getting the type of 'o' (line 510)
    o_34064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 24), 'o')
    str_34065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 27), 'str', 'Hi')
    # Applying the binary operator '==' (line 510)
    result_eq_34066 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 24), '==', o_34064, str_34065)
    
    # Applying the binary operator 'and' (line 510)
    result_and_keyword_34067 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 15), 'and', result_eq_34063, result_eq_34066)
    
    # SSA branch for the else part of an if statement (line 486)
    module_type_store.open_ssa_branch('else')
    
    int_34068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 9), 'int')
    # Testing the type of an if condition (line 511)
    if_condition_34069 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 511, 9), int_34068)
    # Assigning a type to the variable 'if_condition_34069' (line 511)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 9), 'if_condition_34069', if_condition_34069)
    # SSA begins for if statement (line 511)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 512):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 512)
    # Processing the call arguments (line 512)
    str_34071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 26), 'str', 'echo Hello')
    # Processing the call keyword arguments (line 512)
    kwargs_34072 = {}
    # Getting the type of 'exec_command' (line 512)
    exec_command_34070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 13), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 512)
    exec_command_call_result_34073 = invoke(stypy.reporting.localization.Localization(__file__, 512, 13), exec_command_34070, *[str_34071], **kwargs_34072)
    
    # Assigning a type to the variable 'call_assignment_32529' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'call_assignment_32529', exec_command_call_result_34073)
    
    # Assigning a Call to a Name (line 512):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 8), 'int')
    # Processing the call keyword arguments
    kwargs_34077 = {}
    # Getting the type of 'call_assignment_32529' (line 512)
    call_assignment_32529_34074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'call_assignment_32529', False)
    # Obtaining the member '__getitem__' of a type (line 512)
    getitem___34075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 8), call_assignment_32529_34074, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34078 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34075, *[int_34076], **kwargs_34077)
    
    # Assigning a type to the variable 'call_assignment_32530' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'call_assignment_32530', getitem___call_result_34078)
    
    # Assigning a Name to a Name (line 512):
    # Getting the type of 'call_assignment_32530' (line 512)
    call_assignment_32530_34079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'call_assignment_32530')
    # Assigning a type to the variable 's' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 's', call_assignment_32530_34079)
    
    # Assigning a Call to a Name (line 512):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 8), 'int')
    # Processing the call keyword arguments
    kwargs_34083 = {}
    # Getting the type of 'call_assignment_32529' (line 512)
    call_assignment_32529_34080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'call_assignment_32529', False)
    # Obtaining the member '__getitem__' of a type (line 512)
    getitem___34081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 8), call_assignment_32529_34080, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34084 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34081, *[int_34082], **kwargs_34083)
    
    # Assigning a type to the variable 'call_assignment_32531' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'call_assignment_32531', getitem___call_result_34084)
    
    # Assigning a Name to a Name (line 512):
    # Getting the type of 'call_assignment_32531' (line 512)
    call_assignment_32531_34085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'call_assignment_32531')
    # Assigning a type to the variable 'o' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 11), 'o', call_assignment_32531_34085)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 513)
    s_34086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 15), 's')
    int_34087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 18), 'int')
    # Applying the binary operator '==' (line 513)
    result_eq_34088 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 15), '==', s_34086, int_34087)
    
    
    # Getting the type of 'o' (line 513)
    o_34089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 24), 'o')
    str_34090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 27), 'str', 'Hello')
    # Applying the binary operator '==' (line 513)
    result_eq_34091 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 24), '==', o_34089, str_34090)
    
    # Applying the binary operator 'and' (line 513)
    result_and_keyword_34092 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 15), 'and', result_eq_34088, result_eq_34091)
    
    
    # Assigning a Call to a Tuple (line 515):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 515)
    # Processing the call arguments (line 515)
    str_34094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 26), 'str', 'echo a%AAA%')
    # Processing the call keyword arguments (line 515)
    kwargs_34095 = {}
    # Getting the type of 'exec_command' (line 515)
    exec_command_34093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 13), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 515)
    exec_command_call_result_34096 = invoke(stypy.reporting.localization.Localization(__file__, 515, 13), exec_command_34093, *[str_34094], **kwargs_34095)
    
    # Assigning a type to the variable 'call_assignment_32532' (line 515)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 8), 'call_assignment_32532', exec_command_call_result_34096)
    
    # Assigning a Call to a Name (line 515):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 8), 'int')
    # Processing the call keyword arguments
    kwargs_34100 = {}
    # Getting the type of 'call_assignment_32532' (line 515)
    call_assignment_32532_34097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 8), 'call_assignment_32532', False)
    # Obtaining the member '__getitem__' of a type (line 515)
    getitem___34098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 8), call_assignment_32532_34097, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34101 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34098, *[int_34099], **kwargs_34100)
    
    # Assigning a type to the variable 'call_assignment_32533' (line 515)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 8), 'call_assignment_32533', getitem___call_result_34101)
    
    # Assigning a Name to a Name (line 515):
    # Getting the type of 'call_assignment_32533' (line 515)
    call_assignment_32533_34102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 8), 'call_assignment_32533')
    # Assigning a type to the variable 's' (line 515)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 8), 's', call_assignment_32533_34102)
    
    # Assigning a Call to a Name (line 515):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 8), 'int')
    # Processing the call keyword arguments
    kwargs_34106 = {}
    # Getting the type of 'call_assignment_32532' (line 515)
    call_assignment_32532_34103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 8), 'call_assignment_32532', False)
    # Obtaining the member '__getitem__' of a type (line 515)
    getitem___34104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 8), call_assignment_32532_34103, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34107 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34104, *[int_34105], **kwargs_34106)
    
    # Assigning a type to the variable 'call_assignment_32534' (line 515)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 8), 'call_assignment_32534', getitem___call_result_34107)
    
    # Assigning a Name to a Name (line 515):
    # Getting the type of 'call_assignment_32534' (line 515)
    call_assignment_32534_34108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 8), 'call_assignment_32534')
    # Assigning a type to the variable 'o' (line 515)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 11), 'o', call_assignment_32534_34108)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 516)
    s_34109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 15), 's')
    int_34110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 18), 'int')
    # Applying the binary operator '==' (line 516)
    result_eq_34111 = python_operator(stypy.reporting.localization.Localization(__file__, 516, 15), '==', s_34109, int_34110)
    
    
    # Getting the type of 'o' (line 516)
    o_34112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 24), 'o')
    str_34113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 27), 'str', 'a')
    # Applying the binary operator '==' (line 516)
    result_eq_34114 = python_operator(stypy.reporting.localization.Localization(__file__, 516, 24), '==', o_34112, str_34113)
    
    # Applying the binary operator 'and' (line 516)
    result_and_keyword_34115 = python_operator(stypy.reporting.localization.Localization(__file__, 516, 15), 'and', result_eq_34111, result_eq_34114)
    
    
    # Assigning a Call to a Tuple (line 518):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 518)
    # Processing the call arguments (line 518)
    str_34117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 26), 'str', 'echo a%AAA%')
    # Processing the call keyword arguments (line 518)
    str_34118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 45), 'str', 'Tere')
    keyword_34119 = str_34118
    kwargs_34120 = {'AAA': keyword_34119}
    # Getting the type of 'exec_command' (line 518)
    exec_command_34116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 13), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 518)
    exec_command_call_result_34121 = invoke(stypy.reporting.localization.Localization(__file__, 518, 13), exec_command_34116, *[str_34117], **kwargs_34120)
    
    # Assigning a type to the variable 'call_assignment_32535' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'call_assignment_32535', exec_command_call_result_34121)
    
    # Assigning a Call to a Name (line 518):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 8), 'int')
    # Processing the call keyword arguments
    kwargs_34125 = {}
    # Getting the type of 'call_assignment_32535' (line 518)
    call_assignment_32535_34122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'call_assignment_32535', False)
    # Obtaining the member '__getitem__' of a type (line 518)
    getitem___34123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 8), call_assignment_32535_34122, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34126 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34123, *[int_34124], **kwargs_34125)
    
    # Assigning a type to the variable 'call_assignment_32536' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'call_assignment_32536', getitem___call_result_34126)
    
    # Assigning a Name to a Name (line 518):
    # Getting the type of 'call_assignment_32536' (line 518)
    call_assignment_32536_34127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'call_assignment_32536')
    # Assigning a type to the variable 's' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 's', call_assignment_32536_34127)
    
    # Assigning a Call to a Name (line 518):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 8), 'int')
    # Processing the call keyword arguments
    kwargs_34131 = {}
    # Getting the type of 'call_assignment_32535' (line 518)
    call_assignment_32535_34128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'call_assignment_32535', False)
    # Obtaining the member '__getitem__' of a type (line 518)
    getitem___34129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 8), call_assignment_32535_34128, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34132 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34129, *[int_34130], **kwargs_34131)
    
    # Assigning a type to the variable 'call_assignment_32537' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'call_assignment_32537', getitem___call_result_34132)
    
    # Assigning a Name to a Name (line 518):
    # Getting the type of 'call_assignment_32537' (line 518)
    call_assignment_32537_34133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'call_assignment_32537')
    # Assigning a type to the variable 'o' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 11), 'o', call_assignment_32537_34133)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 519)
    s_34134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 15), 's')
    int_34135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 18), 'int')
    # Applying the binary operator '==' (line 519)
    result_eq_34136 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 15), '==', s_34134, int_34135)
    
    
    # Getting the type of 'o' (line 519)
    o_34137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 24), 'o')
    str_34138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 27), 'str', 'aTere')
    # Applying the binary operator '==' (line 519)
    result_eq_34139 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 24), '==', o_34137, str_34138)
    
    # Applying the binary operator 'and' (line 519)
    result_and_keyword_34140 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 15), 'and', result_eq_34136, result_eq_34139)
    
    
    # Assigning a Str to a Subscript (line 521):
    
    # Assigning a Str to a Subscript (line 521):
    str_34141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 28), 'str', 'Hi')
    # Getting the type of 'os' (line 521)
    os_34142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'os')
    # Obtaining the member 'environ' of a type (line 521)
    environ_34143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 8), os_34142, 'environ')
    str_34144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 19), 'str', 'BBB')
    # Storing an element on a container (line 521)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 8), environ_34143, (str_34144, str_34141))
    
    # Assigning a Call to a Tuple (line 522):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 522)
    # Processing the call arguments (line 522)
    str_34146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 26), 'str', 'echo a%BBB%')
    # Processing the call keyword arguments (line 522)
    kwargs_34147 = {}
    # Getting the type of 'exec_command' (line 522)
    exec_command_34145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 13), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 522)
    exec_command_call_result_34148 = invoke(stypy.reporting.localization.Localization(__file__, 522, 13), exec_command_34145, *[str_34146], **kwargs_34147)
    
    # Assigning a type to the variable 'call_assignment_32538' (line 522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'call_assignment_32538', exec_command_call_result_34148)
    
    # Assigning a Call to a Name (line 522):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 8), 'int')
    # Processing the call keyword arguments
    kwargs_34152 = {}
    # Getting the type of 'call_assignment_32538' (line 522)
    call_assignment_32538_34149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'call_assignment_32538', False)
    # Obtaining the member '__getitem__' of a type (line 522)
    getitem___34150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 8), call_assignment_32538_34149, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34153 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34150, *[int_34151], **kwargs_34152)
    
    # Assigning a type to the variable 'call_assignment_32539' (line 522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'call_assignment_32539', getitem___call_result_34153)
    
    # Assigning a Name to a Name (line 522):
    # Getting the type of 'call_assignment_32539' (line 522)
    call_assignment_32539_34154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'call_assignment_32539')
    # Assigning a type to the variable 's' (line 522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 's', call_assignment_32539_34154)
    
    # Assigning a Call to a Name (line 522):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 8), 'int')
    # Processing the call keyword arguments
    kwargs_34158 = {}
    # Getting the type of 'call_assignment_32538' (line 522)
    call_assignment_32538_34155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'call_assignment_32538', False)
    # Obtaining the member '__getitem__' of a type (line 522)
    getitem___34156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 8), call_assignment_32538_34155, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34159 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34156, *[int_34157], **kwargs_34158)
    
    # Assigning a type to the variable 'call_assignment_32540' (line 522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'call_assignment_32540', getitem___call_result_34159)
    
    # Assigning a Name to a Name (line 522):
    # Getting the type of 'call_assignment_32540' (line 522)
    call_assignment_32540_34160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'call_assignment_32540')
    # Assigning a type to the variable 'o' (line 522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 11), 'o', call_assignment_32540_34160)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 523)
    s_34161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 15), 's')
    int_34162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 18), 'int')
    # Applying the binary operator '==' (line 523)
    result_eq_34163 = python_operator(stypy.reporting.localization.Localization(__file__, 523, 15), '==', s_34161, int_34162)
    
    
    # Getting the type of 'o' (line 523)
    o_34164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 24), 'o')
    str_34165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 27), 'str', 'aHi')
    # Applying the binary operator '==' (line 523)
    result_eq_34166 = python_operator(stypy.reporting.localization.Localization(__file__, 523, 24), '==', o_34164, str_34165)
    
    # Applying the binary operator 'and' (line 523)
    result_and_keyword_34167 = python_operator(stypy.reporting.localization.Localization(__file__, 523, 15), 'and', result_eq_34163, result_eq_34166)
    
    
    # Assigning a Call to a Tuple (line 525):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 525)
    # Processing the call arguments (line 525)
    str_34169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 26), 'str', 'echo a%BBB%')
    # Processing the call keyword arguments (line 525)
    str_34170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 45), 'str', 'Hey')
    keyword_34171 = str_34170
    kwargs_34172 = {'BBB': keyword_34171}
    # Getting the type of 'exec_command' (line 525)
    exec_command_34168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 13), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 525)
    exec_command_call_result_34173 = invoke(stypy.reporting.localization.Localization(__file__, 525, 13), exec_command_34168, *[str_34169], **kwargs_34172)
    
    # Assigning a type to the variable 'call_assignment_32541' (line 525)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'call_assignment_32541', exec_command_call_result_34173)
    
    # Assigning a Call to a Name (line 525):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 8), 'int')
    # Processing the call keyword arguments
    kwargs_34177 = {}
    # Getting the type of 'call_assignment_32541' (line 525)
    call_assignment_32541_34174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'call_assignment_32541', False)
    # Obtaining the member '__getitem__' of a type (line 525)
    getitem___34175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 8), call_assignment_32541_34174, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34178 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34175, *[int_34176], **kwargs_34177)
    
    # Assigning a type to the variable 'call_assignment_32542' (line 525)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'call_assignment_32542', getitem___call_result_34178)
    
    # Assigning a Name to a Name (line 525):
    # Getting the type of 'call_assignment_32542' (line 525)
    call_assignment_32542_34179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'call_assignment_32542')
    # Assigning a type to the variable 's' (line 525)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 's', call_assignment_32542_34179)
    
    # Assigning a Call to a Name (line 525):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 8), 'int')
    # Processing the call keyword arguments
    kwargs_34183 = {}
    # Getting the type of 'call_assignment_32541' (line 525)
    call_assignment_32541_34180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'call_assignment_32541', False)
    # Obtaining the member '__getitem__' of a type (line 525)
    getitem___34181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 8), call_assignment_32541_34180, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34184 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34181, *[int_34182], **kwargs_34183)
    
    # Assigning a type to the variable 'call_assignment_32543' (line 525)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'call_assignment_32543', getitem___call_result_34184)
    
    # Assigning a Name to a Name (line 525):
    # Getting the type of 'call_assignment_32543' (line 525)
    call_assignment_32543_34185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'call_assignment_32543')
    # Assigning a type to the variable 'o' (line 525)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 11), 'o', call_assignment_32543_34185)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 526)
    s_34186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 15), 's')
    int_34187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 18), 'int')
    # Applying the binary operator '==' (line 526)
    result_eq_34188 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 15), '==', s_34186, int_34187)
    
    
    # Getting the type of 'o' (line 526)
    o_34189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 24), 'o')
    str_34190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 27), 'str', 'aHey')
    # Applying the binary operator '==' (line 526)
    result_eq_34191 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 24), '==', o_34189, str_34190)
    
    # Applying the binary operator 'and' (line 526)
    result_and_keyword_34192 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 15), 'and', result_eq_34188, result_eq_34191)
    
    
    # Assigning a Call to a Tuple (line 527):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 527)
    # Processing the call arguments (line 527)
    str_34194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 26), 'str', 'echo a%BBB%')
    # Processing the call keyword arguments (line 527)
    kwargs_34195 = {}
    # Getting the type of 'exec_command' (line 527)
    exec_command_34193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 13), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 527)
    exec_command_call_result_34196 = invoke(stypy.reporting.localization.Localization(__file__, 527, 13), exec_command_34193, *[str_34194], **kwargs_34195)
    
    # Assigning a type to the variable 'call_assignment_32544' (line 527)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'call_assignment_32544', exec_command_call_result_34196)
    
    # Assigning a Call to a Name (line 527):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 8), 'int')
    # Processing the call keyword arguments
    kwargs_34200 = {}
    # Getting the type of 'call_assignment_32544' (line 527)
    call_assignment_32544_34197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'call_assignment_32544', False)
    # Obtaining the member '__getitem__' of a type (line 527)
    getitem___34198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 8), call_assignment_32544_34197, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34201 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34198, *[int_34199], **kwargs_34200)
    
    # Assigning a type to the variable 'call_assignment_32545' (line 527)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'call_assignment_32545', getitem___call_result_34201)
    
    # Assigning a Name to a Name (line 527):
    # Getting the type of 'call_assignment_32545' (line 527)
    call_assignment_32545_34202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'call_assignment_32545')
    # Assigning a type to the variable 's' (line 527)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 's', call_assignment_32545_34202)
    
    # Assigning a Call to a Name (line 527):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 8), 'int')
    # Processing the call keyword arguments
    kwargs_34206 = {}
    # Getting the type of 'call_assignment_32544' (line 527)
    call_assignment_32544_34203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'call_assignment_32544', False)
    # Obtaining the member '__getitem__' of a type (line 527)
    getitem___34204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 8), call_assignment_32544_34203, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34207 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34204, *[int_34205], **kwargs_34206)
    
    # Assigning a type to the variable 'call_assignment_32546' (line 527)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'call_assignment_32546', getitem___call_result_34207)
    
    # Assigning a Name to a Name (line 527):
    # Getting the type of 'call_assignment_32546' (line 527)
    call_assignment_32546_34208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'call_assignment_32546')
    # Assigning a type to the variable 'o' (line 527)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 11), 'o', call_assignment_32546_34208)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 528)
    s_34209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 15), 's')
    int_34210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 18), 'int')
    # Applying the binary operator '==' (line 528)
    result_eq_34211 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 15), '==', s_34209, int_34210)
    
    
    # Getting the type of 'o' (line 528)
    o_34212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 24), 'o')
    str_34213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 27), 'str', 'aHi')
    # Applying the binary operator '==' (line 528)
    result_eq_34214 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 24), '==', o_34212, str_34213)
    
    # Applying the binary operator 'and' (line 528)
    result_and_keyword_34215 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 15), 'and', result_eq_34211, result_eq_34214)
    
    
    # Assigning a Call to a Tuple (line 530):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 530)
    # Processing the call arguments (line 530)
    str_34217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 26), 'str', 'this_is_not_a_command')
    # Processing the call keyword arguments (line 530)
    kwargs_34218 = {}
    # Getting the type of 'exec_command' (line 530)
    exec_command_34216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 13), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 530)
    exec_command_call_result_34219 = invoke(stypy.reporting.localization.Localization(__file__, 530, 13), exec_command_34216, *[str_34217], **kwargs_34218)
    
    # Assigning a type to the variable 'call_assignment_32547' (line 530)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'call_assignment_32547', exec_command_call_result_34219)
    
    # Assigning a Call to a Name (line 530):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 8), 'int')
    # Processing the call keyword arguments
    kwargs_34223 = {}
    # Getting the type of 'call_assignment_32547' (line 530)
    call_assignment_32547_34220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'call_assignment_32547', False)
    # Obtaining the member '__getitem__' of a type (line 530)
    getitem___34221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 8), call_assignment_32547_34220, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34224 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34221, *[int_34222], **kwargs_34223)
    
    # Assigning a type to the variable 'call_assignment_32548' (line 530)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'call_assignment_32548', getitem___call_result_34224)
    
    # Assigning a Name to a Name (line 530):
    # Getting the type of 'call_assignment_32548' (line 530)
    call_assignment_32548_34225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'call_assignment_32548')
    # Assigning a type to the variable 's' (line 530)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 's', call_assignment_32548_34225)
    
    # Assigning a Call to a Name (line 530):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 8), 'int')
    # Processing the call keyword arguments
    kwargs_34229 = {}
    # Getting the type of 'call_assignment_32547' (line 530)
    call_assignment_32547_34226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'call_assignment_32547', False)
    # Obtaining the member '__getitem__' of a type (line 530)
    getitem___34227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 8), call_assignment_32547_34226, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34230 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34227, *[int_34228], **kwargs_34229)
    
    # Assigning a type to the variable 'call_assignment_32549' (line 530)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'call_assignment_32549', getitem___call_result_34230)
    
    # Assigning a Name to a Name (line 530):
    # Getting the type of 'call_assignment_32549' (line 530)
    call_assignment_32549_34231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'call_assignment_32549')
    # Assigning a type to the variable 'o' (line 530)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 11), 'o', call_assignment_32549_34231)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    # Getting the type of 's' (line 531)
    s_34232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 15), 's')
    
    # Getting the type of 'o' (line 531)
    o_34233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 21), 'o')
    str_34234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 24), 'str', '')
    # Applying the binary operator '!=' (line 531)
    result_ne_34235 = python_operator(stypy.reporting.localization.Localization(__file__, 531, 21), '!=', o_34233, str_34234)
    
    # Applying the binary operator 'and' (line 531)
    result_and_keyword_34236 = python_operator(stypy.reporting.localization.Localization(__file__, 531, 15), 'and', s_34232, result_ne_34235)
    
    
    # Assigning a Call to a Tuple (line 533):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 533)
    # Processing the call arguments (line 533)
    str_34238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 26), 'str', 'type not_existing_file')
    # Processing the call keyword arguments (line 533)
    kwargs_34239 = {}
    # Getting the type of 'exec_command' (line 533)
    exec_command_34237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 13), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 533)
    exec_command_call_result_34240 = invoke(stypy.reporting.localization.Localization(__file__, 533, 13), exec_command_34237, *[str_34238], **kwargs_34239)
    
    # Assigning a type to the variable 'call_assignment_32550' (line 533)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'call_assignment_32550', exec_command_call_result_34240)
    
    # Assigning a Call to a Name (line 533):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 8), 'int')
    # Processing the call keyword arguments
    kwargs_34244 = {}
    # Getting the type of 'call_assignment_32550' (line 533)
    call_assignment_32550_34241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'call_assignment_32550', False)
    # Obtaining the member '__getitem__' of a type (line 533)
    getitem___34242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 8), call_assignment_32550_34241, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34245 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34242, *[int_34243], **kwargs_34244)
    
    # Assigning a type to the variable 'call_assignment_32551' (line 533)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'call_assignment_32551', getitem___call_result_34245)
    
    # Assigning a Name to a Name (line 533):
    # Getting the type of 'call_assignment_32551' (line 533)
    call_assignment_32551_34246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'call_assignment_32551')
    # Assigning a type to the variable 's' (line 533)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 's', call_assignment_32551_34246)
    
    # Assigning a Call to a Name (line 533):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 8), 'int')
    # Processing the call keyword arguments
    kwargs_34250 = {}
    # Getting the type of 'call_assignment_32550' (line 533)
    call_assignment_32550_34247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'call_assignment_32550', False)
    # Obtaining the member '__getitem__' of a type (line 533)
    getitem___34248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 8), call_assignment_32550_34247, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34251 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34248, *[int_34249], **kwargs_34250)
    
    # Assigning a type to the variable 'call_assignment_32552' (line 533)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'call_assignment_32552', getitem___call_result_34251)
    
    # Assigning a Name to a Name (line 533):
    # Getting the type of 'call_assignment_32552' (line 533)
    call_assignment_32552_34252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'call_assignment_32552')
    # Assigning a type to the variable 'o' (line 533)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 11), 'o', call_assignment_32552_34252)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    # Getting the type of 's' (line 534)
    s_34253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 15), 's')
    
    # Getting the type of 'o' (line 534)
    o_34254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 21), 'o')
    str_34255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 24), 'str', '')
    # Applying the binary operator '!=' (line 534)
    result_ne_34256 = python_operator(stypy.reporting.localization.Localization(__file__, 534, 21), '!=', o_34254, str_34255)
    
    # Applying the binary operator 'and' (line 534)
    result_and_keyword_34257 = python_operator(stypy.reporting.localization.Localization(__file__, 534, 15), 'and', s_34253, result_ne_34256)
    
    # SSA join for if statement (line 511)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 486)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 536):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 536)
    # Processing the call arguments (line 536)
    str_34259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 22), 'str', 'echo path=%path%')
    # Processing the call keyword arguments (line 536)
    kwargs_34260 = {}
    # Getting the type of 'exec_command' (line 536)
    exec_command_34258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 9), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 536)
    exec_command_call_result_34261 = invoke(stypy.reporting.localization.Localization(__file__, 536, 9), exec_command_34258, *[str_34259], **kwargs_34260)
    
    # Assigning a type to the variable 'call_assignment_32553' (line 536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'call_assignment_32553', exec_command_call_result_34261)
    
    # Assigning a Call to a Name (line 536):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34265 = {}
    # Getting the type of 'call_assignment_32553' (line 536)
    call_assignment_32553_34262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'call_assignment_32553', False)
    # Obtaining the member '__getitem__' of a type (line 536)
    getitem___34263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 4), call_assignment_32553_34262, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34266 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34263, *[int_34264], **kwargs_34265)
    
    # Assigning a type to the variable 'call_assignment_32554' (line 536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'call_assignment_32554', getitem___call_result_34266)
    
    # Assigning a Name to a Name (line 536):
    # Getting the type of 'call_assignment_32554' (line 536)
    call_assignment_32554_34267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'call_assignment_32554')
    # Assigning a type to the variable 's' (line 536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 's', call_assignment_32554_34267)
    
    # Assigning a Call to a Name (line 536):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34271 = {}
    # Getting the type of 'call_assignment_32553' (line 536)
    call_assignment_32553_34268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'call_assignment_32553', False)
    # Obtaining the member '__getitem__' of a type (line 536)
    getitem___34269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 4), call_assignment_32553_34268, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34272 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34269, *[int_34270], **kwargs_34271)
    
    # Assigning a type to the variable 'call_assignment_32555' (line 536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'call_assignment_32555', getitem___call_result_34272)
    
    # Assigning a Name to a Name (line 536):
    # Getting the type of 'call_assignment_32555' (line 536)
    call_assignment_32555_34273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'call_assignment_32555')
    # Assigning a type to the variable 'o' (line 536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 7), 'o', call_assignment_32555_34273)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 537)
    s_34274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 11), 's')
    int_34275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 14), 'int')
    # Applying the binary operator '==' (line 537)
    result_eq_34276 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 11), '==', s_34274, int_34275)
    
    
    # Getting the type of 'o' (line 537)
    o_34277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 20), 'o')
    str_34278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 23), 'str', '')
    # Applying the binary operator '!=' (line 537)
    result_ne_34279 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 20), '!=', o_34277, str_34278)
    
    # Applying the binary operator 'and' (line 537)
    result_and_keyword_34280 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 11), 'and', result_eq_34276, result_ne_34279)
    
    
    # Assigning a Call to a Tuple (line 539):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 539)
    # Processing the call arguments (line 539)
    str_34282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 22), 'str', '%s -c "import sys;sys.stderr.write(sys.platform)"')
    # Getting the type of 'pythonexe' (line 540)
    pythonexe_34283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 23), 'pythonexe', False)
    # Applying the binary operator '%' (line 539)
    result_mod_34284 = python_operator(stypy.reporting.localization.Localization(__file__, 539, 22), '%', str_34282, pythonexe_34283)
    
    # Processing the call keyword arguments (line 539)
    kwargs_34285 = {}
    # Getting the type of 'exec_command' (line 539)
    exec_command_34281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 9), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 539)
    exec_command_call_result_34286 = invoke(stypy.reporting.localization.Localization(__file__, 539, 9), exec_command_34281, *[result_mod_34284], **kwargs_34285)
    
    # Assigning a type to the variable 'call_assignment_32556' (line 539)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 4), 'call_assignment_32556', exec_command_call_result_34286)
    
    # Assigning a Call to a Name (line 539):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34290 = {}
    # Getting the type of 'call_assignment_32556' (line 539)
    call_assignment_32556_34287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 4), 'call_assignment_32556', False)
    # Obtaining the member '__getitem__' of a type (line 539)
    getitem___34288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 4), call_assignment_32556_34287, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34291 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34288, *[int_34289], **kwargs_34290)
    
    # Assigning a type to the variable 'call_assignment_32557' (line 539)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 4), 'call_assignment_32557', getitem___call_result_34291)
    
    # Assigning a Name to a Name (line 539):
    # Getting the type of 'call_assignment_32557' (line 539)
    call_assignment_32557_34292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 4), 'call_assignment_32557')
    # Assigning a type to the variable 's' (line 539)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 4), 's', call_assignment_32557_34292)
    
    # Assigning a Call to a Name (line 539):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34296 = {}
    # Getting the type of 'call_assignment_32556' (line 539)
    call_assignment_32556_34293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 4), 'call_assignment_32556', False)
    # Obtaining the member '__getitem__' of a type (line 539)
    getitem___34294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 4), call_assignment_32556_34293, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34297 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34294, *[int_34295], **kwargs_34296)
    
    # Assigning a type to the variable 'call_assignment_32558' (line 539)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 4), 'call_assignment_32558', getitem___call_result_34297)
    
    # Assigning a Name to a Name (line 539):
    # Getting the type of 'call_assignment_32558' (line 539)
    call_assignment_32558_34298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 4), 'call_assignment_32558')
    # Assigning a type to the variable 'o' (line 539)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 7), 'o', call_assignment_32558_34298)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 541)
    s_34299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 11), 's')
    int_34300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 14), 'int')
    # Applying the binary operator '==' (line 541)
    result_eq_34301 = python_operator(stypy.reporting.localization.Localization(__file__, 541, 11), '==', s_34299, int_34300)
    
    
    # Getting the type of 'o' (line 541)
    o_34302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 20), 'o')
    str_34303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 23), 'str', 'win32')
    # Applying the binary operator '==' (line 541)
    result_eq_34304 = python_operator(stypy.reporting.localization.Localization(__file__, 541, 20), '==', o_34302, str_34303)
    
    # Applying the binary operator 'and' (line 541)
    result_and_keyword_34305 = python_operator(stypy.reporting.localization.Localization(__file__, 541, 11), 'and', result_eq_34301, result_eq_34304)
    
    
    # Assigning a Call to a Tuple (line 543):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 543)
    # Processing the call arguments (line 543)
    str_34307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 22), 'str', '%s -c "raise \'Ignore me.\'"')
    # Getting the type of 'pythonexe' (line 543)
    pythonexe_34308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 55), 'pythonexe', False)
    # Applying the binary operator '%' (line 543)
    result_mod_34309 = python_operator(stypy.reporting.localization.Localization(__file__, 543, 22), '%', str_34307, pythonexe_34308)
    
    # Processing the call keyword arguments (line 543)
    kwargs_34310 = {}
    # Getting the type of 'exec_command' (line 543)
    exec_command_34306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 9), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 543)
    exec_command_call_result_34311 = invoke(stypy.reporting.localization.Localization(__file__, 543, 9), exec_command_34306, *[result_mod_34309], **kwargs_34310)
    
    # Assigning a type to the variable 'call_assignment_32559' (line 543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 4), 'call_assignment_32559', exec_command_call_result_34311)
    
    # Assigning a Call to a Name (line 543):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34315 = {}
    # Getting the type of 'call_assignment_32559' (line 543)
    call_assignment_32559_34312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 4), 'call_assignment_32559', False)
    # Obtaining the member '__getitem__' of a type (line 543)
    getitem___34313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 4), call_assignment_32559_34312, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34316 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34313, *[int_34314], **kwargs_34315)
    
    # Assigning a type to the variable 'call_assignment_32560' (line 543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 4), 'call_assignment_32560', getitem___call_result_34316)
    
    # Assigning a Name to a Name (line 543):
    # Getting the type of 'call_assignment_32560' (line 543)
    call_assignment_32560_34317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 4), 'call_assignment_32560')
    # Assigning a type to the variable 's' (line 543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 4), 's', call_assignment_32560_34317)
    
    # Assigning a Call to a Name (line 543):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34321 = {}
    # Getting the type of 'call_assignment_32559' (line 543)
    call_assignment_32559_34318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 4), 'call_assignment_32559', False)
    # Obtaining the member '__getitem__' of a type (line 543)
    getitem___34319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 4), call_assignment_32559_34318, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34322 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34319, *[int_34320], **kwargs_34321)
    
    # Assigning a type to the variable 'call_assignment_32561' (line 543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 4), 'call_assignment_32561', getitem___call_result_34322)
    
    # Assigning a Name to a Name (line 543):
    # Getting the type of 'call_assignment_32561' (line 543)
    call_assignment_32561_34323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 4), 'call_assignment_32561')
    # Assigning a type to the variable 'o' (line 543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 7), 'o', call_assignment_32561_34323)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 544)
    s_34324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 11), 's')
    int_34325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 14), 'int')
    # Applying the binary operator '==' (line 544)
    result_eq_34326 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 11), '==', s_34324, int_34325)
    
    # Getting the type of 'o' (line 544)
    o_34327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 20), 'o')
    # Applying the binary operator 'and' (line 544)
    result_and_keyword_34328 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 11), 'and', result_eq_34326, o_34327)
    
    
    # Assigning a Call to a Tuple (line 546):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 546)
    # Processing the call arguments (line 546)
    str_34330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 22), 'str', '%s -c "import sys;sys.stderr.write(\'0\');sys.stderr.write(\'1\');sys.stderr.write(\'2\')"')
    # Getting the type of 'pythonexe' (line 547)
    pythonexe_34331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 23), 'pythonexe', False)
    # Applying the binary operator '%' (line 546)
    result_mod_34332 = python_operator(stypy.reporting.localization.Localization(__file__, 546, 22), '%', str_34330, pythonexe_34331)
    
    # Processing the call keyword arguments (line 546)
    kwargs_34333 = {}
    # Getting the type of 'exec_command' (line 546)
    exec_command_34329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 9), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 546)
    exec_command_call_result_34334 = invoke(stypy.reporting.localization.Localization(__file__, 546, 9), exec_command_34329, *[result_mod_34332], **kwargs_34333)
    
    # Assigning a type to the variable 'call_assignment_32562' (line 546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 4), 'call_assignment_32562', exec_command_call_result_34334)
    
    # Assigning a Call to a Name (line 546):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34338 = {}
    # Getting the type of 'call_assignment_32562' (line 546)
    call_assignment_32562_34335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 4), 'call_assignment_32562', False)
    # Obtaining the member '__getitem__' of a type (line 546)
    getitem___34336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 4), call_assignment_32562_34335, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34339 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34336, *[int_34337], **kwargs_34338)
    
    # Assigning a type to the variable 'call_assignment_32563' (line 546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 4), 'call_assignment_32563', getitem___call_result_34339)
    
    # Assigning a Name to a Name (line 546):
    # Getting the type of 'call_assignment_32563' (line 546)
    call_assignment_32563_34340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 4), 'call_assignment_32563')
    # Assigning a type to the variable 's' (line 546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 4), 's', call_assignment_32563_34340)
    
    # Assigning a Call to a Name (line 546):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34344 = {}
    # Getting the type of 'call_assignment_32562' (line 546)
    call_assignment_32562_34341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 4), 'call_assignment_32562', False)
    # Obtaining the member '__getitem__' of a type (line 546)
    getitem___34342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 4), call_assignment_32562_34341, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34345 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34342, *[int_34343], **kwargs_34344)
    
    # Assigning a type to the variable 'call_assignment_32564' (line 546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 4), 'call_assignment_32564', getitem___call_result_34345)
    
    # Assigning a Name to a Name (line 546):
    # Getting the type of 'call_assignment_32564' (line 546)
    call_assignment_32564_34346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 4), 'call_assignment_32564')
    # Assigning a type to the variable 'o' (line 546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 7), 'o', call_assignment_32564_34346)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 548)
    s_34347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 11), 's')
    int_34348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 14), 'int')
    # Applying the binary operator '==' (line 548)
    result_eq_34349 = python_operator(stypy.reporting.localization.Localization(__file__, 548, 11), '==', s_34347, int_34348)
    
    
    # Getting the type of 'o' (line 548)
    o_34350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 20), 'o')
    str_34351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 23), 'str', '012')
    # Applying the binary operator '==' (line 548)
    result_eq_34352 = python_operator(stypy.reporting.localization.Localization(__file__, 548, 20), '==', o_34350, str_34351)
    
    # Applying the binary operator 'and' (line 548)
    result_and_keyword_34353 = python_operator(stypy.reporting.localization.Localization(__file__, 548, 11), 'and', result_eq_34349, result_eq_34352)
    
    
    # Assigning a Call to a Tuple (line 550):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 550)
    # Processing the call arguments (line 550)
    str_34355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 22), 'str', '%s -c "import sys;sys.exit(15)"')
    # Getting the type of 'pythonexe' (line 550)
    pythonexe_34356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 58), 'pythonexe', False)
    # Applying the binary operator '%' (line 550)
    result_mod_34357 = python_operator(stypy.reporting.localization.Localization(__file__, 550, 22), '%', str_34355, pythonexe_34356)
    
    # Processing the call keyword arguments (line 550)
    kwargs_34358 = {}
    # Getting the type of 'exec_command' (line 550)
    exec_command_34354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 9), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 550)
    exec_command_call_result_34359 = invoke(stypy.reporting.localization.Localization(__file__, 550, 9), exec_command_34354, *[result_mod_34357], **kwargs_34358)
    
    # Assigning a type to the variable 'call_assignment_32565' (line 550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), 'call_assignment_32565', exec_command_call_result_34359)
    
    # Assigning a Call to a Name (line 550):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34363 = {}
    # Getting the type of 'call_assignment_32565' (line 550)
    call_assignment_32565_34360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), 'call_assignment_32565', False)
    # Obtaining the member '__getitem__' of a type (line 550)
    getitem___34361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 4), call_assignment_32565_34360, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34364 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34361, *[int_34362], **kwargs_34363)
    
    # Assigning a type to the variable 'call_assignment_32566' (line 550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), 'call_assignment_32566', getitem___call_result_34364)
    
    # Assigning a Name to a Name (line 550):
    # Getting the type of 'call_assignment_32566' (line 550)
    call_assignment_32566_34365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), 'call_assignment_32566')
    # Assigning a type to the variable 's' (line 550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), 's', call_assignment_32566_34365)
    
    # Assigning a Call to a Name (line 550):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34369 = {}
    # Getting the type of 'call_assignment_32565' (line 550)
    call_assignment_32565_34366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), 'call_assignment_32565', False)
    # Obtaining the member '__getitem__' of a type (line 550)
    getitem___34367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 4), call_assignment_32565_34366, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34370 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34367, *[int_34368], **kwargs_34369)
    
    # Assigning a type to the variable 'call_assignment_32567' (line 550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), 'call_assignment_32567', getitem___call_result_34370)
    
    # Assigning a Name to a Name (line 550):
    # Getting the type of 'call_assignment_32567' (line 550)
    call_assignment_32567_34371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), 'call_assignment_32567')
    # Assigning a type to the variable 'o' (line 550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 7), 'o', call_assignment_32567_34371)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 551)
    s_34372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 11), 's')
    int_34373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 14), 'int')
    # Applying the binary operator '==' (line 551)
    result_eq_34374 = python_operator(stypy.reporting.localization.Localization(__file__, 551, 11), '==', s_34372, int_34373)
    
    
    # Getting the type of 'o' (line 551)
    o_34375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 21), 'o')
    str_34376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 24), 'str', '')
    # Applying the binary operator '==' (line 551)
    result_eq_34377 = python_operator(stypy.reporting.localization.Localization(__file__, 551, 21), '==', o_34375, str_34376)
    
    # Applying the binary operator 'and' (line 551)
    result_and_keyword_34378 = python_operator(stypy.reporting.localization.Localization(__file__, 551, 11), 'and', result_eq_34374, result_eq_34377)
    
    
    # Assigning a Call to a Tuple (line 553):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 553)
    # Processing the call arguments (line 553)
    str_34380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 22), 'str', '%s -c "print \'Heipa\'"')
    # Getting the type of 'pythonexe' (line 553)
    pythonexe_34381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 50), 'pythonexe', False)
    # Applying the binary operator '%' (line 553)
    result_mod_34382 = python_operator(stypy.reporting.localization.Localization(__file__, 553, 22), '%', str_34380, pythonexe_34381)
    
    # Processing the call keyword arguments (line 553)
    kwargs_34383 = {}
    # Getting the type of 'exec_command' (line 553)
    exec_command_34379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 9), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 553)
    exec_command_call_result_34384 = invoke(stypy.reporting.localization.Localization(__file__, 553, 9), exec_command_34379, *[result_mod_34382], **kwargs_34383)
    
    # Assigning a type to the variable 'call_assignment_32568' (line 553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'call_assignment_32568', exec_command_call_result_34384)
    
    # Assigning a Call to a Name (line 553):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34388 = {}
    # Getting the type of 'call_assignment_32568' (line 553)
    call_assignment_32568_34385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'call_assignment_32568', False)
    # Obtaining the member '__getitem__' of a type (line 553)
    getitem___34386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 4), call_assignment_32568_34385, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34389 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34386, *[int_34387], **kwargs_34388)
    
    # Assigning a type to the variable 'call_assignment_32569' (line 553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'call_assignment_32569', getitem___call_result_34389)
    
    # Assigning a Name to a Name (line 553):
    # Getting the type of 'call_assignment_32569' (line 553)
    call_assignment_32569_34390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'call_assignment_32569')
    # Assigning a type to the variable 's' (line 553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 's', call_assignment_32569_34390)
    
    # Assigning a Call to a Name (line 553):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34394 = {}
    # Getting the type of 'call_assignment_32568' (line 553)
    call_assignment_32568_34391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'call_assignment_32568', False)
    # Obtaining the member '__getitem__' of a type (line 553)
    getitem___34392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 4), call_assignment_32568_34391, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34395 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34392, *[int_34393], **kwargs_34394)
    
    # Assigning a type to the variable 'call_assignment_32570' (line 553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'call_assignment_32570', getitem___call_result_34395)
    
    # Assigning a Name to a Name (line 553):
    # Getting the type of 'call_assignment_32570' (line 553)
    call_assignment_32570_34396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'call_assignment_32570')
    # Assigning a type to the variable 'o' (line 553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 7), 'o', call_assignment_32570_34396)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 554)
    s_34397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 11), 's')
    int_34398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 14), 'int')
    # Applying the binary operator '==' (line 554)
    result_eq_34399 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 11), '==', s_34397, int_34398)
    
    
    # Getting the type of 'o' (line 554)
    o_34400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 20), 'o')
    str_34401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 23), 'str', 'Heipa')
    # Applying the binary operator '==' (line 554)
    result_eq_34402 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 20), '==', o_34400, str_34401)
    
    # Applying the binary operator 'and' (line 554)
    result_and_keyword_34403 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 11), 'and', result_eq_34399, result_eq_34402)
    
    
    # Call to print(...): (line 556)
    # Processing the call arguments (line 556)
    str_34405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 11), 'str', 'ok')
    # Processing the call keyword arguments (line 556)
    kwargs_34406 = {}
    # Getting the type of 'print' (line 556)
    print_34404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 4), 'print', False)
    # Calling print(args, kwargs) (line 556)
    print_call_result_34407 = invoke(stypy.reporting.localization.Localization(__file__, 556, 4), print_34404, *[str_34405], **kwargs_34406)
    
    
    # ################# End of 'test_nt(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_nt' in the type store
    # Getting the type of 'stypy_return_type' (line 482)
    stypy_return_type_34408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_34408)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_nt'
    return stypy_return_type_34408

# Assigning a type to the variable 'test_nt' (line 482)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 0), 'test_nt', test_nt)

@norecursion
def test_posix(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_posix'
    module_type_store = module_type_store.open_function_context('test_posix', 558, 0, False)
    
    # Passed parameters checking function
    test_posix.stypy_localization = localization
    test_posix.stypy_type_of_self = None
    test_posix.stypy_type_store = module_type_store
    test_posix.stypy_function_name = 'test_posix'
    test_posix.stypy_param_names_list = []
    test_posix.stypy_varargs_param_name = None
    test_posix.stypy_kwargs_param_name = 'kws'
    test_posix.stypy_call_defaults = defaults
    test_posix.stypy_call_varargs = varargs
    test_posix.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_posix', [], None, 'kws', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_posix', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_posix(...)' code ##################

    
    # Assigning a Call to a Tuple (line 559):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 559)
    # Processing the call arguments (line 559)
    str_34410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 22), 'str', 'echo Hello')
    # Processing the call keyword arguments (line 559)
    # Getting the type of 'kws' (line 559)
    kws_34411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 37), 'kws', False)
    kwargs_34412 = {'kws_34411': kws_34411}
    # Getting the type of 'exec_command' (line 559)
    exec_command_34409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 9), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 559)
    exec_command_call_result_34413 = invoke(stypy.reporting.localization.Localization(__file__, 559, 9), exec_command_34409, *[str_34410], **kwargs_34412)
    
    # Assigning a type to the variable 'call_assignment_32571' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'call_assignment_32571', exec_command_call_result_34413)
    
    # Assigning a Call to a Name (line 559):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34417 = {}
    # Getting the type of 'call_assignment_32571' (line 559)
    call_assignment_32571_34414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'call_assignment_32571', False)
    # Obtaining the member '__getitem__' of a type (line 559)
    getitem___34415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 4), call_assignment_32571_34414, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34418 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34415, *[int_34416], **kwargs_34417)
    
    # Assigning a type to the variable 'call_assignment_32572' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'call_assignment_32572', getitem___call_result_34418)
    
    # Assigning a Name to a Name (line 559):
    # Getting the type of 'call_assignment_32572' (line 559)
    call_assignment_32572_34419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'call_assignment_32572')
    # Assigning a type to the variable 's' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 's', call_assignment_32572_34419)
    
    # Assigning a Call to a Name (line 559):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34423 = {}
    # Getting the type of 'call_assignment_32571' (line 559)
    call_assignment_32571_34420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'call_assignment_32571', False)
    # Obtaining the member '__getitem__' of a type (line 559)
    getitem___34421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 4), call_assignment_32571_34420, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34424 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34421, *[int_34422], **kwargs_34423)
    
    # Assigning a type to the variable 'call_assignment_32573' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'call_assignment_32573', getitem___call_result_34424)
    
    # Assigning a Name to a Name (line 559):
    # Getting the type of 'call_assignment_32573' (line 559)
    call_assignment_32573_34425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'call_assignment_32573')
    # Assigning a type to the variable 'o' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 7), 'o', call_assignment_32573_34425)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 560)
    s_34426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 11), 's')
    int_34427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 14), 'int')
    # Applying the binary operator '==' (line 560)
    result_eq_34428 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 11), '==', s_34426, int_34427)
    
    
    # Getting the type of 'o' (line 560)
    o_34429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 20), 'o')
    str_34430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 23), 'str', 'Hello')
    # Applying the binary operator '==' (line 560)
    result_eq_34431 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 20), '==', o_34429, str_34430)
    
    # Applying the binary operator 'and' (line 560)
    result_and_keyword_34432 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 11), 'and', result_eq_34428, result_eq_34431)
    
    
    # Assigning a Call to a Tuple (line 562):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 562)
    # Processing the call arguments (line 562)
    str_34434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 22), 'str', 'echo $AAA')
    # Processing the call keyword arguments (line 562)
    # Getting the type of 'kws' (line 562)
    kws_34435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 36), 'kws', False)
    kwargs_34436 = {'kws_34435': kws_34435}
    # Getting the type of 'exec_command' (line 562)
    exec_command_34433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 9), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 562)
    exec_command_call_result_34437 = invoke(stypy.reporting.localization.Localization(__file__, 562, 9), exec_command_34433, *[str_34434], **kwargs_34436)
    
    # Assigning a type to the variable 'call_assignment_32574' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'call_assignment_32574', exec_command_call_result_34437)
    
    # Assigning a Call to a Name (line 562):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34441 = {}
    # Getting the type of 'call_assignment_32574' (line 562)
    call_assignment_32574_34438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'call_assignment_32574', False)
    # Obtaining the member '__getitem__' of a type (line 562)
    getitem___34439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 4), call_assignment_32574_34438, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34442 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34439, *[int_34440], **kwargs_34441)
    
    # Assigning a type to the variable 'call_assignment_32575' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'call_assignment_32575', getitem___call_result_34442)
    
    # Assigning a Name to a Name (line 562):
    # Getting the type of 'call_assignment_32575' (line 562)
    call_assignment_32575_34443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'call_assignment_32575')
    # Assigning a type to the variable 's' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 's', call_assignment_32575_34443)
    
    # Assigning a Call to a Name (line 562):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34447 = {}
    # Getting the type of 'call_assignment_32574' (line 562)
    call_assignment_32574_34444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'call_assignment_32574', False)
    # Obtaining the member '__getitem__' of a type (line 562)
    getitem___34445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 4), call_assignment_32574_34444, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34448 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34445, *[int_34446], **kwargs_34447)
    
    # Assigning a type to the variable 'call_assignment_32576' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'call_assignment_32576', getitem___call_result_34448)
    
    # Assigning a Name to a Name (line 562):
    # Getting the type of 'call_assignment_32576' (line 562)
    call_assignment_32576_34449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'call_assignment_32576')
    # Assigning a type to the variable 'o' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 7), 'o', call_assignment_32576_34449)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 563)
    s_34450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 11), 's')
    int_34451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 14), 'int')
    # Applying the binary operator '==' (line 563)
    result_eq_34452 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 11), '==', s_34450, int_34451)
    
    
    # Getting the type of 'o' (line 563)
    o_34453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 20), 'o')
    str_34454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 23), 'str', '')
    # Applying the binary operator '==' (line 563)
    result_eq_34455 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 20), '==', o_34453, str_34454)
    
    # Applying the binary operator 'and' (line 563)
    result_and_keyword_34456 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 11), 'and', result_eq_34452, result_eq_34455)
    
    
    # Assigning a Call to a Tuple (line 565):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 565)
    # Processing the call arguments (line 565)
    str_34458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 22), 'str', 'echo "$AAA"')
    # Processing the call keyword arguments (line 565)
    str_34459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 40), 'str', 'Tere')
    keyword_34460 = str_34459
    # Getting the type of 'kws' (line 565)
    kws_34461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 49), 'kws', False)
    kwargs_34462 = {'kws_34461': kws_34461, 'AAA': keyword_34460}
    # Getting the type of 'exec_command' (line 565)
    exec_command_34457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 9), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 565)
    exec_command_call_result_34463 = invoke(stypy.reporting.localization.Localization(__file__, 565, 9), exec_command_34457, *[str_34458], **kwargs_34462)
    
    # Assigning a type to the variable 'call_assignment_32577' (line 565)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'call_assignment_32577', exec_command_call_result_34463)
    
    # Assigning a Call to a Name (line 565):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34467 = {}
    # Getting the type of 'call_assignment_32577' (line 565)
    call_assignment_32577_34464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'call_assignment_32577', False)
    # Obtaining the member '__getitem__' of a type (line 565)
    getitem___34465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 4), call_assignment_32577_34464, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34468 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34465, *[int_34466], **kwargs_34467)
    
    # Assigning a type to the variable 'call_assignment_32578' (line 565)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'call_assignment_32578', getitem___call_result_34468)
    
    # Assigning a Name to a Name (line 565):
    # Getting the type of 'call_assignment_32578' (line 565)
    call_assignment_32578_34469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'call_assignment_32578')
    # Assigning a type to the variable 's' (line 565)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 's', call_assignment_32578_34469)
    
    # Assigning a Call to a Name (line 565):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34473 = {}
    # Getting the type of 'call_assignment_32577' (line 565)
    call_assignment_32577_34470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'call_assignment_32577', False)
    # Obtaining the member '__getitem__' of a type (line 565)
    getitem___34471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 4), call_assignment_32577_34470, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34474 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34471, *[int_34472], **kwargs_34473)
    
    # Assigning a type to the variable 'call_assignment_32579' (line 565)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'call_assignment_32579', getitem___call_result_34474)
    
    # Assigning a Name to a Name (line 565):
    # Getting the type of 'call_assignment_32579' (line 565)
    call_assignment_32579_34475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'call_assignment_32579')
    # Assigning a type to the variable 'o' (line 565)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 7), 'o', call_assignment_32579_34475)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 566)
    s_34476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 11), 's')
    int_34477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 14), 'int')
    # Applying the binary operator '==' (line 566)
    result_eq_34478 = python_operator(stypy.reporting.localization.Localization(__file__, 566, 11), '==', s_34476, int_34477)
    
    
    # Getting the type of 'o' (line 566)
    o_34479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 20), 'o')
    str_34480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 23), 'str', 'Tere')
    # Applying the binary operator '==' (line 566)
    result_eq_34481 = python_operator(stypy.reporting.localization.Localization(__file__, 566, 20), '==', o_34479, str_34480)
    
    # Applying the binary operator 'and' (line 566)
    result_and_keyword_34482 = python_operator(stypy.reporting.localization.Localization(__file__, 566, 11), 'and', result_eq_34478, result_eq_34481)
    
    
    # Assigning a Call to a Tuple (line 569):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 569)
    # Processing the call arguments (line 569)
    str_34484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 22), 'str', 'echo "$AAA"')
    # Processing the call keyword arguments (line 569)
    # Getting the type of 'kws' (line 569)
    kws_34485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 38), 'kws', False)
    kwargs_34486 = {'kws_34485': kws_34485}
    # Getting the type of 'exec_command' (line 569)
    exec_command_34483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 9), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 569)
    exec_command_call_result_34487 = invoke(stypy.reporting.localization.Localization(__file__, 569, 9), exec_command_34483, *[str_34484], **kwargs_34486)
    
    # Assigning a type to the variable 'call_assignment_32580' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'call_assignment_32580', exec_command_call_result_34487)
    
    # Assigning a Call to a Name (line 569):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34491 = {}
    # Getting the type of 'call_assignment_32580' (line 569)
    call_assignment_32580_34488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'call_assignment_32580', False)
    # Obtaining the member '__getitem__' of a type (line 569)
    getitem___34489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 4), call_assignment_32580_34488, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34492 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34489, *[int_34490], **kwargs_34491)
    
    # Assigning a type to the variable 'call_assignment_32581' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'call_assignment_32581', getitem___call_result_34492)
    
    # Assigning a Name to a Name (line 569):
    # Getting the type of 'call_assignment_32581' (line 569)
    call_assignment_32581_34493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'call_assignment_32581')
    # Assigning a type to the variable 's' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 's', call_assignment_32581_34493)
    
    # Assigning a Call to a Name (line 569):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34497 = {}
    # Getting the type of 'call_assignment_32580' (line 569)
    call_assignment_32580_34494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'call_assignment_32580', False)
    # Obtaining the member '__getitem__' of a type (line 569)
    getitem___34495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 4), call_assignment_32580_34494, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34498 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34495, *[int_34496], **kwargs_34497)
    
    # Assigning a type to the variable 'call_assignment_32582' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'call_assignment_32582', getitem___call_result_34498)
    
    # Assigning a Name to a Name (line 569):
    # Getting the type of 'call_assignment_32582' (line 569)
    call_assignment_32582_34499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'call_assignment_32582')
    # Assigning a type to the variable 'o' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 7), 'o', call_assignment_32582_34499)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 570)
    s_34500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 11), 's')
    int_34501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 14), 'int')
    # Applying the binary operator '==' (line 570)
    result_eq_34502 = python_operator(stypy.reporting.localization.Localization(__file__, 570, 11), '==', s_34500, int_34501)
    
    
    # Getting the type of 'o' (line 570)
    o_34503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 20), 'o')
    str_34504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 23), 'str', '')
    # Applying the binary operator '==' (line 570)
    result_eq_34505 = python_operator(stypy.reporting.localization.Localization(__file__, 570, 20), '==', o_34503, str_34504)
    
    # Applying the binary operator 'and' (line 570)
    result_and_keyword_34506 = python_operator(stypy.reporting.localization.Localization(__file__, 570, 11), 'and', result_eq_34502, result_eq_34505)
    
    
    # Assigning a Str to a Subscript (line 572):
    
    # Assigning a Str to a Subscript (line 572):
    str_34507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 24), 'str', 'Hi')
    # Getting the type of 'os' (line 572)
    os_34508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 4), 'os')
    # Obtaining the member 'environ' of a type (line 572)
    environ_34509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 4), os_34508, 'environ')
    str_34510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 15), 'str', 'BBB')
    # Storing an element on a container (line 572)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 572, 4), environ_34509, (str_34510, str_34507))
    
    # Assigning a Call to a Tuple (line 573):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 573)
    # Processing the call arguments (line 573)
    str_34512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 22), 'str', 'echo "$BBB"')
    # Processing the call keyword arguments (line 573)
    # Getting the type of 'kws' (line 573)
    kws_34513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 38), 'kws', False)
    kwargs_34514 = {'kws_34513': kws_34513}
    # Getting the type of 'exec_command' (line 573)
    exec_command_34511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 9), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 573)
    exec_command_call_result_34515 = invoke(stypy.reporting.localization.Localization(__file__, 573, 9), exec_command_34511, *[str_34512], **kwargs_34514)
    
    # Assigning a type to the variable 'call_assignment_32583' (line 573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 4), 'call_assignment_32583', exec_command_call_result_34515)
    
    # Assigning a Call to a Name (line 573):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34519 = {}
    # Getting the type of 'call_assignment_32583' (line 573)
    call_assignment_32583_34516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 4), 'call_assignment_32583', False)
    # Obtaining the member '__getitem__' of a type (line 573)
    getitem___34517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 4), call_assignment_32583_34516, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34520 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34517, *[int_34518], **kwargs_34519)
    
    # Assigning a type to the variable 'call_assignment_32584' (line 573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 4), 'call_assignment_32584', getitem___call_result_34520)
    
    # Assigning a Name to a Name (line 573):
    # Getting the type of 'call_assignment_32584' (line 573)
    call_assignment_32584_34521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 4), 'call_assignment_32584')
    # Assigning a type to the variable 's' (line 573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 4), 's', call_assignment_32584_34521)
    
    # Assigning a Call to a Name (line 573):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34525 = {}
    # Getting the type of 'call_assignment_32583' (line 573)
    call_assignment_32583_34522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 4), 'call_assignment_32583', False)
    # Obtaining the member '__getitem__' of a type (line 573)
    getitem___34523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 4), call_assignment_32583_34522, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34526 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34523, *[int_34524], **kwargs_34525)
    
    # Assigning a type to the variable 'call_assignment_32585' (line 573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 4), 'call_assignment_32585', getitem___call_result_34526)
    
    # Assigning a Name to a Name (line 573):
    # Getting the type of 'call_assignment_32585' (line 573)
    call_assignment_32585_34527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 4), 'call_assignment_32585')
    # Assigning a type to the variable 'o' (line 573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 7), 'o', call_assignment_32585_34527)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 574)
    s_34528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 11), 's')
    int_34529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 14), 'int')
    # Applying the binary operator '==' (line 574)
    result_eq_34530 = python_operator(stypy.reporting.localization.Localization(__file__, 574, 11), '==', s_34528, int_34529)
    
    
    # Getting the type of 'o' (line 574)
    o_34531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 20), 'o')
    str_34532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 23), 'str', 'Hi')
    # Applying the binary operator '==' (line 574)
    result_eq_34533 = python_operator(stypy.reporting.localization.Localization(__file__, 574, 20), '==', o_34531, str_34532)
    
    # Applying the binary operator 'and' (line 574)
    result_and_keyword_34534 = python_operator(stypy.reporting.localization.Localization(__file__, 574, 11), 'and', result_eq_34530, result_eq_34533)
    
    
    # Assigning a Call to a Tuple (line 576):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 576)
    # Processing the call arguments (line 576)
    str_34536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 22), 'str', 'echo "$BBB"')
    # Processing the call keyword arguments (line 576)
    str_34537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 40), 'str', 'Hey')
    keyword_34538 = str_34537
    # Getting the type of 'kws' (line 576)
    kws_34539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 48), 'kws', False)
    kwargs_34540 = {'kws_34539': kws_34539, 'BBB': keyword_34538}
    # Getting the type of 'exec_command' (line 576)
    exec_command_34535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 9), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 576)
    exec_command_call_result_34541 = invoke(stypy.reporting.localization.Localization(__file__, 576, 9), exec_command_34535, *[str_34536], **kwargs_34540)
    
    # Assigning a type to the variable 'call_assignment_32586' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 4), 'call_assignment_32586', exec_command_call_result_34541)
    
    # Assigning a Call to a Name (line 576):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34545 = {}
    # Getting the type of 'call_assignment_32586' (line 576)
    call_assignment_32586_34542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 4), 'call_assignment_32586', False)
    # Obtaining the member '__getitem__' of a type (line 576)
    getitem___34543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 4), call_assignment_32586_34542, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34546 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34543, *[int_34544], **kwargs_34545)
    
    # Assigning a type to the variable 'call_assignment_32587' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 4), 'call_assignment_32587', getitem___call_result_34546)
    
    # Assigning a Name to a Name (line 576):
    # Getting the type of 'call_assignment_32587' (line 576)
    call_assignment_32587_34547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 4), 'call_assignment_32587')
    # Assigning a type to the variable 's' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 4), 's', call_assignment_32587_34547)
    
    # Assigning a Call to a Name (line 576):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34551 = {}
    # Getting the type of 'call_assignment_32586' (line 576)
    call_assignment_32586_34548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 4), 'call_assignment_32586', False)
    # Obtaining the member '__getitem__' of a type (line 576)
    getitem___34549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 4), call_assignment_32586_34548, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34552 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34549, *[int_34550], **kwargs_34551)
    
    # Assigning a type to the variable 'call_assignment_32588' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 4), 'call_assignment_32588', getitem___call_result_34552)
    
    # Assigning a Name to a Name (line 576):
    # Getting the type of 'call_assignment_32588' (line 576)
    call_assignment_32588_34553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 4), 'call_assignment_32588')
    # Assigning a type to the variable 'o' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 7), 'o', call_assignment_32588_34553)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 577)
    s_34554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 11), 's')
    int_34555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 14), 'int')
    # Applying the binary operator '==' (line 577)
    result_eq_34556 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 11), '==', s_34554, int_34555)
    
    
    # Getting the type of 'o' (line 577)
    o_34557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 20), 'o')
    str_34558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 23), 'str', 'Hey')
    # Applying the binary operator '==' (line 577)
    result_eq_34559 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 20), '==', o_34557, str_34558)
    
    # Applying the binary operator 'and' (line 577)
    result_and_keyword_34560 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 11), 'and', result_eq_34556, result_eq_34559)
    
    
    # Assigning a Call to a Tuple (line 579):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 579)
    # Processing the call arguments (line 579)
    str_34562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 22), 'str', 'echo "$BBB"')
    # Processing the call keyword arguments (line 579)
    # Getting the type of 'kws' (line 579)
    kws_34563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 38), 'kws', False)
    kwargs_34564 = {'kws_34563': kws_34563}
    # Getting the type of 'exec_command' (line 579)
    exec_command_34561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 9), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 579)
    exec_command_call_result_34565 = invoke(stypy.reporting.localization.Localization(__file__, 579, 9), exec_command_34561, *[str_34562], **kwargs_34564)
    
    # Assigning a type to the variable 'call_assignment_32589' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'call_assignment_32589', exec_command_call_result_34565)
    
    # Assigning a Call to a Name (line 579):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34569 = {}
    # Getting the type of 'call_assignment_32589' (line 579)
    call_assignment_32589_34566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'call_assignment_32589', False)
    # Obtaining the member '__getitem__' of a type (line 579)
    getitem___34567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 4), call_assignment_32589_34566, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34570 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34567, *[int_34568], **kwargs_34569)
    
    # Assigning a type to the variable 'call_assignment_32590' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'call_assignment_32590', getitem___call_result_34570)
    
    # Assigning a Name to a Name (line 579):
    # Getting the type of 'call_assignment_32590' (line 579)
    call_assignment_32590_34571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'call_assignment_32590')
    # Assigning a type to the variable 's' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 's', call_assignment_32590_34571)
    
    # Assigning a Call to a Name (line 579):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34575 = {}
    # Getting the type of 'call_assignment_32589' (line 579)
    call_assignment_32589_34572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'call_assignment_32589', False)
    # Obtaining the member '__getitem__' of a type (line 579)
    getitem___34573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 4), call_assignment_32589_34572, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34576 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34573, *[int_34574], **kwargs_34575)
    
    # Assigning a type to the variable 'call_assignment_32591' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'call_assignment_32591', getitem___call_result_34576)
    
    # Assigning a Name to a Name (line 579):
    # Getting the type of 'call_assignment_32591' (line 579)
    call_assignment_32591_34577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'call_assignment_32591')
    # Assigning a type to the variable 'o' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 7), 'o', call_assignment_32591_34577)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 580)
    s_34578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 11), 's')
    int_34579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 14), 'int')
    # Applying the binary operator '==' (line 580)
    result_eq_34580 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 11), '==', s_34578, int_34579)
    
    
    # Getting the type of 'o' (line 580)
    o_34581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 20), 'o')
    str_34582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 23), 'str', 'Hi')
    # Applying the binary operator '==' (line 580)
    result_eq_34583 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 20), '==', o_34581, str_34582)
    
    # Applying the binary operator 'and' (line 580)
    result_and_keyword_34584 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 11), 'and', result_eq_34580, result_eq_34583)
    
    
    # Assigning a Call to a Tuple (line 583):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 583)
    # Processing the call arguments (line 583)
    str_34586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 22), 'str', 'this_is_not_a_command')
    # Processing the call keyword arguments (line 583)
    # Getting the type of 'kws' (line 583)
    kws_34587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 48), 'kws', False)
    kwargs_34588 = {'kws_34587': kws_34587}
    # Getting the type of 'exec_command' (line 583)
    exec_command_34585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 9), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 583)
    exec_command_call_result_34589 = invoke(stypy.reporting.localization.Localization(__file__, 583, 9), exec_command_34585, *[str_34586], **kwargs_34588)
    
    # Assigning a type to the variable 'call_assignment_32592' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'call_assignment_32592', exec_command_call_result_34589)
    
    # Assigning a Call to a Name (line 583):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34593 = {}
    # Getting the type of 'call_assignment_32592' (line 583)
    call_assignment_32592_34590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'call_assignment_32592', False)
    # Obtaining the member '__getitem__' of a type (line 583)
    getitem___34591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 4), call_assignment_32592_34590, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34594 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34591, *[int_34592], **kwargs_34593)
    
    # Assigning a type to the variable 'call_assignment_32593' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'call_assignment_32593', getitem___call_result_34594)
    
    # Assigning a Name to a Name (line 583):
    # Getting the type of 'call_assignment_32593' (line 583)
    call_assignment_32593_34595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'call_assignment_32593')
    # Assigning a type to the variable 's' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 's', call_assignment_32593_34595)
    
    # Assigning a Call to a Name (line 583):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34599 = {}
    # Getting the type of 'call_assignment_32592' (line 583)
    call_assignment_32592_34596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'call_assignment_32592', False)
    # Obtaining the member '__getitem__' of a type (line 583)
    getitem___34597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 4), call_assignment_32592_34596, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34600 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34597, *[int_34598], **kwargs_34599)
    
    # Assigning a type to the variable 'call_assignment_32594' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'call_assignment_32594', getitem___call_result_34600)
    
    # Assigning a Name to a Name (line 583):
    # Getting the type of 'call_assignment_32594' (line 583)
    call_assignment_32594_34601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'call_assignment_32594')
    # Assigning a type to the variable 'o' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 7), 'o', call_assignment_32594_34601)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 584)
    s_34602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 11), 's')
    int_34603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 14), 'int')
    # Applying the binary operator '!=' (line 584)
    result_ne_34604 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 11), '!=', s_34602, int_34603)
    
    
    # Getting the type of 'o' (line 584)
    o_34605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 20), 'o')
    str_34606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 23), 'str', '')
    # Applying the binary operator '!=' (line 584)
    result_ne_34607 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 20), '!=', o_34605, str_34606)
    
    # Applying the binary operator 'and' (line 584)
    result_and_keyword_34608 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 11), 'and', result_ne_34604, result_ne_34607)
    
    
    # Assigning a Call to a Tuple (line 586):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 586)
    # Processing the call arguments (line 586)
    str_34610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 22), 'str', 'echo path=$PATH')
    # Processing the call keyword arguments (line 586)
    # Getting the type of 'kws' (line 586)
    kws_34611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 42), 'kws', False)
    kwargs_34612 = {'kws_34611': kws_34611}
    # Getting the type of 'exec_command' (line 586)
    exec_command_34609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 9), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 586)
    exec_command_call_result_34613 = invoke(stypy.reporting.localization.Localization(__file__, 586, 9), exec_command_34609, *[str_34610], **kwargs_34612)
    
    # Assigning a type to the variable 'call_assignment_32595' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'call_assignment_32595', exec_command_call_result_34613)
    
    # Assigning a Call to a Name (line 586):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34617 = {}
    # Getting the type of 'call_assignment_32595' (line 586)
    call_assignment_32595_34614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'call_assignment_32595', False)
    # Obtaining the member '__getitem__' of a type (line 586)
    getitem___34615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 4), call_assignment_32595_34614, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34618 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34615, *[int_34616], **kwargs_34617)
    
    # Assigning a type to the variable 'call_assignment_32596' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'call_assignment_32596', getitem___call_result_34618)
    
    # Assigning a Name to a Name (line 586):
    # Getting the type of 'call_assignment_32596' (line 586)
    call_assignment_32596_34619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'call_assignment_32596')
    # Assigning a type to the variable 's' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 's', call_assignment_32596_34619)
    
    # Assigning a Call to a Name (line 586):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34623 = {}
    # Getting the type of 'call_assignment_32595' (line 586)
    call_assignment_32595_34620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'call_assignment_32595', False)
    # Obtaining the member '__getitem__' of a type (line 586)
    getitem___34621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 4), call_assignment_32595_34620, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34624 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34621, *[int_34622], **kwargs_34623)
    
    # Assigning a type to the variable 'call_assignment_32597' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'call_assignment_32597', getitem___call_result_34624)
    
    # Assigning a Name to a Name (line 586):
    # Getting the type of 'call_assignment_32597' (line 586)
    call_assignment_32597_34625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'call_assignment_32597')
    # Assigning a type to the variable 'o' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 7), 'o', call_assignment_32597_34625)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 587)
    s_34626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 11), 's')
    int_34627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 14), 'int')
    # Applying the binary operator '==' (line 587)
    result_eq_34628 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 11), '==', s_34626, int_34627)
    
    
    # Getting the type of 'o' (line 587)
    o_34629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 20), 'o')
    str_34630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 23), 'str', '')
    # Applying the binary operator '!=' (line 587)
    result_ne_34631 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 20), '!=', o_34629, str_34630)
    
    # Applying the binary operator 'and' (line 587)
    result_and_keyword_34632 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 11), 'and', result_eq_34628, result_ne_34631)
    
    
    # Assigning a Call to a Tuple (line 589):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 589)
    # Processing the call arguments (line 589)
    str_34634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 22), 'str', 'python -c "import sys,os;sys.stderr.write(os.name)"')
    # Processing the call keyword arguments (line 589)
    # Getting the type of 'kws' (line 589)
    kws_34635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 78), 'kws', False)
    kwargs_34636 = {'kws_34635': kws_34635}
    # Getting the type of 'exec_command' (line 589)
    exec_command_34633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 9), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 589)
    exec_command_call_result_34637 = invoke(stypy.reporting.localization.Localization(__file__, 589, 9), exec_command_34633, *[str_34634], **kwargs_34636)
    
    # Assigning a type to the variable 'call_assignment_32598' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'call_assignment_32598', exec_command_call_result_34637)
    
    # Assigning a Call to a Name (line 589):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34641 = {}
    # Getting the type of 'call_assignment_32598' (line 589)
    call_assignment_32598_34638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'call_assignment_32598', False)
    # Obtaining the member '__getitem__' of a type (line 589)
    getitem___34639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 4), call_assignment_32598_34638, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34642 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34639, *[int_34640], **kwargs_34641)
    
    # Assigning a type to the variable 'call_assignment_32599' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'call_assignment_32599', getitem___call_result_34642)
    
    # Assigning a Name to a Name (line 589):
    # Getting the type of 'call_assignment_32599' (line 589)
    call_assignment_32599_34643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'call_assignment_32599')
    # Assigning a type to the variable 's' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 's', call_assignment_32599_34643)
    
    # Assigning a Call to a Name (line 589):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34647 = {}
    # Getting the type of 'call_assignment_32598' (line 589)
    call_assignment_32598_34644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'call_assignment_32598', False)
    # Obtaining the member '__getitem__' of a type (line 589)
    getitem___34645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 4), call_assignment_32598_34644, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34648 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34645, *[int_34646], **kwargs_34647)
    
    # Assigning a type to the variable 'call_assignment_32600' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'call_assignment_32600', getitem___call_result_34648)
    
    # Assigning a Name to a Name (line 589):
    # Getting the type of 'call_assignment_32600' (line 589)
    call_assignment_32600_34649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'call_assignment_32600')
    # Assigning a type to the variable 'o' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 7), 'o', call_assignment_32600_34649)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 590)
    s_34650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 11), 's')
    int_34651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 14), 'int')
    # Applying the binary operator '==' (line 590)
    result_eq_34652 = python_operator(stypy.reporting.localization.Localization(__file__, 590, 11), '==', s_34650, int_34651)
    
    
    # Getting the type of 'o' (line 590)
    o_34653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 20), 'o')
    str_34654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 23), 'str', 'posix')
    # Applying the binary operator '==' (line 590)
    result_eq_34655 = python_operator(stypy.reporting.localization.Localization(__file__, 590, 20), '==', o_34653, str_34654)
    
    # Applying the binary operator 'and' (line 590)
    result_and_keyword_34656 = python_operator(stypy.reporting.localization.Localization(__file__, 590, 11), 'and', result_eq_34652, result_eq_34655)
    
    
    # Assigning a Call to a Tuple (line 592):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 592)
    # Processing the call arguments (line 592)
    str_34658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 22), 'str', 'python -c "raise \'Ignore me.\'"')
    # Processing the call keyword arguments (line 592)
    # Getting the type of 'kws' (line 592)
    kws_34659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 59), 'kws', False)
    kwargs_34660 = {'kws_34659': kws_34659}
    # Getting the type of 'exec_command' (line 592)
    exec_command_34657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 9), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 592)
    exec_command_call_result_34661 = invoke(stypy.reporting.localization.Localization(__file__, 592, 9), exec_command_34657, *[str_34658], **kwargs_34660)
    
    # Assigning a type to the variable 'call_assignment_32601' (line 592)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 4), 'call_assignment_32601', exec_command_call_result_34661)
    
    # Assigning a Call to a Name (line 592):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34665 = {}
    # Getting the type of 'call_assignment_32601' (line 592)
    call_assignment_32601_34662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 4), 'call_assignment_32601', False)
    # Obtaining the member '__getitem__' of a type (line 592)
    getitem___34663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 4), call_assignment_32601_34662, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34666 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34663, *[int_34664], **kwargs_34665)
    
    # Assigning a type to the variable 'call_assignment_32602' (line 592)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 4), 'call_assignment_32602', getitem___call_result_34666)
    
    # Assigning a Name to a Name (line 592):
    # Getting the type of 'call_assignment_32602' (line 592)
    call_assignment_32602_34667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 4), 'call_assignment_32602')
    # Assigning a type to the variable 's' (line 592)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 4), 's', call_assignment_32602_34667)
    
    # Assigning a Call to a Name (line 592):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34671 = {}
    # Getting the type of 'call_assignment_32601' (line 592)
    call_assignment_32601_34668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 4), 'call_assignment_32601', False)
    # Obtaining the member '__getitem__' of a type (line 592)
    getitem___34669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 4), call_assignment_32601_34668, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34672 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34669, *[int_34670], **kwargs_34671)
    
    # Assigning a type to the variable 'call_assignment_32603' (line 592)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 4), 'call_assignment_32603', getitem___call_result_34672)
    
    # Assigning a Name to a Name (line 592):
    # Getting the type of 'call_assignment_32603' (line 592)
    call_assignment_32603_34673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 4), 'call_assignment_32603')
    # Assigning a type to the variable 'o' (line 592)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 7), 'o', call_assignment_32603_34673)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 593)
    s_34674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 11), 's')
    int_34675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 14), 'int')
    # Applying the binary operator '==' (line 593)
    result_eq_34676 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 11), '==', s_34674, int_34675)
    
    # Getting the type of 'o' (line 593)
    o_34677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 20), 'o')
    # Applying the binary operator 'and' (line 593)
    result_and_keyword_34678 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 11), 'and', result_eq_34676, o_34677)
    
    
    # Assigning a Call to a Tuple (line 595):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 595)
    # Processing the call arguments (line 595)
    str_34680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 22), 'str', 'python -c "import sys;sys.stderr.write(\'0\');sys.stderr.write(\'1\');sys.stderr.write(\'2\')"')
    # Processing the call keyword arguments (line 595)
    # Getting the type of 'kws' (line 595)
    kws_34681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 121), 'kws', False)
    kwargs_34682 = {'kws_34681': kws_34681}
    # Getting the type of 'exec_command' (line 595)
    exec_command_34679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 9), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 595)
    exec_command_call_result_34683 = invoke(stypy.reporting.localization.Localization(__file__, 595, 9), exec_command_34679, *[str_34680], **kwargs_34682)
    
    # Assigning a type to the variable 'call_assignment_32604' (line 595)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 4), 'call_assignment_32604', exec_command_call_result_34683)
    
    # Assigning a Call to a Name (line 595):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34687 = {}
    # Getting the type of 'call_assignment_32604' (line 595)
    call_assignment_32604_34684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 4), 'call_assignment_32604', False)
    # Obtaining the member '__getitem__' of a type (line 595)
    getitem___34685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 4), call_assignment_32604_34684, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34688 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34685, *[int_34686], **kwargs_34687)
    
    # Assigning a type to the variable 'call_assignment_32605' (line 595)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 4), 'call_assignment_32605', getitem___call_result_34688)
    
    # Assigning a Name to a Name (line 595):
    # Getting the type of 'call_assignment_32605' (line 595)
    call_assignment_32605_34689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 4), 'call_assignment_32605')
    # Assigning a type to the variable 's' (line 595)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 4), 's', call_assignment_32605_34689)
    
    # Assigning a Call to a Name (line 595):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34693 = {}
    # Getting the type of 'call_assignment_32604' (line 595)
    call_assignment_32604_34690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 4), 'call_assignment_32604', False)
    # Obtaining the member '__getitem__' of a type (line 595)
    getitem___34691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 4), call_assignment_32604_34690, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34694 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34691, *[int_34692], **kwargs_34693)
    
    # Assigning a type to the variable 'call_assignment_32606' (line 595)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 4), 'call_assignment_32606', getitem___call_result_34694)
    
    # Assigning a Name to a Name (line 595):
    # Getting the type of 'call_assignment_32606' (line 595)
    call_assignment_32606_34695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 4), 'call_assignment_32606')
    # Assigning a type to the variable 'o' (line 595)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 7), 'o', call_assignment_32606_34695)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 596)
    s_34696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 11), 's')
    int_34697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 14), 'int')
    # Applying the binary operator '==' (line 596)
    result_eq_34698 = python_operator(stypy.reporting.localization.Localization(__file__, 596, 11), '==', s_34696, int_34697)
    
    
    # Getting the type of 'o' (line 596)
    o_34699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 20), 'o')
    str_34700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 23), 'str', '012')
    # Applying the binary operator '==' (line 596)
    result_eq_34701 = python_operator(stypy.reporting.localization.Localization(__file__, 596, 20), '==', o_34699, str_34700)
    
    # Applying the binary operator 'and' (line 596)
    result_and_keyword_34702 = python_operator(stypy.reporting.localization.Localization(__file__, 596, 11), 'and', result_eq_34698, result_eq_34701)
    
    
    # Assigning a Call to a Tuple (line 598):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 598)
    # Processing the call arguments (line 598)
    str_34704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 22), 'str', 'python -c "import sys;sys.exit(15)"')
    # Processing the call keyword arguments (line 598)
    # Getting the type of 'kws' (line 598)
    kws_34705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 62), 'kws', False)
    kwargs_34706 = {'kws_34705': kws_34705}
    # Getting the type of 'exec_command' (line 598)
    exec_command_34703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 9), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 598)
    exec_command_call_result_34707 = invoke(stypy.reporting.localization.Localization(__file__, 598, 9), exec_command_34703, *[str_34704], **kwargs_34706)
    
    # Assigning a type to the variable 'call_assignment_32607' (line 598)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'call_assignment_32607', exec_command_call_result_34707)
    
    # Assigning a Call to a Name (line 598):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34711 = {}
    # Getting the type of 'call_assignment_32607' (line 598)
    call_assignment_32607_34708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'call_assignment_32607', False)
    # Obtaining the member '__getitem__' of a type (line 598)
    getitem___34709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 4), call_assignment_32607_34708, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34712 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34709, *[int_34710], **kwargs_34711)
    
    # Assigning a type to the variable 'call_assignment_32608' (line 598)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'call_assignment_32608', getitem___call_result_34712)
    
    # Assigning a Name to a Name (line 598):
    # Getting the type of 'call_assignment_32608' (line 598)
    call_assignment_32608_34713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'call_assignment_32608')
    # Assigning a type to the variable 's' (line 598)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 's', call_assignment_32608_34713)
    
    # Assigning a Call to a Name (line 598):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34717 = {}
    # Getting the type of 'call_assignment_32607' (line 598)
    call_assignment_32607_34714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'call_assignment_32607', False)
    # Obtaining the member '__getitem__' of a type (line 598)
    getitem___34715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 4), call_assignment_32607_34714, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34718 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34715, *[int_34716], **kwargs_34717)
    
    # Assigning a type to the variable 'call_assignment_32609' (line 598)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'call_assignment_32609', getitem___call_result_34718)
    
    # Assigning a Name to a Name (line 598):
    # Getting the type of 'call_assignment_32609' (line 598)
    call_assignment_32609_34719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'call_assignment_32609')
    # Assigning a type to the variable 'o' (line 598)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 7), 'o', call_assignment_32609_34719)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 599)
    s_34720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 11), 's')
    int_34721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 14), 'int')
    # Applying the binary operator '==' (line 599)
    result_eq_34722 = python_operator(stypy.reporting.localization.Localization(__file__, 599, 11), '==', s_34720, int_34721)
    
    
    # Getting the type of 'o' (line 599)
    o_34723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 21), 'o')
    str_34724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 24), 'str', '')
    # Applying the binary operator '==' (line 599)
    result_eq_34725 = python_operator(stypy.reporting.localization.Localization(__file__, 599, 21), '==', o_34723, str_34724)
    
    # Applying the binary operator 'and' (line 599)
    result_and_keyword_34726 = python_operator(stypy.reporting.localization.Localization(__file__, 599, 11), 'and', result_eq_34722, result_eq_34725)
    
    
    # Assigning a Call to a Tuple (line 601):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 601)
    # Processing the call arguments (line 601)
    str_34728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 22), 'str', 'python -c "print \'Heipa\'"')
    # Processing the call keyword arguments (line 601)
    # Getting the type of 'kws' (line 601)
    kws_34729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 54), 'kws', False)
    kwargs_34730 = {'kws_34729': kws_34729}
    # Getting the type of 'exec_command' (line 601)
    exec_command_34727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 9), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 601)
    exec_command_call_result_34731 = invoke(stypy.reporting.localization.Localization(__file__, 601, 9), exec_command_34727, *[str_34728], **kwargs_34730)
    
    # Assigning a type to the variable 'call_assignment_32610' (line 601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 4), 'call_assignment_32610', exec_command_call_result_34731)
    
    # Assigning a Call to a Name (line 601):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34735 = {}
    # Getting the type of 'call_assignment_32610' (line 601)
    call_assignment_32610_34732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 4), 'call_assignment_32610', False)
    # Obtaining the member '__getitem__' of a type (line 601)
    getitem___34733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 4), call_assignment_32610_34732, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34736 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34733, *[int_34734], **kwargs_34735)
    
    # Assigning a type to the variable 'call_assignment_32611' (line 601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 4), 'call_assignment_32611', getitem___call_result_34736)
    
    # Assigning a Name to a Name (line 601):
    # Getting the type of 'call_assignment_32611' (line 601)
    call_assignment_32611_34737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 4), 'call_assignment_32611')
    # Assigning a type to the variable 's' (line 601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 4), 's', call_assignment_32611_34737)
    
    # Assigning a Call to a Name (line 601):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34741 = {}
    # Getting the type of 'call_assignment_32610' (line 601)
    call_assignment_32610_34738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 4), 'call_assignment_32610', False)
    # Obtaining the member '__getitem__' of a type (line 601)
    getitem___34739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 4), call_assignment_32610_34738, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34742 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34739, *[int_34740], **kwargs_34741)
    
    # Assigning a type to the variable 'call_assignment_32612' (line 601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 4), 'call_assignment_32612', getitem___call_result_34742)
    
    # Assigning a Name to a Name (line 601):
    # Getting the type of 'call_assignment_32612' (line 601)
    call_assignment_32612_34743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 4), 'call_assignment_32612')
    # Assigning a type to the variable 'o' (line 601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 7), 'o', call_assignment_32612_34743)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 602)
    s_34744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 11), 's')
    int_34745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 14), 'int')
    # Applying the binary operator '==' (line 602)
    result_eq_34746 = python_operator(stypy.reporting.localization.Localization(__file__, 602, 11), '==', s_34744, int_34745)
    
    
    # Getting the type of 'o' (line 602)
    o_34747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 20), 'o')
    str_34748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 23), 'str', 'Heipa')
    # Applying the binary operator '==' (line 602)
    result_eq_34749 = python_operator(stypy.reporting.localization.Localization(__file__, 602, 20), '==', o_34747, str_34748)
    
    # Applying the binary operator 'and' (line 602)
    result_and_keyword_34750 = python_operator(stypy.reporting.localization.Localization(__file__, 602, 11), 'and', result_eq_34746, result_eq_34749)
    
    
    # Call to print(...): (line 604)
    # Processing the call arguments (line 604)
    str_34752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 11), 'str', 'ok')
    # Processing the call keyword arguments (line 604)
    kwargs_34753 = {}
    # Getting the type of 'print' (line 604)
    print_34751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 4), 'print', False)
    # Calling print(args, kwargs) (line 604)
    print_call_result_34754 = invoke(stypy.reporting.localization.Localization(__file__, 604, 4), print_34751, *[str_34752], **kwargs_34753)
    
    
    # ################# End of 'test_posix(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_posix' in the type store
    # Getting the type of 'stypy_return_type' (line 558)
    stypy_return_type_34755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_34755)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_posix'
    return stypy_return_type_34755

# Assigning a type to the variable 'test_posix' (line 558)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 0), 'test_posix', test_posix)

@norecursion
def test_execute_in(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_execute_in'
    module_type_store = module_type_store.open_function_context('test_execute_in', 606, 0, False)
    
    # Passed parameters checking function
    test_execute_in.stypy_localization = localization
    test_execute_in.stypy_type_of_self = None
    test_execute_in.stypy_type_store = module_type_store
    test_execute_in.stypy_function_name = 'test_execute_in'
    test_execute_in.stypy_param_names_list = []
    test_execute_in.stypy_varargs_param_name = None
    test_execute_in.stypy_kwargs_param_name = 'kws'
    test_execute_in.stypy_call_defaults = defaults
    test_execute_in.stypy_call_varargs = varargs
    test_execute_in.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_execute_in', [], None, 'kws', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_execute_in', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_execute_in(...)' code ##################

    
    # Assigning a Call to a Name (line 607):
    
    # Assigning a Call to a Name (line 607):
    
    # Call to get_pythonexe(...): (line 607)
    # Processing the call keyword arguments (line 607)
    kwargs_34757 = {}
    # Getting the type of 'get_pythonexe' (line 607)
    get_pythonexe_34756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 16), 'get_pythonexe', False)
    # Calling get_pythonexe(args, kwargs) (line 607)
    get_pythonexe_call_result_34758 = invoke(stypy.reporting.localization.Localization(__file__, 607, 16), get_pythonexe_34756, *[], **kwargs_34757)
    
    # Assigning a type to the variable 'pythonexe' (line 607)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 4), 'pythonexe', get_pythonexe_call_result_34758)
    
    # Assigning a Call to a Name (line 608):
    
    # Assigning a Call to a Name (line 608):
    
    # Call to temp_file_name(...): (line 608)
    # Processing the call keyword arguments (line 608)
    kwargs_34760 = {}
    # Getting the type of 'temp_file_name' (line 608)
    temp_file_name_34759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 14), 'temp_file_name', False)
    # Calling temp_file_name(args, kwargs) (line 608)
    temp_file_name_call_result_34761 = invoke(stypy.reporting.localization.Localization(__file__, 608, 14), temp_file_name_34759, *[], **kwargs_34760)
    
    # Assigning a type to the variable 'tmpfile' (line 608)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 4), 'tmpfile', temp_file_name_call_result_34761)
    
    # Assigning a Call to a Name (line 609):
    
    # Assigning a Call to a Name (line 609):
    
    # Call to basename(...): (line 609)
    # Processing the call arguments (line 609)
    # Getting the type of 'tmpfile' (line 609)
    tmpfile_34765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 26), 'tmpfile', False)
    # Processing the call keyword arguments (line 609)
    kwargs_34766 = {}
    # Getting the type of 'os' (line 609)
    os_34762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 9), 'os', False)
    # Obtaining the member 'path' of a type (line 609)
    path_34763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 9), os_34762, 'path')
    # Obtaining the member 'basename' of a type (line 609)
    basename_34764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 9), path_34763, 'basename')
    # Calling basename(args, kwargs) (line 609)
    basename_call_result_34767 = invoke(stypy.reporting.localization.Localization(__file__, 609, 9), basename_34764, *[tmpfile_34765], **kwargs_34766)
    
    # Assigning a type to the variable 'fn' (line 609)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 4), 'fn', basename_call_result_34767)
    
    # Assigning a Call to a Name (line 610):
    
    # Assigning a Call to a Name (line 610):
    
    # Call to dirname(...): (line 610)
    # Processing the call arguments (line 610)
    # Getting the type of 'tmpfile' (line 610)
    tmpfile_34771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 29), 'tmpfile', False)
    # Processing the call keyword arguments (line 610)
    kwargs_34772 = {}
    # Getting the type of 'os' (line 610)
    os_34768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 13), 'os', False)
    # Obtaining the member 'path' of a type (line 610)
    path_34769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 13), os_34768, 'path')
    # Obtaining the member 'dirname' of a type (line 610)
    dirname_34770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 13), path_34769, 'dirname')
    # Calling dirname(args, kwargs) (line 610)
    dirname_call_result_34773 = invoke(stypy.reporting.localization.Localization(__file__, 610, 13), dirname_34770, *[tmpfile_34771], **kwargs_34772)
    
    # Assigning a type to the variable 'tmpdir' (line 610)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 4), 'tmpdir', dirname_call_result_34773)
    
    # Assigning a Call to a Name (line 611):
    
    # Assigning a Call to a Name (line 611):
    
    # Call to open(...): (line 611)
    # Processing the call arguments (line 611)
    # Getting the type of 'tmpfile' (line 611)
    tmpfile_34775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 13), 'tmpfile', False)
    str_34776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 22), 'str', 'w')
    # Processing the call keyword arguments (line 611)
    kwargs_34777 = {}
    # Getting the type of 'open' (line 611)
    open_34774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 8), 'open', False)
    # Calling open(args, kwargs) (line 611)
    open_call_result_34778 = invoke(stypy.reporting.localization.Localization(__file__, 611, 8), open_34774, *[tmpfile_34775, str_34776], **kwargs_34777)
    
    # Assigning a type to the variable 'f' (line 611)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 4), 'f', open_call_result_34778)
    
    # Call to write(...): (line 612)
    # Processing the call arguments (line 612)
    str_34781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 12), 'str', 'Hello')
    # Processing the call keyword arguments (line 612)
    kwargs_34782 = {}
    # Getting the type of 'f' (line 612)
    f_34779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 4), 'f', False)
    # Obtaining the member 'write' of a type (line 612)
    write_34780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 4), f_34779, 'write')
    # Calling write(args, kwargs) (line 612)
    write_call_result_34783 = invoke(stypy.reporting.localization.Localization(__file__, 612, 4), write_34780, *[str_34781], **kwargs_34782)
    
    
    # Call to close(...): (line 613)
    # Processing the call keyword arguments (line 613)
    kwargs_34786 = {}
    # Getting the type of 'f' (line 613)
    f_34784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 4), 'f', False)
    # Obtaining the member 'close' of a type (line 613)
    close_34785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 4), f_34784, 'close')
    # Calling close(args, kwargs) (line 613)
    close_call_result_34787 = invoke(stypy.reporting.localization.Localization(__file__, 613, 4), close_34785, *[], **kwargs_34786)
    
    
    # Assigning a Call to a Tuple (line 615):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 615)
    # Processing the call arguments (line 615)
    str_34789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 24), 'str', '%s -c "print \'Ignore the following IOError:\',open(%r,\'r\')"')
    
    # Obtaining an instance of the builtin type 'tuple' (line 616)
    tuple_34790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 616)
    # Adding element type (line 616)
    # Getting the type of 'pythonexe' (line 616)
    pythonexe_34791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 44), 'pythonexe', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 616, 44), tuple_34790, pythonexe_34791)
    # Adding element type (line 616)
    # Getting the type of 'fn' (line 616)
    fn_34792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 55), 'fn', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 616, 44), tuple_34790, fn_34792)
    
    # Applying the binary operator '%' (line 615)
    result_mod_34793 = python_operator(stypy.reporting.localization.Localization(__file__, 615, 24), '%', str_34789, tuple_34790)
    
    # Processing the call keyword arguments (line 615)
    # Getting the type of 'kws' (line 616)
    kws_34794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 61), 'kws', False)
    kwargs_34795 = {'kws_34794': kws_34794}
    # Getting the type of 'exec_command' (line 615)
    exec_command_34788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 11), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 615)
    exec_command_call_result_34796 = invoke(stypy.reporting.localization.Localization(__file__, 615, 11), exec_command_34788, *[result_mod_34793], **kwargs_34795)
    
    # Assigning a type to the variable 'call_assignment_32613' (line 615)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 4), 'call_assignment_32613', exec_command_call_result_34796)
    
    # Assigning a Call to a Name (line 615):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34800 = {}
    # Getting the type of 'call_assignment_32613' (line 615)
    call_assignment_32613_34797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 4), 'call_assignment_32613', False)
    # Obtaining the member '__getitem__' of a type (line 615)
    getitem___34798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 4), call_assignment_32613_34797, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34801 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34798, *[int_34799], **kwargs_34800)
    
    # Assigning a type to the variable 'call_assignment_32614' (line 615)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 4), 'call_assignment_32614', getitem___call_result_34801)
    
    # Assigning a Name to a Name (line 615):
    # Getting the type of 'call_assignment_32614' (line 615)
    call_assignment_32614_34802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 4), 'call_assignment_32614')
    # Assigning a type to the variable 's' (line 615)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 4), 's', call_assignment_32614_34802)
    
    # Assigning a Call to a Name (line 615):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34806 = {}
    # Getting the type of 'call_assignment_32613' (line 615)
    call_assignment_32613_34803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 4), 'call_assignment_32613', False)
    # Obtaining the member '__getitem__' of a type (line 615)
    getitem___34804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 4), call_assignment_32613_34803, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34807 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34804, *[int_34805], **kwargs_34806)
    
    # Assigning a type to the variable 'call_assignment_32615' (line 615)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 4), 'call_assignment_32615', getitem___call_result_34807)
    
    # Assigning a Name to a Name (line 615):
    # Getting the type of 'call_assignment_32615' (line 615)
    call_assignment_32615_34808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 4), 'call_assignment_32615')
    # Assigning a type to the variable 'o' (line 615)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 7), 'o', call_assignment_32615_34808)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    # Getting the type of 's' (line 617)
    s_34809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 11), 's')
    
    # Getting the type of 'o' (line 617)
    o_34810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 17), 'o')
    str_34811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 20), 'str', '')
    # Applying the binary operator '!=' (line 617)
    result_ne_34812 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 17), '!=', o_34810, str_34811)
    
    # Applying the binary operator 'and' (line 617)
    result_and_keyword_34813 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 11), 'and', s_34809, result_ne_34812)
    
    
    # Assigning a Call to a Tuple (line 618):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 618)
    # Processing the call arguments (line 618)
    str_34815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 24), 'str', '%s -c "print open(%r,\'r\').read()"')
    
    # Obtaining an instance of the builtin type 'tuple' (line 618)
    tuple_34816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 65), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 618)
    # Adding element type (line 618)
    # Getting the type of 'pythonexe' (line 618)
    pythonexe_34817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 65), 'pythonexe', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 618, 65), tuple_34816, pythonexe_34817)
    # Adding element type (line 618)
    # Getting the type of 'fn' (line 618)
    fn_34818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 76), 'fn', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 618, 65), tuple_34816, fn_34818)
    
    # Applying the binary operator '%' (line 618)
    result_mod_34819 = python_operator(stypy.reporting.localization.Localization(__file__, 618, 24), '%', str_34815, tuple_34816)
    
    # Processing the call keyword arguments (line 618)
    # Getting the type of 'tmpdir' (line 619)
    tmpdir_34820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 36), 'tmpdir', False)
    keyword_34821 = tmpdir_34820
    # Getting the type of 'kws' (line 619)
    kws_34822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 45), 'kws', False)
    kwargs_34823 = {'kws_34822': kws_34822, 'execute_in': keyword_34821}
    # Getting the type of 'exec_command' (line 618)
    exec_command_34814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 11), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 618)
    exec_command_call_result_34824 = invoke(stypy.reporting.localization.Localization(__file__, 618, 11), exec_command_34814, *[result_mod_34819], **kwargs_34823)
    
    # Assigning a type to the variable 'call_assignment_32616' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'call_assignment_32616', exec_command_call_result_34824)
    
    # Assigning a Call to a Name (line 618):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34828 = {}
    # Getting the type of 'call_assignment_32616' (line 618)
    call_assignment_32616_34825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'call_assignment_32616', False)
    # Obtaining the member '__getitem__' of a type (line 618)
    getitem___34826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 4), call_assignment_32616_34825, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34829 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34826, *[int_34827], **kwargs_34828)
    
    # Assigning a type to the variable 'call_assignment_32617' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'call_assignment_32617', getitem___call_result_34829)
    
    # Assigning a Name to a Name (line 618):
    # Getting the type of 'call_assignment_32617' (line 618)
    call_assignment_32617_34830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'call_assignment_32617')
    # Assigning a type to the variable 's' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 's', call_assignment_32617_34830)
    
    # Assigning a Call to a Name (line 618):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34834 = {}
    # Getting the type of 'call_assignment_32616' (line 618)
    call_assignment_32616_34831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'call_assignment_32616', False)
    # Obtaining the member '__getitem__' of a type (line 618)
    getitem___34832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 4), call_assignment_32616_34831, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34835 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34832, *[int_34833], **kwargs_34834)
    
    # Assigning a type to the variable 'call_assignment_32618' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'call_assignment_32618', getitem___call_result_34835)
    
    # Assigning a Name to a Name (line 618):
    # Getting the type of 'call_assignment_32618' (line 618)
    call_assignment_32618_34836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'call_assignment_32618')
    # Assigning a type to the variable 'o' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 7), 'o', call_assignment_32618_34836)
    # Evaluating assert statement condition
    
    # Evaluating a boolean operation
    
    # Getting the type of 's' (line 620)
    s_34837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 11), 's')
    int_34838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 14), 'int')
    # Applying the binary operator '==' (line 620)
    result_eq_34839 = python_operator(stypy.reporting.localization.Localization(__file__, 620, 11), '==', s_34837, int_34838)
    
    
    # Getting the type of 'o' (line 620)
    o_34840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 20), 'o')
    str_34841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 23), 'str', 'Hello')
    # Applying the binary operator '==' (line 620)
    result_eq_34842 = python_operator(stypy.reporting.localization.Localization(__file__, 620, 20), '==', o_34840, str_34841)
    
    # Applying the binary operator 'and' (line 620)
    result_and_keyword_34843 = python_operator(stypy.reporting.localization.Localization(__file__, 620, 11), 'and', result_eq_34839, result_eq_34842)
    
    
    # Call to remove(...): (line 621)
    # Processing the call arguments (line 621)
    # Getting the type of 'tmpfile' (line 621)
    tmpfile_34846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 14), 'tmpfile', False)
    # Processing the call keyword arguments (line 621)
    kwargs_34847 = {}
    # Getting the type of 'os' (line 621)
    os_34844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 4), 'os', False)
    # Obtaining the member 'remove' of a type (line 621)
    remove_34845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 4), os_34844, 'remove')
    # Calling remove(args, kwargs) (line 621)
    remove_call_result_34848 = invoke(stypy.reporting.localization.Localization(__file__, 621, 4), remove_34845, *[tmpfile_34846], **kwargs_34847)
    
    
    # Call to print(...): (line 622)
    # Processing the call arguments (line 622)
    str_34850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 11), 'str', 'ok')
    # Processing the call keyword arguments (line 622)
    kwargs_34851 = {}
    # Getting the type of 'print' (line 622)
    print_34849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 4), 'print', False)
    # Calling print(args, kwargs) (line 622)
    print_call_result_34852 = invoke(stypy.reporting.localization.Localization(__file__, 622, 4), print_34849, *[str_34850], **kwargs_34851)
    
    
    # ################# End of 'test_execute_in(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_execute_in' in the type store
    # Getting the type of 'stypy_return_type' (line 606)
    stypy_return_type_34853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_34853)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_execute_in'
    return stypy_return_type_34853

# Assigning a type to the variable 'test_execute_in' (line 606)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 0), 'test_execute_in', test_execute_in)

@norecursion
def test_svn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_svn'
    module_type_store = module_type_store.open_function_context('test_svn', 624, 0, False)
    
    # Passed parameters checking function
    test_svn.stypy_localization = localization
    test_svn.stypy_type_of_self = None
    test_svn.stypy_type_store = module_type_store
    test_svn.stypy_function_name = 'test_svn'
    test_svn.stypy_param_names_list = []
    test_svn.stypy_varargs_param_name = None
    test_svn.stypy_kwargs_param_name = 'kws'
    test_svn.stypy_call_defaults = defaults
    test_svn.stypy_call_varargs = varargs
    test_svn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_svn', [], None, 'kws', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_svn', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_svn(...)' code ##################

    
    # Assigning a Call to a Tuple (line 625):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 625)
    # Processing the call arguments (line 625)
    
    # Obtaining an instance of the builtin type 'list' (line 625)
    list_34855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 625)
    # Adding element type (line 625)
    str_34856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 25), 'str', 'svn')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 625, 24), list_34855, str_34856)
    # Adding element type (line 625)
    str_34857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 32), 'str', 'status')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 625, 24), list_34855, str_34857)
    
    # Processing the call keyword arguments (line 625)
    # Getting the type of 'kws' (line 625)
    kws_34858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 44), 'kws', False)
    kwargs_34859 = {'kws_34858': kws_34858}
    # Getting the type of 'exec_command' (line 625)
    exec_command_34854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 11), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 625)
    exec_command_call_result_34860 = invoke(stypy.reporting.localization.Localization(__file__, 625, 11), exec_command_34854, *[list_34855], **kwargs_34859)
    
    # Assigning a type to the variable 'call_assignment_32619' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 4), 'call_assignment_32619', exec_command_call_result_34860)
    
    # Assigning a Call to a Name (line 625):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34864 = {}
    # Getting the type of 'call_assignment_32619' (line 625)
    call_assignment_32619_34861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 4), 'call_assignment_32619', False)
    # Obtaining the member '__getitem__' of a type (line 625)
    getitem___34862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 4), call_assignment_32619_34861, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34865 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34862, *[int_34863], **kwargs_34864)
    
    # Assigning a type to the variable 'call_assignment_32620' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 4), 'call_assignment_32620', getitem___call_result_34865)
    
    # Assigning a Name to a Name (line 625):
    # Getting the type of 'call_assignment_32620' (line 625)
    call_assignment_32620_34866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 4), 'call_assignment_32620')
    # Assigning a type to the variable 's' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 4), 's', call_assignment_32620_34866)
    
    # Assigning a Call to a Name (line 625):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 4), 'int')
    # Processing the call keyword arguments
    kwargs_34870 = {}
    # Getting the type of 'call_assignment_32619' (line 625)
    call_assignment_32619_34867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 4), 'call_assignment_32619', False)
    # Obtaining the member '__getitem__' of a type (line 625)
    getitem___34868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 4), call_assignment_32619_34867, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34871 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34868, *[int_34869], **kwargs_34870)
    
    # Assigning a type to the variable 'call_assignment_32621' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 4), 'call_assignment_32621', getitem___call_result_34871)
    
    # Assigning a Name to a Name (line 625):
    # Getting the type of 'call_assignment_32621' (line 625)
    call_assignment_32621_34872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 4), 'call_assignment_32621')
    # Assigning a type to the variable 'o' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 7), 'o', call_assignment_32621_34872)
    # Evaluating assert statement condition
    # Getting the type of 's' (line 626)
    s_34873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 11), 's')
    
    # Call to print(...): (line 627)
    # Processing the call arguments (line 627)
    str_34875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 11), 'str', 'svn ok')
    # Processing the call keyword arguments (line 627)
    kwargs_34876 = {}
    # Getting the type of 'print' (line 627)
    print_34874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 4), 'print', False)
    # Calling print(args, kwargs) (line 627)
    print_call_result_34877 = invoke(stypy.reporting.localization.Localization(__file__, 627, 4), print_34874, *[str_34875], **kwargs_34876)
    
    
    # ################# End of 'test_svn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_svn' in the type store
    # Getting the type of 'stypy_return_type' (line 624)
    stypy_return_type_34878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_34878)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_svn'
    return stypy_return_type_34878

# Assigning a type to the variable 'test_svn' (line 624)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 0), 'test_svn', test_svn)

@norecursion
def test_cl(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_cl'
    module_type_store = module_type_store.open_function_context('test_cl', 629, 0, False)
    
    # Passed parameters checking function
    test_cl.stypy_localization = localization
    test_cl.stypy_type_of_self = None
    test_cl.stypy_type_store = module_type_store
    test_cl.stypy_function_name = 'test_cl'
    test_cl.stypy_param_names_list = []
    test_cl.stypy_varargs_param_name = None
    test_cl.stypy_kwargs_param_name = 'kws'
    test_cl.stypy_call_defaults = defaults
    test_cl.stypy_call_varargs = varargs
    test_cl.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_cl', [], None, 'kws', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_cl', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_cl(...)' code ##################

    
    
    # Getting the type of 'os' (line 630)
    os_34879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 7), 'os')
    # Obtaining the member 'name' of a type (line 630)
    name_34880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 7), os_34879, 'name')
    str_34881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 16), 'str', 'nt')
    # Applying the binary operator '==' (line 630)
    result_eq_34882 = python_operator(stypy.reporting.localization.Localization(__file__, 630, 7), '==', name_34880, str_34881)
    
    # Testing the type of an if condition (line 630)
    if_condition_34883 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 630, 4), result_eq_34882)
    # Assigning a type to the variable 'if_condition_34883' (line 630)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 4), 'if_condition_34883', if_condition_34883)
    # SSA begins for if statement (line 630)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 631):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 631)
    # Processing the call arguments (line 631)
    
    # Obtaining an instance of the builtin type 'list' (line 631)
    list_34885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 631)
    # Adding element type (line 631)
    str_34886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 29), 'str', 'cl')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 631, 28), list_34885, str_34886)
    # Adding element type (line 631)
    str_34887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 35), 'str', '/V')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 631, 28), list_34885, str_34887)
    
    # Processing the call keyword arguments (line 631)
    # Getting the type of 'kws' (line 631)
    kws_34888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 43), 'kws', False)
    kwargs_34889 = {'kws_34888': kws_34888}
    # Getting the type of 'exec_command' (line 631)
    exec_command_34884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 15), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 631)
    exec_command_call_result_34890 = invoke(stypy.reporting.localization.Localization(__file__, 631, 15), exec_command_34884, *[list_34885], **kwargs_34889)
    
    # Assigning a type to the variable 'call_assignment_32622' (line 631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 8), 'call_assignment_32622', exec_command_call_result_34890)
    
    # Assigning a Call to a Name (line 631):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 8), 'int')
    # Processing the call keyword arguments
    kwargs_34894 = {}
    # Getting the type of 'call_assignment_32622' (line 631)
    call_assignment_32622_34891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 8), 'call_assignment_32622', False)
    # Obtaining the member '__getitem__' of a type (line 631)
    getitem___34892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 8), call_assignment_32622_34891, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34895 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34892, *[int_34893], **kwargs_34894)
    
    # Assigning a type to the variable 'call_assignment_32623' (line 631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 8), 'call_assignment_32623', getitem___call_result_34895)
    
    # Assigning a Name to a Name (line 631):
    # Getting the type of 'call_assignment_32623' (line 631)
    call_assignment_32623_34896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 8), 'call_assignment_32623')
    # Assigning a type to the variable 's' (line 631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 8), 's', call_assignment_32623_34896)
    
    # Assigning a Call to a Name (line 631):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_34899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 8), 'int')
    # Processing the call keyword arguments
    kwargs_34900 = {}
    # Getting the type of 'call_assignment_32622' (line 631)
    call_assignment_32622_34897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 8), 'call_assignment_32622', False)
    # Obtaining the member '__getitem__' of a type (line 631)
    getitem___34898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 8), call_assignment_32622_34897, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_34901 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___34898, *[int_34899], **kwargs_34900)
    
    # Assigning a type to the variable 'call_assignment_32624' (line 631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 8), 'call_assignment_32624', getitem___call_result_34901)
    
    # Assigning a Name to a Name (line 631):
    # Getting the type of 'call_assignment_32624' (line 631)
    call_assignment_32624_34902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 8), 'call_assignment_32624')
    # Assigning a type to the variable 'o' (line 631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 11), 'o', call_assignment_32624_34902)
    # Evaluating assert statement condition
    # Getting the type of 's' (line 632)
    s_34903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 15), 's')
    
    # Call to print(...): (line 633)
    # Processing the call arguments (line 633)
    str_34905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 15), 'str', 'cl ok')
    # Processing the call keyword arguments (line 633)
    kwargs_34906 = {}
    # Getting the type of 'print' (line 633)
    print_34904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 8), 'print', False)
    # Calling print(args, kwargs) (line 633)
    print_call_result_34907 = invoke(stypy.reporting.localization.Localization(__file__, 633, 8), print_34904, *[str_34905], **kwargs_34906)
    
    # SSA join for if statement (line 630)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_cl(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_cl' in the type store
    # Getting the type of 'stypy_return_type' (line 629)
    stypy_return_type_34908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_34908)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_cl'
    return stypy_return_type_34908

# Assigning a type to the variable 'test_cl' (line 629)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 0), 'test_cl', test_cl)


# Getting the type of 'os' (line 635)
os_34909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 3), 'os')
# Obtaining the member 'name' of a type (line 635)
name_34910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 3), os_34909, 'name')
str_34911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 12), 'str', 'posix')
# Applying the binary operator '==' (line 635)
result_eq_34912 = python_operator(stypy.reporting.localization.Localization(__file__, 635, 3), '==', name_34910, str_34911)

# Testing the type of an if condition (line 635)
if_condition_34913 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 635, 0), result_eq_34912)
# Assigning a type to the variable 'if_condition_34913' (line 635)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 0), 'if_condition_34913', if_condition_34913)
# SSA begins for if statement (line 635)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Name (line 636):

# Assigning a Name to a Name (line 636):
# Getting the type of 'test_posix' (line 636)
test_posix_34914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 11), 'test_posix')
# Assigning a type to the variable 'test' (line 636)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 4), 'test', test_posix_34914)
# SSA branch for the else part of an if statement (line 635)
module_type_store.open_ssa_branch('else')


# Getting the type of 'os' (line 637)
os_34915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 5), 'os')
# Obtaining the member 'name' of a type (line 637)
name_34916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 5), os_34915, 'name')

# Obtaining an instance of the builtin type 'list' (line 637)
list_34917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 637)
# Adding element type (line 637)
str_34918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 17), 'str', 'nt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 637, 16), list_34917, str_34918)
# Adding element type (line 637)
str_34919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 23), 'str', 'dos')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 637, 16), list_34917, str_34919)

# Applying the binary operator 'in' (line 637)
result_contains_34920 = python_operator(stypy.reporting.localization.Localization(__file__, 637, 5), 'in', name_34916, list_34917)

# Testing the type of an if condition (line 637)
if_condition_34921 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 637, 5), result_contains_34920)
# Assigning a type to the variable 'if_condition_34921' (line 637)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 5), 'if_condition_34921', if_condition_34921)
# SSA begins for if statement (line 637)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Name (line 638):

# Assigning a Name to a Name (line 638):
# Getting the type of 'test_nt' (line 638)
test_nt_34922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 11), 'test_nt')
# Assigning a type to the variable 'test' (line 638)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 4), 'test', test_nt_34922)
# SSA branch for the else part of an if statement (line 637)
module_type_store.open_ssa_branch('else')

# Call to NotImplementedError(...): (line 640)
# Processing the call arguments (line 640)
str_34924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 30), 'str', 'exec_command tests for ')
# Getting the type of 'os' (line 640)
os_34925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 57), 'os', False)
# Obtaining the member 'name' of a type (line 640)
name_34926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 57), os_34925, 'name')
# Processing the call keyword arguments (line 640)
kwargs_34927 = {}
# Getting the type of 'NotImplementedError' (line 640)
NotImplementedError_34923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 10), 'NotImplementedError', False)
# Calling NotImplementedError(args, kwargs) (line 640)
NotImplementedError_call_result_34928 = invoke(stypy.reporting.localization.Localization(__file__, 640, 10), NotImplementedError_34923, *[str_34924, name_34926], **kwargs_34927)

ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 640, 4), NotImplementedError_call_result_34928, 'raise parameter', BaseException)
# SSA join for if statement (line 637)
module_type_store = module_type_store.join_ssa_context()

# SSA join for if statement (line 635)
module_type_store = module_type_store.join_ssa_context()


if (__name__ == '__main__'):
    
    # Call to test(...): (line 646)
    # Processing the call keyword arguments (line 646)
    int_34930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 17), 'int')
    keyword_34931 = int_34930
    kwargs_34932 = {'use_tee': keyword_34931}
    # Getting the type of 'test' (line 646)
    test_34929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 4), 'test', False)
    # Calling test(args, kwargs) (line 646)
    test_call_result_34933 = invoke(stypy.reporting.localization.Localization(__file__, 646, 4), test_34929, *[], **kwargs_34932)
    
    
    # Call to test(...): (line 647)
    # Processing the call keyword arguments (line 647)
    int_34935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 647, 17), 'int')
    keyword_34936 = int_34935
    kwargs_34937 = {'use_tee': keyword_34936}
    # Getting the type of 'test' (line 647)
    test_34934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 4), 'test', False)
    # Calling test(args, kwargs) (line 647)
    test_call_result_34938 = invoke(stypy.reporting.localization.Localization(__file__, 647, 4), test_34934, *[], **kwargs_34937)
    
    
    # Call to test_execute_in(...): (line 648)
    # Processing the call keyword arguments (line 648)
    int_34940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, 28), 'int')
    keyword_34941 = int_34940
    kwargs_34942 = {'use_tee': keyword_34941}
    # Getting the type of 'test_execute_in' (line 648)
    test_execute_in_34939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 4), 'test_execute_in', False)
    # Calling test_execute_in(args, kwargs) (line 648)
    test_execute_in_call_result_34943 = invoke(stypy.reporting.localization.Localization(__file__, 648, 4), test_execute_in_34939, *[], **kwargs_34942)
    
    
    # Call to test_execute_in(...): (line 649)
    # Processing the call keyword arguments (line 649)
    int_34945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 28), 'int')
    keyword_34946 = int_34945
    kwargs_34947 = {'use_tee': keyword_34946}
    # Getting the type of 'test_execute_in' (line 649)
    test_execute_in_34944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'test_execute_in', False)
    # Calling test_execute_in(args, kwargs) (line 649)
    test_execute_in_call_result_34948 = invoke(stypy.reporting.localization.Localization(__file__, 649, 4), test_execute_in_34944, *[], **kwargs_34947)
    
    
    # Call to test_svn(...): (line 650)
    # Processing the call keyword arguments (line 650)
    int_34950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 21), 'int')
    keyword_34951 = int_34950
    kwargs_34952 = {'use_tee': keyword_34951}
    # Getting the type of 'test_svn' (line 650)
    test_svn_34949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 4), 'test_svn', False)
    # Calling test_svn(args, kwargs) (line 650)
    test_svn_call_result_34953 = invoke(stypy.reporting.localization.Localization(__file__, 650, 4), test_svn_34949, *[], **kwargs_34952)
    
    
    # Call to test_cl(...): (line 651)
    # Processing the call keyword arguments (line 651)
    int_34955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 20), 'int')
    keyword_34956 = int_34955
    kwargs_34957 = {'use_tee': keyword_34956}
    # Getting the type of 'test_cl' (line 651)
    test_cl_34954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 4), 'test_cl', False)
    # Calling test_cl(args, kwargs) (line 651)
    test_cl_call_result_34958 = invoke(stypy.reporting.localization.Localization(__file__, 651, 4), test_cl_34954, *[], **kwargs_34957)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

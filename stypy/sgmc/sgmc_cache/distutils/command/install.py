
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.command.install
2: 
3: Implements the Distutils 'install' command.'''
4: 
5: from distutils import log
6: 
7: # This module should be kept compatible with Python 2.1.
8: 
9: __revision__ = "$Id$"
10: 
11: import sys, os, string
12: from types import *
13: from distutils.core import Command
14: from distutils.debug import DEBUG
15: from distutils.sysconfig import get_config_vars
16: from distutils.errors import DistutilsPlatformError
17: from distutils.file_util import write_file
18: from distutils.util import convert_path, subst_vars, change_root
19: from distutils.util import get_platform
20: from distutils.errors import DistutilsOptionError
21: from site import USER_BASE
22: from site import USER_SITE
23: 
24: 
25: if sys.version < "2.2":
26:     WINDOWS_SCHEME = {
27:         'purelib': '$base',
28:         'platlib': '$base',
29:         'headers': '$base/Include/$dist_name',
30:         'scripts': '$base/Scripts',
31:         'data'   : '$base',
32:     }
33: else:
34:     WINDOWS_SCHEME = {
35:         'purelib': '$base/Lib/site-packages',
36:         'platlib': '$base/Lib/site-packages',
37:         'headers': '$base/Include/$dist_name',
38:         'scripts': '$base/Scripts',
39:         'data'   : '$base',
40:     }
41: 
42: INSTALL_SCHEMES = {
43:     'unix_prefix': {
44:         'purelib': '$base/lib/python$py_version_short/site-packages',
45:         'platlib': '$platbase/lib/python$py_version_short/site-packages',
46:         'headers': '$base/include/python$py_version_short/$dist_name',
47:         'scripts': '$base/bin',
48:         'data'   : '$base',
49:         },
50:     'unix_home': {
51:         'purelib': '$base/lib/python',
52:         'platlib': '$base/lib/python',
53:         'headers': '$base/include/python/$dist_name',
54:         'scripts': '$base/bin',
55:         'data'   : '$base',
56:         },
57:     'unix_user': {
58:         'purelib': '$usersite',
59:         'platlib': '$usersite',
60:         'headers': '$userbase/include/python$py_version_short/$dist_name',
61:         'scripts': '$userbase/bin',
62:         'data'   : '$userbase',
63:         },
64:     'nt': WINDOWS_SCHEME,
65:     'nt_user': {
66:         'purelib': '$usersite',
67:         'platlib': '$usersite',
68:         'headers': '$userbase/Python$py_version_nodot/Include/$dist_name',
69:         'scripts': '$userbase/Scripts',
70:         'data'   : '$userbase',
71:         },
72:     'os2': {
73:         'purelib': '$base/Lib/site-packages',
74:         'platlib': '$base/Lib/site-packages',
75:         'headers': '$base/Include/$dist_name',
76:         'scripts': '$base/Scripts',
77:         'data'   : '$base',
78:         },
79:     'os2_home': {
80:         'purelib': '$usersite',
81:         'platlib': '$usersite',
82:         'headers': '$userbase/include/python$py_version_short/$dist_name',
83:         'scripts': '$userbase/bin',
84:         'data'   : '$userbase',
85:         },
86:     }
87: 
88: # The keys to an installation scheme; if any new types of files are to be
89: # installed, be sure to add an entry to every installation scheme above,
90: # and to SCHEME_KEYS here.
91: SCHEME_KEYS = ('purelib', 'platlib', 'headers', 'scripts', 'data')
92: 
93: 
94: class install (Command):
95: 
96:     description = "install everything from build directory"
97: 
98:     user_options = [
99:         # Select installation scheme and set base director(y|ies)
100:         ('prefix=', None,
101:          "installation prefix"),
102:         ('exec-prefix=', None,
103:          "(Unix only) prefix for platform-specific files"),
104:         ('home=', None,
105:          "(Unix only) home directory to install under"),
106:         ('user', None,
107:          "install in user site-package '%s'" % USER_SITE),
108: 
109:         # Or, just set the base director(y|ies)
110:         ('install-base=', None,
111:          "base installation directory (instead of --prefix or --home)"),
112:         ('install-platbase=', None,
113:          "base installation directory for platform-specific files " +
114:          "(instead of --exec-prefix or --home)"),
115:         ('root=', None,
116:          "install everything relative to this alternate root directory"),
117: 
118:         # Or, explicitly set the installation scheme
119:         ('install-purelib=', None,
120:          "installation directory for pure Python module distributions"),
121:         ('install-platlib=', None,
122:          "installation directory for non-pure module distributions"),
123:         ('install-lib=', None,
124:          "installation directory for all module distributions " +
125:          "(overrides --install-purelib and --install-platlib)"),
126: 
127:         ('install-headers=', None,
128:          "installation directory for C/C++ headers"),
129:         ('install-scripts=', None,
130:          "installation directory for Python scripts"),
131:         ('install-data=', None,
132:          "installation directory for data files"),
133: 
134:         # Byte-compilation options -- see install_lib.py for details, as
135:         # these are duplicated from there (but only install_lib does
136:         # anything with them).
137:         ('compile', 'c', "compile .py to .pyc [default]"),
138:         ('no-compile', None, "don't compile .py files"),
139:         ('optimize=', 'O',
140:          "also compile with optimization: -O1 for \"python -O\", "
141:          "-O2 for \"python -OO\", and -O0 to disable [default: -O0]"),
142: 
143:         # Miscellaneous control options
144:         ('force', 'f',
145:          "force installation (overwrite any existing files)"),
146:         ('skip-build', None,
147:          "skip rebuilding everything (for testing/debugging)"),
148: 
149:         # Where to install documentation (eventually!)
150:         #('doc-format=', None, "format of documentation to generate"),
151:         #('install-man=', None, "directory for Unix man pages"),
152:         #('install-html=', None, "directory for HTML documentation"),
153:         #('install-info=', None, "directory for GNU info files"),
154: 
155:         ('record=', None,
156:          "filename in which to record list of installed files"),
157:         ]
158: 
159:     boolean_options = ['compile', 'force', 'skip-build', 'user']
160:     negative_opt = {'no-compile' : 'compile'}
161: 
162: 
163:     def initialize_options (self):
164: 
165:         # High-level options: these select both an installation base
166:         # and scheme.
167:         self.prefix = None
168:         self.exec_prefix = None
169:         self.home = None
170:         self.user = 0
171: 
172:         # These select only the installation base; it's up to the user to
173:         # specify the installation scheme (currently, that means supplying
174:         # the --install-{platlib,purelib,scripts,data} options).
175:         self.install_base = None
176:         self.install_platbase = None
177:         self.root = None
178: 
179:         # These options are the actual installation directories; if not
180:         # supplied by the user, they are filled in using the installation
181:         # scheme implied by prefix/exec-prefix/home and the contents of
182:         # that installation scheme.
183:         self.install_purelib = None     # for pure module distributions
184:         self.install_platlib = None     # non-pure (dists w/ extensions)
185:         self.install_headers = None     # for C/C++ headers
186:         self.install_lib = None         # set to either purelib or platlib
187:         self.install_scripts = None
188:         self.install_data = None
189:         self.install_userbase = USER_BASE
190:         self.install_usersite = USER_SITE
191: 
192:         self.compile = None
193:         self.optimize = None
194: 
195:         # These two are for putting non-packagized distributions into their
196:         # own directory and creating a .pth file if it makes sense.
197:         # 'extra_path' comes from the setup file; 'install_path_file' can
198:         # be turned off if it makes no sense to install a .pth file.  (But
199:         # better to install it uselessly than to guess wrong and not
200:         # install it when it's necessary and would be used!)  Currently,
201:         # 'install_path_file' is always true unless some outsider meddles
202:         # with it.
203:         self.extra_path = None
204:         self.install_path_file = 1
205: 
206:         # 'force' forces installation, even if target files are not
207:         # out-of-date.  'skip_build' skips running the "build" command,
208:         # handy if you know it's not necessary.  'warn_dir' (which is *not*
209:         # a user option, it's just there so the bdist_* commands can turn
210:         # it off) determines whether we warn about installing to a
211:         # directory not in sys.path.
212:         self.force = 0
213:         self.skip_build = 0
214:         self.warn_dir = 1
215: 
216:         # These are only here as a conduit from the 'build' command to the
217:         # 'install_*' commands that do the real work.  ('build_base' isn't
218:         # actually used anywhere, but it might be useful in future.)  They
219:         # are not user options, because if the user told the install
220:         # command where the build directory is, that wouldn't affect the
221:         # build command.
222:         self.build_base = None
223:         self.build_lib = None
224: 
225:         # Not defined yet because we don't know anything about
226:         # documentation yet.
227:         #self.install_man = None
228:         #self.install_html = None
229:         #self.install_info = None
230: 
231:         self.record = None
232: 
233: 
234:     # -- Option finalizing methods -------------------------------------
235:     # (This is rather more involved than for most commands,
236:     # because this is where the policy for installing third-
237:     # party Python modules on various platforms given a wide
238:     # array of user input is decided.  Yes, it's quite complex!)
239: 
240:     def finalize_options (self):
241: 
242:         # This method (and its pliant slaves, like 'finalize_unix()',
243:         # 'finalize_other()', and 'select_scheme()') is where the default
244:         # installation directories for modules, extension modules, and
245:         # anything else we care to install from a Python module
246:         # distribution.  Thus, this code makes a pretty important policy
247:         # statement about how third-party stuff is added to a Python
248:         # installation!  Note that the actual work of installation is done
249:         # by the relatively simple 'install_*' commands; they just take
250:         # their orders from the installation directory options determined
251:         # here.
252: 
253:         # Check for errors/inconsistencies in the options; first, stuff
254:         # that's wrong on any platform.
255: 
256:         if ((self.prefix or self.exec_prefix or self.home) and
257:             (self.install_base or self.install_platbase)):
258:             raise DistutilsOptionError, \
259:                   ("must supply either prefix/exec-prefix/home or " +
260:                    "install-base/install-platbase -- not both")
261: 
262:         if self.home and (self.prefix or self.exec_prefix):
263:             raise DistutilsOptionError, \
264:                   "must supply either home or prefix/exec-prefix -- not both"
265: 
266:         if self.user and (self.prefix or self.exec_prefix or self.home or
267:                 self.install_base or self.install_platbase):
268:             raise DistutilsOptionError("can't combine user with prefix, "
269:                                        "exec_prefix/home, or install_(plat)base")
270: 
271:         # Next, stuff that's wrong (or dubious) only on certain platforms.
272:         if os.name != "posix":
273:             if self.exec_prefix:
274:                 self.warn("exec-prefix option ignored on this platform")
275:                 self.exec_prefix = None
276: 
277:         # Now the interesting logic -- so interesting that we farm it out
278:         # to other methods.  The goal of these methods is to set the final
279:         # values for the install_{lib,scripts,data,...}  options, using as
280:         # input a heady brew of prefix, exec_prefix, home, install_base,
281:         # install_platbase, user-supplied versions of
282:         # install_{purelib,platlib,lib,scripts,data,...}, and the
283:         # INSTALL_SCHEME dictionary above.  Phew!
284: 
285:         self.dump_dirs("pre-finalize_{unix,other}")
286: 
287:         if os.name == 'posix':
288:             self.finalize_unix()
289:         else:
290:             self.finalize_other()
291: 
292:         self.dump_dirs("post-finalize_{unix,other}()")
293: 
294:         # Expand configuration variables, tilde, etc. in self.install_base
295:         # and self.install_platbase -- that way, we can use $base or
296:         # $platbase in the other installation directories and not worry
297:         # about needing recursive variable expansion (shudder).
298: 
299:         py_version = (string.split(sys.version))[0]
300:         (prefix, exec_prefix) = get_config_vars('prefix', 'exec_prefix')
301:         self.config_vars = {'dist_name': self.distribution.get_name(),
302:                             'dist_version': self.distribution.get_version(),
303:                             'dist_fullname': self.distribution.get_fullname(),
304:                             'py_version': py_version,
305:                             'py_version_short': py_version[0:3],
306:                             'py_version_nodot': py_version[0] + py_version[2],
307:                             'sys_prefix': prefix,
308:                             'prefix': prefix,
309:                             'sys_exec_prefix': exec_prefix,
310:                             'exec_prefix': exec_prefix,
311:                             'userbase': self.install_userbase,
312:                             'usersite': self.install_usersite,
313:                            }
314:         self.expand_basedirs()
315: 
316:         self.dump_dirs("post-expand_basedirs()")
317: 
318:         # Now define config vars for the base directories so we can expand
319:         # everything else.
320:         self.config_vars['base'] = self.install_base
321:         self.config_vars['platbase'] = self.install_platbase
322: 
323:         if DEBUG:
324:             from pprint import pprint
325:             print "config vars:"
326:             pprint(self.config_vars)
327: 
328:         # Expand "~" and configuration variables in the installation
329:         # directories.
330:         self.expand_dirs()
331: 
332:         self.dump_dirs("post-expand_dirs()")
333: 
334:         # Create directories in the home dir:
335:         if self.user:
336:             self.create_home_path()
337: 
338:         # Pick the actual directory to install all modules to: either
339:         # install_purelib or install_platlib, depending on whether this
340:         # module distribution is pure or not.  Of course, if the user
341:         # already specified install_lib, use their selection.
342:         if self.install_lib is None:
343:             if self.distribution.ext_modules: # has extensions: non-pure
344:                 self.install_lib = self.install_platlib
345:             else:
346:                 self.install_lib = self.install_purelib
347: 
348: 
349:         # Convert directories from Unix /-separated syntax to the local
350:         # convention.
351:         self.convert_paths('lib', 'purelib', 'platlib',
352:                            'scripts', 'data', 'headers',
353:                            'userbase', 'usersite')
354: 
355:         # Well, we're not actually fully completely finalized yet: we still
356:         # have to deal with 'extra_path', which is the hack for allowing
357:         # non-packagized module distributions (hello, Numerical Python!) to
358:         # get their own directories.
359:         self.handle_extra_path()
360:         self.install_libbase = self.install_lib # needed for .pth file
361:         self.install_lib = os.path.join(self.install_lib, self.extra_dirs)
362: 
363:         # If a new root directory was supplied, make all the installation
364:         # dirs relative to it.
365:         if self.root is not None:
366:             self.change_roots('libbase', 'lib', 'purelib', 'platlib',
367:                               'scripts', 'data', 'headers')
368: 
369:         self.dump_dirs("after prepending root")
370: 
371:         # Find out the build directories, ie. where to install from.
372:         self.set_undefined_options('build',
373:                                    ('build_base', 'build_base'),
374:                                    ('build_lib', 'build_lib'))
375: 
376:         # Punt on doc directories for now -- after all, we're punting on
377:         # documentation completely!
378: 
379:     # finalize_options ()
380: 
381: 
382:     def dump_dirs (self, msg):
383:         if DEBUG:
384:             from distutils.fancy_getopt import longopt_xlate
385:             print msg + ":"
386:             for opt in self.user_options:
387:                 opt_name = opt[0]
388:                 if opt_name[-1] == "=":
389:                     opt_name = opt_name[0:-1]
390:                 if opt_name in self.negative_opt:
391:                     opt_name = string.translate(self.negative_opt[opt_name],
392:                                                 longopt_xlate)
393:                     val = not getattr(self, opt_name)
394:                 else:
395:                     opt_name = string.translate(opt_name, longopt_xlate)
396:                     val = getattr(self, opt_name)
397:                 print "  %s: %s" % (opt_name, val)
398: 
399: 
400:     def finalize_unix (self):
401: 
402:         if self.install_base is not None or self.install_platbase is not None:
403:             if ((self.install_lib is None and
404:                  self.install_purelib is None and
405:                  self.install_platlib is None) or
406:                 self.install_headers is None or
407:                 self.install_scripts is None or
408:                 self.install_data is None):
409:                 raise DistutilsOptionError, \
410:                       ("install-base or install-platbase supplied, but "
411:                       "installation scheme is incomplete")
412:             return
413: 
414:         if self.user:
415:             if self.install_userbase is None:
416:                 raise DistutilsPlatformError(
417:                     "User base directory is not specified")
418:             self.install_base = self.install_platbase = self.install_userbase
419:             self.select_scheme("unix_user")
420:         elif self.home is not None:
421:             self.install_base = self.install_platbase = self.home
422:             self.select_scheme("unix_home")
423:         else:
424:             if self.prefix is None:
425:                 if self.exec_prefix is not None:
426:                     raise DistutilsOptionError, \
427:                           "must not supply exec-prefix without prefix"
428: 
429:                 self.prefix = os.path.normpath(sys.prefix)
430:                 self.exec_prefix = os.path.normpath(sys.exec_prefix)
431: 
432:             else:
433:                 if self.exec_prefix is None:
434:                     self.exec_prefix = self.prefix
435: 
436:             self.install_base = self.prefix
437:             self.install_platbase = self.exec_prefix
438:             self.select_scheme("unix_prefix")
439: 
440:     # finalize_unix ()
441: 
442: 
443:     def finalize_other (self):          # Windows and Mac OS for now
444: 
445:         if self.user:
446:             if self.install_userbase is None:
447:                 raise DistutilsPlatformError(
448:                     "User base directory is not specified")
449:             self.install_base = self.install_platbase = self.install_userbase
450:             self.select_scheme(os.name + "_user")
451:         elif self.home is not None:
452:             self.install_base = self.install_platbase = self.home
453:             self.select_scheme("unix_home")
454:         else:
455:             if self.prefix is None:
456:                 self.prefix = os.path.normpath(sys.prefix)
457: 
458:             self.install_base = self.install_platbase = self.prefix
459:             try:
460:                 self.select_scheme(os.name)
461:             except KeyError:
462:                 raise DistutilsPlatformError, \
463:                       "I don't know how to install stuff on '%s'" % os.name
464: 
465:     # finalize_other ()
466: 
467: 
468:     def select_scheme (self, name):
469:         # it's the caller's problem if they supply a bad name!
470:         scheme = INSTALL_SCHEMES[name]
471:         for key in SCHEME_KEYS:
472:             attrname = 'install_' + key
473:             if getattr(self, attrname) is None:
474:                 setattr(self, attrname, scheme[key])
475: 
476: 
477:     def _expand_attrs (self, attrs):
478:         for attr in attrs:
479:             val = getattr(self, attr)
480:             if val is not None:
481:                 if os.name == 'posix' or os.name == 'nt':
482:                     val = os.path.expanduser(val)
483:                 val = subst_vars(val, self.config_vars)
484:                 setattr(self, attr, val)
485: 
486: 
487:     def expand_basedirs (self):
488:         self._expand_attrs(['install_base',
489:                             'install_platbase',
490:                             'root'])
491: 
492:     def expand_dirs (self):
493:         self._expand_attrs(['install_purelib',
494:                             'install_platlib',
495:                             'install_lib',
496:                             'install_headers',
497:                             'install_scripts',
498:                             'install_data',])
499: 
500: 
501:     def convert_paths (self, *names):
502:         for name in names:
503:             attr = "install_" + name
504:             setattr(self, attr, convert_path(getattr(self, attr)))
505: 
506: 
507:     def handle_extra_path (self):
508: 
509:         if self.extra_path is None:
510:             self.extra_path = self.distribution.extra_path
511: 
512:         if self.extra_path is not None:
513:             if type(self.extra_path) is StringType:
514:                 self.extra_path = string.split(self.extra_path, ',')
515: 
516:             if len(self.extra_path) == 1:
517:                 path_file = extra_dirs = self.extra_path[0]
518:             elif len(self.extra_path) == 2:
519:                 (path_file, extra_dirs) = self.extra_path
520:             else:
521:                 raise DistutilsOptionError, \
522:                       ("'extra_path' option must be a list, tuple, or "
523:                       "comma-separated string with 1 or 2 elements")
524: 
525:             # convert to local form in case Unix notation used (as it
526:             # should be in setup scripts)
527:             extra_dirs = convert_path(extra_dirs)
528: 
529:         else:
530:             path_file = None
531:             extra_dirs = ''
532: 
533:         # XXX should we warn if path_file and not extra_dirs? (in which
534:         # case the path file would be harmless but pointless)
535:         self.path_file = path_file
536:         self.extra_dirs = extra_dirs
537: 
538:     # handle_extra_path ()
539: 
540: 
541:     def change_roots (self, *names):
542:         for name in names:
543:             attr = "install_" + name
544:             setattr(self, attr, change_root(self.root, getattr(self, attr)))
545: 
546:     def create_home_path(self):
547:         '''Create directories under ~
548:         '''
549:         if not self.user:
550:             return
551:         home = convert_path(os.path.expanduser("~"))
552:         for name, path in self.config_vars.iteritems():
553:             if path.startswith(home) and not os.path.isdir(path):
554:                 self.debug_print("os.makedirs('%s', 0700)" % path)
555:                 os.makedirs(path, 0700)
556: 
557:     # -- Command execution methods -------------------------------------
558: 
559:     def run (self):
560: 
561:         # Obviously have to build before we can install
562:         if not self.skip_build:
563:             self.run_command('build')
564:             # If we built for any other platform, we can't install.
565:             build_plat = self.distribution.get_command_obj('build').plat_name
566:             # check warn_dir - it is a clue that the 'install' is happening
567:             # internally, and not to sys.path, so we don't check the platform
568:             # matches what we are running.
569:             if self.warn_dir and build_plat != get_platform():
570:                 raise DistutilsPlatformError("Can't install when "
571:                                              "cross-compiling")
572: 
573:         # Run all sub-commands (at least those that need to be run)
574:         for cmd_name in self.get_sub_commands():
575:             self.run_command(cmd_name)
576: 
577:         if self.path_file:
578:             self.create_path_file()
579: 
580:         # write list of installed files, if requested.
581:         if self.record:
582:             outputs = self.get_outputs()
583:             if self.root:               # strip any package prefix
584:                 root_len = len(self.root)
585:                 for counter in xrange(len(outputs)):
586:                     outputs[counter] = outputs[counter][root_len:]
587:             self.execute(write_file,
588:                          (self.record, outputs),
589:                          "writing list of installed files to '%s'" %
590:                          self.record)
591: 
592:         sys_path = map(os.path.normpath, sys.path)
593:         sys_path = map(os.path.normcase, sys_path)
594:         install_lib = os.path.normcase(os.path.normpath(self.install_lib))
595:         if (self.warn_dir and
596:             not (self.path_file and self.install_path_file) and
597:             install_lib not in sys_path):
598:             log.debug(("modules installed to '%s', which is not in "
599:                        "Python's module search path (sys.path) -- "
600:                        "you'll have to change the search path yourself"),
601:                        self.install_lib)
602: 
603:     # run ()
604: 
605:     def create_path_file (self):
606:         filename = os.path.join(self.install_libbase,
607:                                 self.path_file + ".pth")
608:         if self.install_path_file:
609:             self.execute(write_file,
610:                          (filename, [self.extra_dirs]),
611:                          "creating %s" % filename)
612:         else:
613:             self.warn("path file '%s' not created" % filename)
614: 
615: 
616:     # -- Reporting methods ---------------------------------------------
617: 
618:     def get_outputs (self):
619:         # Assemble the outputs of all the sub-commands.
620:         outputs = []
621:         for cmd_name in self.get_sub_commands():
622:             cmd = self.get_finalized_command(cmd_name)
623:             # Add the contents of cmd.get_outputs(), ensuring
624:             # that outputs doesn't contain duplicate entries
625:             for filename in cmd.get_outputs():
626:                 if filename not in outputs:
627:                     outputs.append(filename)
628: 
629:         if self.path_file and self.install_path_file:
630:             outputs.append(os.path.join(self.install_libbase,
631:                                         self.path_file + ".pth"))
632: 
633:         return outputs
634: 
635:     def get_inputs (self):
636:         # XXX gee, this looks familiar ;-(
637:         inputs = []
638:         for cmd_name in self.get_sub_commands():
639:             cmd = self.get_finalized_command(cmd_name)
640:             inputs.extend(cmd.get_inputs())
641: 
642:         return inputs
643: 
644: 
645:     # -- Predicates for sub-command list -------------------------------
646: 
647:     def has_lib (self):
648:         '''Return true if the current distribution has any Python
649:         modules to install.'''
650:         return (self.distribution.has_pure_modules() or
651:                 self.distribution.has_ext_modules())
652: 
653:     def has_headers (self):
654:         return self.distribution.has_headers()
655: 
656:     def has_scripts (self):
657:         return self.distribution.has_scripts()
658: 
659:     def has_data (self):
660:         return self.distribution.has_data_files()
661: 
662: 
663:     # 'sub_commands': a list of commands this command might have to run to
664:     # get its work done.  See cmd.py for more info.
665:     sub_commands = [('install_lib',     has_lib),
666:                     ('install_headers', has_headers),
667:                     ('install_scripts', has_scripts),
668:                     ('install_data',    has_data),
669:                     ('install_egg_info', lambda self:True),
670:                    ]
671: 
672: # class install
673: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_2306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', "distutils.command.install\n\nImplements the Distutils 'install' command.")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from distutils import log' statement (line 5)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils', None, module_type_store, ['log'], [log])


# Assigning a Str to a Name (line 9):

# Assigning a Str to a Name (line 9):
str_2307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), '__revision__', str_2307)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# Multiple import statement. import sys (1/3) (line 11)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'sys', sys, module_type_store)
# Multiple import statement. import os (2/3) (line 11)
import os

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'os', os, module_type_store)
# Multiple import statement. import string (3/3) (line 11)
import string

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'string', string, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from types import ' statement (line 12)
try:
    from types import *

except:
    pass
import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'types', None, module_type_store, ['*'], None)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from distutils.core import Command' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_2308 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.core')

if (type(import_2308) is not StypyTypeError):

    if (import_2308 != 'pyd_module'):
        __import__(import_2308)
        sys_modules_2309 = sys.modules[import_2308]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.core', sys_modules_2309.module_type_store, module_type_store, ['Command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_2309, sys_modules_2309.module_type_store, module_type_store)
    else:
        from distutils.core import Command

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.core', None, module_type_store, ['Command'], [Command])

else:
    # Assigning a type to the variable 'distutils.core' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.core', import_2308)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from distutils.debug import DEBUG' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_2310 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.debug')

if (type(import_2310) is not StypyTypeError):

    if (import_2310 != 'pyd_module'):
        __import__(import_2310)
        sys_modules_2311 = sys.modules[import_2310]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.debug', sys_modules_2311.module_type_store, module_type_store, ['DEBUG'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_2311, sys_modules_2311.module_type_store, module_type_store)
    else:
        from distutils.debug import DEBUG

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.debug', None, module_type_store, ['DEBUG'], [DEBUG])

else:
    # Assigning a type to the variable 'distutils.debug' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.debug', import_2310)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from distutils.sysconfig import get_config_vars' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_2312 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.sysconfig')

if (type(import_2312) is not StypyTypeError):

    if (import_2312 != 'pyd_module'):
        __import__(import_2312)
        sys_modules_2313 = sys.modules[import_2312]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.sysconfig', sys_modules_2313.module_type_store, module_type_store, ['get_config_vars'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_2313, sys_modules_2313.module_type_store, module_type_store)
    else:
        from distutils.sysconfig import get_config_vars

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.sysconfig', None, module_type_store, ['get_config_vars'], [get_config_vars])

else:
    # Assigning a type to the variable 'distutils.sysconfig' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.sysconfig', import_2312)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from distutils.errors import DistutilsPlatformError' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_2314 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.errors')

if (type(import_2314) is not StypyTypeError):

    if (import_2314 != 'pyd_module'):
        __import__(import_2314)
        sys_modules_2315 = sys.modules[import_2314]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.errors', sys_modules_2315.module_type_store, module_type_store, ['DistutilsPlatformError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_2315, sys_modules_2315.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsPlatformError

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.errors', None, module_type_store, ['DistutilsPlatformError'], [DistutilsPlatformError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.errors', import_2314)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from distutils.file_util import write_file' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_2316 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.file_util')

if (type(import_2316) is not StypyTypeError):

    if (import_2316 != 'pyd_module'):
        __import__(import_2316)
        sys_modules_2317 = sys.modules[import_2316]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.file_util', sys_modules_2317.module_type_store, module_type_store, ['write_file'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_2317, sys_modules_2317.module_type_store, module_type_store)
    else:
        from distutils.file_util import write_file

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.file_util', None, module_type_store, ['write_file'], [write_file])

else:
    # Assigning a type to the variable 'distutils.file_util' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.file_util', import_2316)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from distutils.util import convert_path, subst_vars, change_root' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_2318 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.util')

if (type(import_2318) is not StypyTypeError):

    if (import_2318 != 'pyd_module'):
        __import__(import_2318)
        sys_modules_2319 = sys.modules[import_2318]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.util', sys_modules_2319.module_type_store, module_type_store, ['convert_path', 'subst_vars', 'change_root'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_2319, sys_modules_2319.module_type_store, module_type_store)
    else:
        from distutils.util import convert_path, subst_vars, change_root

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.util', None, module_type_store, ['convert_path', 'subst_vars', 'change_root'], [convert_path, subst_vars, change_root])

else:
    # Assigning a type to the variable 'distutils.util' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.util', import_2318)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from distutils.util import get_platform' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_2320 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.util')

if (type(import_2320) is not StypyTypeError):

    if (import_2320 != 'pyd_module'):
        __import__(import_2320)
        sys_modules_2321 = sys.modules[import_2320]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.util', sys_modules_2321.module_type_store, module_type_store, ['get_platform'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_2321, sys_modules_2321.module_type_store, module_type_store)
    else:
        from distutils.util import get_platform

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.util', None, module_type_store, ['get_platform'], [get_platform])

else:
    # Assigning a type to the variable 'distutils.util' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.util', import_2320)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from distutils.errors import DistutilsOptionError' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_2322 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils.errors')

if (type(import_2322) is not StypyTypeError):

    if (import_2322 != 'pyd_module'):
        __import__(import_2322)
        sys_modules_2323 = sys.modules[import_2322]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils.errors', sys_modules_2323.module_type_store, module_type_store, ['DistutilsOptionError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_2323, sys_modules_2323.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsOptionError

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils.errors', None, module_type_store, ['DistutilsOptionError'], [DistutilsOptionError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils.errors', import_2322)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from site import USER_BASE' statement (line 21)
try:
    from site import USER_BASE

except:
    USER_BASE = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'site', None, module_type_store, ['USER_BASE'], [USER_BASE])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from site import USER_SITE' statement (line 22)
try:
    from site import USER_SITE

except:
    USER_SITE = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'site', None, module_type_store, ['USER_SITE'], [USER_SITE])



# Getting the type of 'sys' (line 25)
sys_2324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 3), 'sys')
# Obtaining the member 'version' of a type (line 25)
version_2325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 3), sys_2324, 'version')
str_2326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 17), 'str', '2.2')
# Applying the binary operator '<' (line 25)
result_lt_2327 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 3), '<', version_2325, str_2326)

# Testing the type of an if condition (line 25)
if_condition_2328 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 25, 0), result_lt_2327)
# Assigning a type to the variable 'if_condition_2328' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'if_condition_2328', if_condition_2328)
# SSA begins for if statement (line 25)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Dict to a Name (line 26):

# Assigning a Dict to a Name (line 26):

# Obtaining an instance of the builtin type 'dict' (line 26)
dict_2329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 21), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 26)
# Adding element type (key, value) (line 26)
str_2330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 8), 'str', 'purelib')
str_2331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 19), 'str', '$base')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 21), dict_2329, (str_2330, str_2331))
# Adding element type (key, value) (line 26)
str_2332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 8), 'str', 'platlib')
str_2333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 19), 'str', '$base')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 21), dict_2329, (str_2332, str_2333))
# Adding element type (key, value) (line 26)
str_2334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 8), 'str', 'headers')
str_2335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 19), 'str', '$base/Include/$dist_name')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 21), dict_2329, (str_2334, str_2335))
# Adding element type (key, value) (line 26)
str_2336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 8), 'str', 'scripts')
str_2337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 19), 'str', '$base/Scripts')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 21), dict_2329, (str_2336, str_2337))
# Adding element type (key, value) (line 26)
str_2338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 8), 'str', 'data')
str_2339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 19), 'str', '$base')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 21), dict_2329, (str_2338, str_2339))

# Assigning a type to the variable 'WINDOWS_SCHEME' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'WINDOWS_SCHEME', dict_2329)
# SSA branch for the else part of an if statement (line 25)
module_type_store.open_ssa_branch('else')

# Assigning a Dict to a Name (line 34):

# Assigning a Dict to a Name (line 34):

# Obtaining an instance of the builtin type 'dict' (line 34)
dict_2340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 21), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 34)
# Adding element type (key, value) (line 34)
str_2341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 8), 'str', 'purelib')
str_2342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 19), 'str', '$base/Lib/site-packages')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 21), dict_2340, (str_2341, str_2342))
# Adding element type (key, value) (line 34)
str_2343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 8), 'str', 'platlib')
str_2344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 19), 'str', '$base/Lib/site-packages')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 21), dict_2340, (str_2343, str_2344))
# Adding element type (key, value) (line 34)
str_2345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 8), 'str', 'headers')
str_2346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 19), 'str', '$base/Include/$dist_name')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 21), dict_2340, (str_2345, str_2346))
# Adding element type (key, value) (line 34)
str_2347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 8), 'str', 'scripts')
str_2348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 19), 'str', '$base/Scripts')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 21), dict_2340, (str_2347, str_2348))
# Adding element type (key, value) (line 34)
str_2349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 8), 'str', 'data')
str_2350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 19), 'str', '$base')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 21), dict_2340, (str_2349, str_2350))

# Assigning a type to the variable 'WINDOWS_SCHEME' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'WINDOWS_SCHEME', dict_2340)
# SSA join for if statement (line 25)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Dict to a Name (line 42):

# Assigning a Dict to a Name (line 42):

# Obtaining an instance of the builtin type 'dict' (line 42)
dict_2351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 42)
# Adding element type (key, value) (line 42)
str_2352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 4), 'str', 'unix_prefix')

# Obtaining an instance of the builtin type 'dict' (line 43)
dict_2353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 43)
# Adding element type (key, value) (line 43)
str_2354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 8), 'str', 'purelib')
str_2355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 19), 'str', '$base/lib/python$py_version_short/site-packages')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 19), dict_2353, (str_2354, str_2355))
# Adding element type (key, value) (line 43)
str_2356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 8), 'str', 'platlib')
str_2357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 19), 'str', '$platbase/lib/python$py_version_short/site-packages')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 19), dict_2353, (str_2356, str_2357))
# Adding element type (key, value) (line 43)
str_2358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 8), 'str', 'headers')
str_2359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 19), 'str', '$base/include/python$py_version_short/$dist_name')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 19), dict_2353, (str_2358, str_2359))
# Adding element type (key, value) (line 43)
str_2360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 8), 'str', 'scripts')
str_2361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 19), 'str', '$base/bin')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 19), dict_2353, (str_2360, str_2361))
# Adding element type (key, value) (line 43)
str_2362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 8), 'str', 'data')
str_2363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 19), 'str', '$base')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 19), dict_2353, (str_2362, str_2363))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 18), dict_2351, (str_2352, dict_2353))
# Adding element type (key, value) (line 42)
str_2364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 4), 'str', 'unix_home')

# Obtaining an instance of the builtin type 'dict' (line 50)
dict_2365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 17), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 50)
# Adding element type (key, value) (line 50)
str_2366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 8), 'str', 'purelib')
str_2367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 19), 'str', '$base/lib/python')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_2365, (str_2366, str_2367))
# Adding element type (key, value) (line 50)
str_2368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 8), 'str', 'platlib')
str_2369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 19), 'str', '$base/lib/python')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_2365, (str_2368, str_2369))
# Adding element type (key, value) (line 50)
str_2370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 8), 'str', 'headers')
str_2371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 19), 'str', '$base/include/python/$dist_name')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_2365, (str_2370, str_2371))
# Adding element type (key, value) (line 50)
str_2372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 8), 'str', 'scripts')
str_2373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 19), 'str', '$base/bin')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_2365, (str_2372, str_2373))
# Adding element type (key, value) (line 50)
str_2374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 8), 'str', 'data')
str_2375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 19), 'str', '$base')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_2365, (str_2374, str_2375))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 18), dict_2351, (str_2364, dict_2365))
# Adding element type (key, value) (line 42)
str_2376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 4), 'str', 'unix_user')

# Obtaining an instance of the builtin type 'dict' (line 57)
dict_2377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 17), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 57)
# Adding element type (key, value) (line 57)
str_2378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 8), 'str', 'purelib')
str_2379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 19), 'str', '$usersite')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 17), dict_2377, (str_2378, str_2379))
# Adding element type (key, value) (line 57)
str_2380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 8), 'str', 'platlib')
str_2381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 19), 'str', '$usersite')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 17), dict_2377, (str_2380, str_2381))
# Adding element type (key, value) (line 57)
str_2382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 8), 'str', 'headers')
str_2383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 19), 'str', '$userbase/include/python$py_version_short/$dist_name')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 17), dict_2377, (str_2382, str_2383))
# Adding element type (key, value) (line 57)
str_2384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 8), 'str', 'scripts')
str_2385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 19), 'str', '$userbase/bin')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 17), dict_2377, (str_2384, str_2385))
# Adding element type (key, value) (line 57)
str_2386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 8), 'str', 'data')
str_2387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 19), 'str', '$userbase')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 17), dict_2377, (str_2386, str_2387))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 18), dict_2351, (str_2376, dict_2377))
# Adding element type (key, value) (line 42)
str_2388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 4), 'str', 'nt')
# Getting the type of 'WINDOWS_SCHEME' (line 64)
WINDOWS_SCHEME_2389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 10), 'WINDOWS_SCHEME')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 18), dict_2351, (str_2388, WINDOWS_SCHEME_2389))
# Adding element type (key, value) (line 42)
str_2390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 4), 'str', 'nt_user')

# Obtaining an instance of the builtin type 'dict' (line 65)
dict_2391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 15), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 65)
# Adding element type (key, value) (line 65)
str_2392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 8), 'str', 'purelib')
str_2393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 19), 'str', '$usersite')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 15), dict_2391, (str_2392, str_2393))
# Adding element type (key, value) (line 65)
str_2394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 8), 'str', 'platlib')
str_2395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 19), 'str', '$usersite')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 15), dict_2391, (str_2394, str_2395))
# Adding element type (key, value) (line 65)
str_2396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 8), 'str', 'headers')
str_2397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 19), 'str', '$userbase/Python$py_version_nodot/Include/$dist_name')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 15), dict_2391, (str_2396, str_2397))
# Adding element type (key, value) (line 65)
str_2398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 8), 'str', 'scripts')
str_2399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 19), 'str', '$userbase/Scripts')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 15), dict_2391, (str_2398, str_2399))
# Adding element type (key, value) (line 65)
str_2400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 8), 'str', 'data')
str_2401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 19), 'str', '$userbase')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 15), dict_2391, (str_2400, str_2401))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 18), dict_2351, (str_2390, dict_2391))
# Adding element type (key, value) (line 42)
str_2402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 4), 'str', 'os2')

# Obtaining an instance of the builtin type 'dict' (line 72)
dict_2403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 11), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 72)
# Adding element type (key, value) (line 72)
str_2404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 8), 'str', 'purelib')
str_2405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 19), 'str', '$base/Lib/site-packages')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 11), dict_2403, (str_2404, str_2405))
# Adding element type (key, value) (line 72)
str_2406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 8), 'str', 'platlib')
str_2407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 19), 'str', '$base/Lib/site-packages')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 11), dict_2403, (str_2406, str_2407))
# Adding element type (key, value) (line 72)
str_2408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 8), 'str', 'headers')
str_2409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 19), 'str', '$base/Include/$dist_name')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 11), dict_2403, (str_2408, str_2409))
# Adding element type (key, value) (line 72)
str_2410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 8), 'str', 'scripts')
str_2411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 19), 'str', '$base/Scripts')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 11), dict_2403, (str_2410, str_2411))
# Adding element type (key, value) (line 72)
str_2412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 8), 'str', 'data')
str_2413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 19), 'str', '$base')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 11), dict_2403, (str_2412, str_2413))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 18), dict_2351, (str_2402, dict_2403))
# Adding element type (key, value) (line 42)
str_2414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 4), 'str', 'os2_home')

# Obtaining an instance of the builtin type 'dict' (line 79)
dict_2415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 16), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 79)
# Adding element type (key, value) (line 79)
str_2416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 8), 'str', 'purelib')
str_2417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 19), 'str', '$usersite')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 16), dict_2415, (str_2416, str_2417))
# Adding element type (key, value) (line 79)
str_2418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 8), 'str', 'platlib')
str_2419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 19), 'str', '$usersite')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 16), dict_2415, (str_2418, str_2419))
# Adding element type (key, value) (line 79)
str_2420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 8), 'str', 'headers')
str_2421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 19), 'str', '$userbase/include/python$py_version_short/$dist_name')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 16), dict_2415, (str_2420, str_2421))
# Adding element type (key, value) (line 79)
str_2422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 8), 'str', 'scripts')
str_2423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 19), 'str', '$userbase/bin')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 16), dict_2415, (str_2422, str_2423))
# Adding element type (key, value) (line 79)
str_2424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 8), 'str', 'data')
str_2425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 19), 'str', '$userbase')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 16), dict_2415, (str_2424, str_2425))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 18), dict_2351, (str_2414, dict_2415))

# Assigning a type to the variable 'INSTALL_SCHEMES' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'INSTALL_SCHEMES', dict_2351)

# Assigning a Tuple to a Name (line 91):

# Assigning a Tuple to a Name (line 91):

# Obtaining an instance of the builtin type 'tuple' (line 91)
tuple_2426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 91)
# Adding element type (line 91)
str_2427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 15), 'str', 'purelib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 15), tuple_2426, str_2427)
# Adding element type (line 91)
str_2428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 26), 'str', 'platlib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 15), tuple_2426, str_2428)
# Adding element type (line 91)
str_2429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 37), 'str', 'headers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 15), tuple_2426, str_2429)
# Adding element type (line 91)
str_2430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 48), 'str', 'scripts')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 15), tuple_2426, str_2430)
# Adding element type (line 91)
str_2431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 59), 'str', 'data')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 15), tuple_2426, str_2431)

# Assigning a type to the variable 'SCHEME_KEYS' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'SCHEME_KEYS', tuple_2426)
# Declaration of the 'install' class
# Getting the type of 'Command' (line 94)
Command_2432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 15), 'Command')

class install(Command_2432, ):
    
    # Assigning a Str to a Name (line 96):
    
    # Assigning a List to a Name (line 98):
    
    # Assigning a List to a Name (line 159):
    
    # Assigning a Dict to a Name (line 160):

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 163, 4, False)
        # Assigning a type to the variable 'self' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        install.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        install.initialize_options.__dict__.__setitem__('stypy_function_name', 'install.initialize_options')
        install.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        install.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        install.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        install.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        install.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install.initialize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 167):
        
        # Assigning a Name to a Attribute (line 167):
        # Getting the type of 'None' (line 167)
        None_2433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 22), 'None')
        # Getting the type of 'self' (line 167)
        self_2434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'self')
        # Setting the type of the member 'prefix' of a type (line 167)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), self_2434, 'prefix', None_2433)
        
        # Assigning a Name to a Attribute (line 168):
        
        # Assigning a Name to a Attribute (line 168):
        # Getting the type of 'None' (line 168)
        None_2435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 27), 'None')
        # Getting the type of 'self' (line 168)
        self_2436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'self')
        # Setting the type of the member 'exec_prefix' of a type (line 168)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), self_2436, 'exec_prefix', None_2435)
        
        # Assigning a Name to a Attribute (line 169):
        
        # Assigning a Name to a Attribute (line 169):
        # Getting the type of 'None' (line 169)
        None_2437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 20), 'None')
        # Getting the type of 'self' (line 169)
        self_2438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'self')
        # Setting the type of the member 'home' of a type (line 169)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), self_2438, 'home', None_2437)
        
        # Assigning a Num to a Attribute (line 170):
        
        # Assigning a Num to a Attribute (line 170):
        int_2439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 20), 'int')
        # Getting the type of 'self' (line 170)
        self_2440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'self')
        # Setting the type of the member 'user' of a type (line 170)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), self_2440, 'user', int_2439)
        
        # Assigning a Name to a Attribute (line 175):
        
        # Assigning a Name to a Attribute (line 175):
        # Getting the type of 'None' (line 175)
        None_2441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 28), 'None')
        # Getting the type of 'self' (line 175)
        self_2442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'self')
        # Setting the type of the member 'install_base' of a type (line 175)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), self_2442, 'install_base', None_2441)
        
        # Assigning a Name to a Attribute (line 176):
        
        # Assigning a Name to a Attribute (line 176):
        # Getting the type of 'None' (line 176)
        None_2443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 32), 'None')
        # Getting the type of 'self' (line 176)
        self_2444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'self')
        # Setting the type of the member 'install_platbase' of a type (line 176)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), self_2444, 'install_platbase', None_2443)
        
        # Assigning a Name to a Attribute (line 177):
        
        # Assigning a Name to a Attribute (line 177):
        # Getting the type of 'None' (line 177)
        None_2445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 20), 'None')
        # Getting the type of 'self' (line 177)
        self_2446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'self')
        # Setting the type of the member 'root' of a type (line 177)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), self_2446, 'root', None_2445)
        
        # Assigning a Name to a Attribute (line 183):
        
        # Assigning a Name to a Attribute (line 183):
        # Getting the type of 'None' (line 183)
        None_2447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 31), 'None')
        # Getting the type of 'self' (line 183)
        self_2448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'self')
        # Setting the type of the member 'install_purelib' of a type (line 183)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 8), self_2448, 'install_purelib', None_2447)
        
        # Assigning a Name to a Attribute (line 184):
        
        # Assigning a Name to a Attribute (line 184):
        # Getting the type of 'None' (line 184)
        None_2449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 31), 'None')
        # Getting the type of 'self' (line 184)
        self_2450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'self')
        # Setting the type of the member 'install_platlib' of a type (line 184)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), self_2450, 'install_platlib', None_2449)
        
        # Assigning a Name to a Attribute (line 185):
        
        # Assigning a Name to a Attribute (line 185):
        # Getting the type of 'None' (line 185)
        None_2451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 31), 'None')
        # Getting the type of 'self' (line 185)
        self_2452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'self')
        # Setting the type of the member 'install_headers' of a type (line 185)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 8), self_2452, 'install_headers', None_2451)
        
        # Assigning a Name to a Attribute (line 186):
        
        # Assigning a Name to a Attribute (line 186):
        # Getting the type of 'None' (line 186)
        None_2453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 27), 'None')
        # Getting the type of 'self' (line 186)
        self_2454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'self')
        # Setting the type of the member 'install_lib' of a type (line 186)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), self_2454, 'install_lib', None_2453)
        
        # Assigning a Name to a Attribute (line 187):
        
        # Assigning a Name to a Attribute (line 187):
        # Getting the type of 'None' (line 187)
        None_2455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 31), 'None')
        # Getting the type of 'self' (line 187)
        self_2456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'self')
        # Setting the type of the member 'install_scripts' of a type (line 187)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 8), self_2456, 'install_scripts', None_2455)
        
        # Assigning a Name to a Attribute (line 188):
        
        # Assigning a Name to a Attribute (line 188):
        # Getting the type of 'None' (line 188)
        None_2457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 28), 'None')
        # Getting the type of 'self' (line 188)
        self_2458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'self')
        # Setting the type of the member 'install_data' of a type (line 188)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 8), self_2458, 'install_data', None_2457)
        
        # Assigning a Name to a Attribute (line 189):
        
        # Assigning a Name to a Attribute (line 189):
        # Getting the type of 'USER_BASE' (line 189)
        USER_BASE_2459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 32), 'USER_BASE')
        # Getting the type of 'self' (line 189)
        self_2460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'self')
        # Setting the type of the member 'install_userbase' of a type (line 189)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 8), self_2460, 'install_userbase', USER_BASE_2459)
        
        # Assigning a Name to a Attribute (line 190):
        
        # Assigning a Name to a Attribute (line 190):
        # Getting the type of 'USER_SITE' (line 190)
        USER_SITE_2461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 32), 'USER_SITE')
        # Getting the type of 'self' (line 190)
        self_2462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'self')
        # Setting the type of the member 'install_usersite' of a type (line 190)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 8), self_2462, 'install_usersite', USER_SITE_2461)
        
        # Assigning a Name to a Attribute (line 192):
        
        # Assigning a Name to a Attribute (line 192):
        # Getting the type of 'None' (line 192)
        None_2463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 23), 'None')
        # Getting the type of 'self' (line 192)
        self_2464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'self')
        # Setting the type of the member 'compile' of a type (line 192)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), self_2464, 'compile', None_2463)
        
        # Assigning a Name to a Attribute (line 193):
        
        # Assigning a Name to a Attribute (line 193):
        # Getting the type of 'None' (line 193)
        None_2465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 24), 'None')
        # Getting the type of 'self' (line 193)
        self_2466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'self')
        # Setting the type of the member 'optimize' of a type (line 193)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 8), self_2466, 'optimize', None_2465)
        
        # Assigning a Name to a Attribute (line 203):
        
        # Assigning a Name to a Attribute (line 203):
        # Getting the type of 'None' (line 203)
        None_2467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 26), 'None')
        # Getting the type of 'self' (line 203)
        self_2468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'self')
        # Setting the type of the member 'extra_path' of a type (line 203)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 8), self_2468, 'extra_path', None_2467)
        
        # Assigning a Num to a Attribute (line 204):
        
        # Assigning a Num to a Attribute (line 204):
        int_2469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 33), 'int')
        # Getting the type of 'self' (line 204)
        self_2470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'self')
        # Setting the type of the member 'install_path_file' of a type (line 204)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), self_2470, 'install_path_file', int_2469)
        
        # Assigning a Num to a Attribute (line 212):
        
        # Assigning a Num to a Attribute (line 212):
        int_2471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 21), 'int')
        # Getting the type of 'self' (line 212)
        self_2472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'self')
        # Setting the type of the member 'force' of a type (line 212)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), self_2472, 'force', int_2471)
        
        # Assigning a Num to a Attribute (line 213):
        
        # Assigning a Num to a Attribute (line 213):
        int_2473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 26), 'int')
        # Getting the type of 'self' (line 213)
        self_2474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'self')
        # Setting the type of the member 'skip_build' of a type (line 213)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 8), self_2474, 'skip_build', int_2473)
        
        # Assigning a Num to a Attribute (line 214):
        
        # Assigning a Num to a Attribute (line 214):
        int_2475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 24), 'int')
        # Getting the type of 'self' (line 214)
        self_2476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'self')
        # Setting the type of the member 'warn_dir' of a type (line 214)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), self_2476, 'warn_dir', int_2475)
        
        # Assigning a Name to a Attribute (line 222):
        
        # Assigning a Name to a Attribute (line 222):
        # Getting the type of 'None' (line 222)
        None_2477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 26), 'None')
        # Getting the type of 'self' (line 222)
        self_2478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'self')
        # Setting the type of the member 'build_base' of a type (line 222)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), self_2478, 'build_base', None_2477)
        
        # Assigning a Name to a Attribute (line 223):
        
        # Assigning a Name to a Attribute (line 223):
        # Getting the type of 'None' (line 223)
        None_2479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 25), 'None')
        # Getting the type of 'self' (line 223)
        self_2480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'self')
        # Setting the type of the member 'build_lib' of a type (line 223)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), self_2480, 'build_lib', None_2479)
        
        # Assigning a Name to a Attribute (line 231):
        
        # Assigning a Name to a Attribute (line 231):
        # Getting the type of 'None' (line 231)
        None_2481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 22), 'None')
        # Getting the type of 'self' (line 231)
        self_2482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'self')
        # Setting the type of the member 'record' of a type (line 231)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 8), self_2482, 'record', None_2481)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 163)
        stypy_return_type_2483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2483)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_2483


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 240, 4, False)
        # Assigning a type to the variable 'self' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        install.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        install.finalize_options.__dict__.__setitem__('stypy_function_name', 'install.finalize_options')
        install.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        install.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        install.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        install.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        install.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 256)
        self_2484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 13), 'self')
        # Obtaining the member 'prefix' of a type (line 256)
        prefix_2485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 13), self_2484, 'prefix')
        # Getting the type of 'self' (line 256)
        self_2486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 28), 'self')
        # Obtaining the member 'exec_prefix' of a type (line 256)
        exec_prefix_2487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 28), self_2486, 'exec_prefix')
        # Applying the binary operator 'or' (line 256)
        result_or_keyword_2488 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 13), 'or', prefix_2485, exec_prefix_2487)
        # Getting the type of 'self' (line 256)
        self_2489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 48), 'self')
        # Obtaining the member 'home' of a type (line 256)
        home_2490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 48), self_2489, 'home')
        # Applying the binary operator 'or' (line 256)
        result_or_keyword_2491 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 13), 'or', result_or_keyword_2488, home_2490)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 257)
        self_2492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 13), 'self')
        # Obtaining the member 'install_base' of a type (line 257)
        install_base_2493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 13), self_2492, 'install_base')
        # Getting the type of 'self' (line 257)
        self_2494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 34), 'self')
        # Obtaining the member 'install_platbase' of a type (line 257)
        install_platbase_2495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 34), self_2494, 'install_platbase')
        # Applying the binary operator 'or' (line 257)
        result_or_keyword_2496 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 13), 'or', install_base_2493, install_platbase_2495)
        
        # Applying the binary operator 'and' (line 256)
        result_and_keyword_2497 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 12), 'and', result_or_keyword_2491, result_or_keyword_2496)
        
        # Testing the type of an if condition (line 256)
        if_condition_2498 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 256, 8), result_and_keyword_2497)
        # Assigning a type to the variable 'if_condition_2498' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'if_condition_2498', if_condition_2498)
        # SSA begins for if statement (line 256)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsOptionError' (line 258)
        DistutilsOptionError_2499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 18), 'DistutilsOptionError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 258, 12), DistutilsOptionError_2499, 'raise parameter', BaseException)
        # SSA join for if statement (line 256)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 262)
        self_2500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 11), 'self')
        # Obtaining the member 'home' of a type (line 262)
        home_2501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 11), self_2500, 'home')
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 262)
        self_2502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 26), 'self')
        # Obtaining the member 'prefix' of a type (line 262)
        prefix_2503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 26), self_2502, 'prefix')
        # Getting the type of 'self' (line 262)
        self_2504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 41), 'self')
        # Obtaining the member 'exec_prefix' of a type (line 262)
        exec_prefix_2505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 41), self_2504, 'exec_prefix')
        # Applying the binary operator 'or' (line 262)
        result_or_keyword_2506 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 26), 'or', prefix_2503, exec_prefix_2505)
        
        # Applying the binary operator 'and' (line 262)
        result_and_keyword_2507 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 11), 'and', home_2501, result_or_keyword_2506)
        
        # Testing the type of an if condition (line 262)
        if_condition_2508 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 262, 8), result_and_keyword_2507)
        # Assigning a type to the variable 'if_condition_2508' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'if_condition_2508', if_condition_2508)
        # SSA begins for if statement (line 262)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsOptionError' (line 263)
        DistutilsOptionError_2509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 18), 'DistutilsOptionError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 263, 12), DistutilsOptionError_2509, 'raise parameter', BaseException)
        # SSA join for if statement (line 262)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 266)
        self_2510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 11), 'self')
        # Obtaining the member 'user' of a type (line 266)
        user_2511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 11), self_2510, 'user')
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 266)
        self_2512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 26), 'self')
        # Obtaining the member 'prefix' of a type (line 266)
        prefix_2513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 26), self_2512, 'prefix')
        # Getting the type of 'self' (line 266)
        self_2514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 41), 'self')
        # Obtaining the member 'exec_prefix' of a type (line 266)
        exec_prefix_2515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 41), self_2514, 'exec_prefix')
        # Applying the binary operator 'or' (line 266)
        result_or_keyword_2516 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 26), 'or', prefix_2513, exec_prefix_2515)
        # Getting the type of 'self' (line 266)
        self_2517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 61), 'self')
        # Obtaining the member 'home' of a type (line 266)
        home_2518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 61), self_2517, 'home')
        # Applying the binary operator 'or' (line 266)
        result_or_keyword_2519 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 26), 'or', result_or_keyword_2516, home_2518)
        # Getting the type of 'self' (line 267)
        self_2520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 16), 'self')
        # Obtaining the member 'install_base' of a type (line 267)
        install_base_2521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 16), self_2520, 'install_base')
        # Applying the binary operator 'or' (line 266)
        result_or_keyword_2522 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 26), 'or', result_or_keyword_2519, install_base_2521)
        # Getting the type of 'self' (line 267)
        self_2523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 37), 'self')
        # Obtaining the member 'install_platbase' of a type (line 267)
        install_platbase_2524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 37), self_2523, 'install_platbase')
        # Applying the binary operator 'or' (line 266)
        result_or_keyword_2525 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 26), 'or', result_or_keyword_2522, install_platbase_2524)
        
        # Applying the binary operator 'and' (line 266)
        result_and_keyword_2526 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 11), 'and', user_2511, result_or_keyword_2525)
        
        # Testing the type of an if condition (line 266)
        if_condition_2527 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 266, 8), result_and_keyword_2526)
        # Assigning a type to the variable 'if_condition_2527' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'if_condition_2527', if_condition_2527)
        # SSA begins for if statement (line 266)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to DistutilsOptionError(...): (line 268)
        # Processing the call arguments (line 268)
        str_2529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 39), 'str', "can't combine user with prefix, exec_prefix/home, or install_(plat)base")
        # Processing the call keyword arguments (line 268)
        kwargs_2530 = {}
        # Getting the type of 'DistutilsOptionError' (line 268)
        DistutilsOptionError_2528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 18), 'DistutilsOptionError', False)
        # Calling DistutilsOptionError(args, kwargs) (line 268)
        DistutilsOptionError_call_result_2531 = invoke(stypy.reporting.localization.Localization(__file__, 268, 18), DistutilsOptionError_2528, *[str_2529], **kwargs_2530)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 268, 12), DistutilsOptionError_call_result_2531, 'raise parameter', BaseException)
        # SSA join for if statement (line 266)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'os' (line 272)
        os_2532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 11), 'os')
        # Obtaining the member 'name' of a type (line 272)
        name_2533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 11), os_2532, 'name')
        str_2534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 22), 'str', 'posix')
        # Applying the binary operator '!=' (line 272)
        result_ne_2535 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 11), '!=', name_2533, str_2534)
        
        # Testing the type of an if condition (line 272)
        if_condition_2536 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 272, 8), result_ne_2535)
        # Assigning a type to the variable 'if_condition_2536' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'if_condition_2536', if_condition_2536)
        # SSA begins for if statement (line 272)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 273)
        self_2537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 15), 'self')
        # Obtaining the member 'exec_prefix' of a type (line 273)
        exec_prefix_2538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 15), self_2537, 'exec_prefix')
        # Testing the type of an if condition (line 273)
        if_condition_2539 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 12), exec_prefix_2538)
        # Assigning a type to the variable 'if_condition_2539' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'if_condition_2539', if_condition_2539)
        # SSA begins for if statement (line 273)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 274)
        # Processing the call arguments (line 274)
        str_2542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 26), 'str', 'exec-prefix option ignored on this platform')
        # Processing the call keyword arguments (line 274)
        kwargs_2543 = {}
        # Getting the type of 'self' (line 274)
        self_2540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'self', False)
        # Obtaining the member 'warn' of a type (line 274)
        warn_2541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 16), self_2540, 'warn')
        # Calling warn(args, kwargs) (line 274)
        warn_call_result_2544 = invoke(stypy.reporting.localization.Localization(__file__, 274, 16), warn_2541, *[str_2542], **kwargs_2543)
        
        
        # Assigning a Name to a Attribute (line 275):
        
        # Assigning a Name to a Attribute (line 275):
        # Getting the type of 'None' (line 275)
        None_2545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 35), 'None')
        # Getting the type of 'self' (line 275)
        self_2546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'self')
        # Setting the type of the member 'exec_prefix' of a type (line 275)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 16), self_2546, 'exec_prefix', None_2545)
        # SSA join for if statement (line 273)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 272)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to dump_dirs(...): (line 285)
        # Processing the call arguments (line 285)
        str_2549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 23), 'str', 'pre-finalize_{unix,other}')
        # Processing the call keyword arguments (line 285)
        kwargs_2550 = {}
        # Getting the type of 'self' (line 285)
        self_2547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'self', False)
        # Obtaining the member 'dump_dirs' of a type (line 285)
        dump_dirs_2548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 8), self_2547, 'dump_dirs')
        # Calling dump_dirs(args, kwargs) (line 285)
        dump_dirs_call_result_2551 = invoke(stypy.reporting.localization.Localization(__file__, 285, 8), dump_dirs_2548, *[str_2549], **kwargs_2550)
        
        
        
        # Getting the type of 'os' (line 287)
        os_2552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 11), 'os')
        # Obtaining the member 'name' of a type (line 287)
        name_2553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 11), os_2552, 'name')
        str_2554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 22), 'str', 'posix')
        # Applying the binary operator '==' (line 287)
        result_eq_2555 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 11), '==', name_2553, str_2554)
        
        # Testing the type of an if condition (line 287)
        if_condition_2556 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 287, 8), result_eq_2555)
        # Assigning a type to the variable 'if_condition_2556' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'if_condition_2556', if_condition_2556)
        # SSA begins for if statement (line 287)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to finalize_unix(...): (line 288)
        # Processing the call keyword arguments (line 288)
        kwargs_2559 = {}
        # Getting the type of 'self' (line 288)
        self_2557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'self', False)
        # Obtaining the member 'finalize_unix' of a type (line 288)
        finalize_unix_2558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 12), self_2557, 'finalize_unix')
        # Calling finalize_unix(args, kwargs) (line 288)
        finalize_unix_call_result_2560 = invoke(stypy.reporting.localization.Localization(__file__, 288, 12), finalize_unix_2558, *[], **kwargs_2559)
        
        # SSA branch for the else part of an if statement (line 287)
        module_type_store.open_ssa_branch('else')
        
        # Call to finalize_other(...): (line 290)
        # Processing the call keyword arguments (line 290)
        kwargs_2563 = {}
        # Getting the type of 'self' (line 290)
        self_2561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'self', False)
        # Obtaining the member 'finalize_other' of a type (line 290)
        finalize_other_2562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 12), self_2561, 'finalize_other')
        # Calling finalize_other(args, kwargs) (line 290)
        finalize_other_call_result_2564 = invoke(stypy.reporting.localization.Localization(__file__, 290, 12), finalize_other_2562, *[], **kwargs_2563)
        
        # SSA join for if statement (line 287)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to dump_dirs(...): (line 292)
        # Processing the call arguments (line 292)
        str_2567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 23), 'str', 'post-finalize_{unix,other}()')
        # Processing the call keyword arguments (line 292)
        kwargs_2568 = {}
        # Getting the type of 'self' (line 292)
        self_2565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'self', False)
        # Obtaining the member 'dump_dirs' of a type (line 292)
        dump_dirs_2566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), self_2565, 'dump_dirs')
        # Calling dump_dirs(args, kwargs) (line 292)
        dump_dirs_call_result_2569 = invoke(stypy.reporting.localization.Localization(__file__, 292, 8), dump_dirs_2566, *[str_2567], **kwargs_2568)
        
        
        # Assigning a Subscript to a Name (line 299):
        
        # Assigning a Subscript to a Name (line 299):
        
        # Obtaining the type of the subscript
        int_2570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 49), 'int')
        
        # Call to split(...): (line 299)
        # Processing the call arguments (line 299)
        # Getting the type of 'sys' (line 299)
        sys_2573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 35), 'sys', False)
        # Obtaining the member 'version' of a type (line 299)
        version_2574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 35), sys_2573, 'version')
        # Processing the call keyword arguments (line 299)
        kwargs_2575 = {}
        # Getting the type of 'string' (line 299)
        string_2571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 22), 'string', False)
        # Obtaining the member 'split' of a type (line 299)
        split_2572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 22), string_2571, 'split')
        # Calling split(args, kwargs) (line 299)
        split_call_result_2576 = invoke(stypy.reporting.localization.Localization(__file__, 299, 22), split_2572, *[version_2574], **kwargs_2575)
        
        # Obtaining the member '__getitem__' of a type (line 299)
        getitem___2577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 22), split_call_result_2576, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 299)
        subscript_call_result_2578 = invoke(stypy.reporting.localization.Localization(__file__, 299, 22), getitem___2577, int_2570)
        
        # Assigning a type to the variable 'py_version' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'py_version', subscript_call_result_2578)
        
        # Assigning a Call to a Tuple (line 300):
        
        # Assigning a Subscript to a Name (line 300):
        
        # Obtaining the type of the subscript
        int_2579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 8), 'int')
        
        # Call to get_config_vars(...): (line 300)
        # Processing the call arguments (line 300)
        str_2581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 48), 'str', 'prefix')
        str_2582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 58), 'str', 'exec_prefix')
        # Processing the call keyword arguments (line 300)
        kwargs_2583 = {}
        # Getting the type of 'get_config_vars' (line 300)
        get_config_vars_2580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 32), 'get_config_vars', False)
        # Calling get_config_vars(args, kwargs) (line 300)
        get_config_vars_call_result_2584 = invoke(stypy.reporting.localization.Localization(__file__, 300, 32), get_config_vars_2580, *[str_2581, str_2582], **kwargs_2583)
        
        # Obtaining the member '__getitem__' of a type (line 300)
        getitem___2585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 8), get_config_vars_call_result_2584, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 300)
        subscript_call_result_2586 = invoke(stypy.reporting.localization.Localization(__file__, 300, 8), getitem___2585, int_2579)
        
        # Assigning a type to the variable 'tuple_var_assignment_2302' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'tuple_var_assignment_2302', subscript_call_result_2586)
        
        # Assigning a Subscript to a Name (line 300):
        
        # Obtaining the type of the subscript
        int_2587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 8), 'int')
        
        # Call to get_config_vars(...): (line 300)
        # Processing the call arguments (line 300)
        str_2589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 48), 'str', 'prefix')
        str_2590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 58), 'str', 'exec_prefix')
        # Processing the call keyword arguments (line 300)
        kwargs_2591 = {}
        # Getting the type of 'get_config_vars' (line 300)
        get_config_vars_2588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 32), 'get_config_vars', False)
        # Calling get_config_vars(args, kwargs) (line 300)
        get_config_vars_call_result_2592 = invoke(stypy.reporting.localization.Localization(__file__, 300, 32), get_config_vars_2588, *[str_2589, str_2590], **kwargs_2591)
        
        # Obtaining the member '__getitem__' of a type (line 300)
        getitem___2593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 8), get_config_vars_call_result_2592, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 300)
        subscript_call_result_2594 = invoke(stypy.reporting.localization.Localization(__file__, 300, 8), getitem___2593, int_2587)
        
        # Assigning a type to the variable 'tuple_var_assignment_2303' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'tuple_var_assignment_2303', subscript_call_result_2594)
        
        # Assigning a Name to a Name (line 300):
        # Getting the type of 'tuple_var_assignment_2302' (line 300)
        tuple_var_assignment_2302_2595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'tuple_var_assignment_2302')
        # Assigning a type to the variable 'prefix' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 9), 'prefix', tuple_var_assignment_2302_2595)
        
        # Assigning a Name to a Name (line 300):
        # Getting the type of 'tuple_var_assignment_2303' (line 300)
        tuple_var_assignment_2303_2596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'tuple_var_assignment_2303')
        # Assigning a type to the variable 'exec_prefix' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 17), 'exec_prefix', tuple_var_assignment_2303_2596)
        
        # Assigning a Dict to a Attribute (line 301):
        
        # Assigning a Dict to a Attribute (line 301):
        
        # Obtaining an instance of the builtin type 'dict' (line 301)
        dict_2597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 27), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 301)
        # Adding element type (key, value) (line 301)
        str_2598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 28), 'str', 'dist_name')
        
        # Call to get_name(...): (line 301)
        # Processing the call keyword arguments (line 301)
        kwargs_2602 = {}
        # Getting the type of 'self' (line 301)
        self_2599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 41), 'self', False)
        # Obtaining the member 'distribution' of a type (line 301)
        distribution_2600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 41), self_2599, 'distribution')
        # Obtaining the member 'get_name' of a type (line 301)
        get_name_2601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 41), distribution_2600, 'get_name')
        # Calling get_name(args, kwargs) (line 301)
        get_name_call_result_2603 = invoke(stypy.reporting.localization.Localization(__file__, 301, 41), get_name_2601, *[], **kwargs_2602)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 27), dict_2597, (str_2598, get_name_call_result_2603))
        # Adding element type (key, value) (line 301)
        str_2604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 28), 'str', 'dist_version')
        
        # Call to get_version(...): (line 302)
        # Processing the call keyword arguments (line 302)
        kwargs_2608 = {}
        # Getting the type of 'self' (line 302)
        self_2605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 44), 'self', False)
        # Obtaining the member 'distribution' of a type (line 302)
        distribution_2606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 44), self_2605, 'distribution')
        # Obtaining the member 'get_version' of a type (line 302)
        get_version_2607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 44), distribution_2606, 'get_version')
        # Calling get_version(args, kwargs) (line 302)
        get_version_call_result_2609 = invoke(stypy.reporting.localization.Localization(__file__, 302, 44), get_version_2607, *[], **kwargs_2608)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 27), dict_2597, (str_2604, get_version_call_result_2609))
        # Adding element type (key, value) (line 301)
        str_2610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 28), 'str', 'dist_fullname')
        
        # Call to get_fullname(...): (line 303)
        # Processing the call keyword arguments (line 303)
        kwargs_2614 = {}
        # Getting the type of 'self' (line 303)
        self_2611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 45), 'self', False)
        # Obtaining the member 'distribution' of a type (line 303)
        distribution_2612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 45), self_2611, 'distribution')
        # Obtaining the member 'get_fullname' of a type (line 303)
        get_fullname_2613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 45), distribution_2612, 'get_fullname')
        # Calling get_fullname(args, kwargs) (line 303)
        get_fullname_call_result_2615 = invoke(stypy.reporting.localization.Localization(__file__, 303, 45), get_fullname_2613, *[], **kwargs_2614)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 27), dict_2597, (str_2610, get_fullname_call_result_2615))
        # Adding element type (key, value) (line 301)
        str_2616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 28), 'str', 'py_version')
        # Getting the type of 'py_version' (line 304)
        py_version_2617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 42), 'py_version')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 27), dict_2597, (str_2616, py_version_2617))
        # Adding element type (key, value) (line 301)
        str_2618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 28), 'str', 'py_version_short')
        
        # Obtaining the type of the subscript
        int_2619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 59), 'int')
        int_2620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 61), 'int')
        slice_2621 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 305, 48), int_2619, int_2620, None)
        # Getting the type of 'py_version' (line 305)
        py_version_2622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 48), 'py_version')
        # Obtaining the member '__getitem__' of a type (line 305)
        getitem___2623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 48), py_version_2622, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 305)
        subscript_call_result_2624 = invoke(stypy.reporting.localization.Localization(__file__, 305, 48), getitem___2623, slice_2621)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 27), dict_2597, (str_2618, subscript_call_result_2624))
        # Adding element type (key, value) (line 301)
        str_2625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 28), 'str', 'py_version_nodot')
        
        # Obtaining the type of the subscript
        int_2626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 59), 'int')
        # Getting the type of 'py_version' (line 306)
        py_version_2627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 48), 'py_version')
        # Obtaining the member '__getitem__' of a type (line 306)
        getitem___2628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 48), py_version_2627, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 306)
        subscript_call_result_2629 = invoke(stypy.reporting.localization.Localization(__file__, 306, 48), getitem___2628, int_2626)
        
        
        # Obtaining the type of the subscript
        int_2630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 75), 'int')
        # Getting the type of 'py_version' (line 306)
        py_version_2631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 64), 'py_version')
        # Obtaining the member '__getitem__' of a type (line 306)
        getitem___2632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 64), py_version_2631, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 306)
        subscript_call_result_2633 = invoke(stypy.reporting.localization.Localization(__file__, 306, 64), getitem___2632, int_2630)
        
        # Applying the binary operator '+' (line 306)
        result_add_2634 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 48), '+', subscript_call_result_2629, subscript_call_result_2633)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 27), dict_2597, (str_2625, result_add_2634))
        # Adding element type (key, value) (line 301)
        str_2635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 28), 'str', 'sys_prefix')
        # Getting the type of 'prefix' (line 307)
        prefix_2636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 42), 'prefix')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 27), dict_2597, (str_2635, prefix_2636))
        # Adding element type (key, value) (line 301)
        str_2637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 28), 'str', 'prefix')
        # Getting the type of 'prefix' (line 308)
        prefix_2638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 38), 'prefix')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 27), dict_2597, (str_2637, prefix_2638))
        # Adding element type (key, value) (line 301)
        str_2639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 28), 'str', 'sys_exec_prefix')
        # Getting the type of 'exec_prefix' (line 309)
        exec_prefix_2640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 47), 'exec_prefix')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 27), dict_2597, (str_2639, exec_prefix_2640))
        # Adding element type (key, value) (line 301)
        str_2641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 28), 'str', 'exec_prefix')
        # Getting the type of 'exec_prefix' (line 310)
        exec_prefix_2642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 43), 'exec_prefix')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 27), dict_2597, (str_2641, exec_prefix_2642))
        # Adding element type (key, value) (line 301)
        str_2643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 28), 'str', 'userbase')
        # Getting the type of 'self' (line 311)
        self_2644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 40), 'self')
        # Obtaining the member 'install_userbase' of a type (line 311)
        install_userbase_2645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 40), self_2644, 'install_userbase')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 27), dict_2597, (str_2643, install_userbase_2645))
        # Adding element type (key, value) (line 301)
        str_2646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 28), 'str', 'usersite')
        # Getting the type of 'self' (line 312)
        self_2647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 40), 'self')
        # Obtaining the member 'install_usersite' of a type (line 312)
        install_usersite_2648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 40), self_2647, 'install_usersite')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 27), dict_2597, (str_2646, install_usersite_2648))
        
        # Getting the type of 'self' (line 301)
        self_2649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'self')
        # Setting the type of the member 'config_vars' of a type (line 301)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 8), self_2649, 'config_vars', dict_2597)
        
        # Call to expand_basedirs(...): (line 314)
        # Processing the call keyword arguments (line 314)
        kwargs_2652 = {}
        # Getting the type of 'self' (line 314)
        self_2650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'self', False)
        # Obtaining the member 'expand_basedirs' of a type (line 314)
        expand_basedirs_2651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 8), self_2650, 'expand_basedirs')
        # Calling expand_basedirs(args, kwargs) (line 314)
        expand_basedirs_call_result_2653 = invoke(stypy.reporting.localization.Localization(__file__, 314, 8), expand_basedirs_2651, *[], **kwargs_2652)
        
        
        # Call to dump_dirs(...): (line 316)
        # Processing the call arguments (line 316)
        str_2656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 23), 'str', 'post-expand_basedirs()')
        # Processing the call keyword arguments (line 316)
        kwargs_2657 = {}
        # Getting the type of 'self' (line 316)
        self_2654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'self', False)
        # Obtaining the member 'dump_dirs' of a type (line 316)
        dump_dirs_2655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 8), self_2654, 'dump_dirs')
        # Calling dump_dirs(args, kwargs) (line 316)
        dump_dirs_call_result_2658 = invoke(stypy.reporting.localization.Localization(__file__, 316, 8), dump_dirs_2655, *[str_2656], **kwargs_2657)
        
        
        # Assigning a Attribute to a Subscript (line 320):
        
        # Assigning a Attribute to a Subscript (line 320):
        # Getting the type of 'self' (line 320)
        self_2659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 35), 'self')
        # Obtaining the member 'install_base' of a type (line 320)
        install_base_2660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 35), self_2659, 'install_base')
        # Getting the type of 'self' (line 320)
        self_2661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'self')
        # Obtaining the member 'config_vars' of a type (line 320)
        config_vars_2662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 8), self_2661, 'config_vars')
        str_2663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 25), 'str', 'base')
        # Storing an element on a container (line 320)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 8), config_vars_2662, (str_2663, install_base_2660))
        
        # Assigning a Attribute to a Subscript (line 321):
        
        # Assigning a Attribute to a Subscript (line 321):
        # Getting the type of 'self' (line 321)
        self_2664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 39), 'self')
        # Obtaining the member 'install_platbase' of a type (line 321)
        install_platbase_2665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 39), self_2664, 'install_platbase')
        # Getting the type of 'self' (line 321)
        self_2666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'self')
        # Obtaining the member 'config_vars' of a type (line 321)
        config_vars_2667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 8), self_2666, 'config_vars')
        str_2668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 25), 'str', 'platbase')
        # Storing an element on a container (line 321)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 8), config_vars_2667, (str_2668, install_platbase_2665))
        
        # Getting the type of 'DEBUG' (line 323)
        DEBUG_2669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 11), 'DEBUG')
        # Testing the type of an if condition (line 323)
        if_condition_2670 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 323, 8), DEBUG_2669)
        # Assigning a type to the variable 'if_condition_2670' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'if_condition_2670', if_condition_2670)
        # SSA begins for if statement (line 323)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 324, 12))
        
        # 'from pprint import pprint' statement (line 324)
        try:
            from pprint import pprint

        except:
            pprint = UndefinedType
        import_from_module(stypy.reporting.localization.Localization(__file__, 324, 12), 'pprint', None, module_type_store, ['pprint'], [pprint])
        
        str_2671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 18), 'str', 'config vars:')
        
        # Call to pprint(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'self' (line 326)
        self_2673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 19), 'self', False)
        # Obtaining the member 'config_vars' of a type (line 326)
        config_vars_2674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 19), self_2673, 'config_vars')
        # Processing the call keyword arguments (line 326)
        kwargs_2675 = {}
        # Getting the type of 'pprint' (line 326)
        pprint_2672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'pprint', False)
        # Calling pprint(args, kwargs) (line 326)
        pprint_call_result_2676 = invoke(stypy.reporting.localization.Localization(__file__, 326, 12), pprint_2672, *[config_vars_2674], **kwargs_2675)
        
        # SSA join for if statement (line 323)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to expand_dirs(...): (line 330)
        # Processing the call keyword arguments (line 330)
        kwargs_2679 = {}
        # Getting the type of 'self' (line 330)
        self_2677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'self', False)
        # Obtaining the member 'expand_dirs' of a type (line 330)
        expand_dirs_2678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 8), self_2677, 'expand_dirs')
        # Calling expand_dirs(args, kwargs) (line 330)
        expand_dirs_call_result_2680 = invoke(stypy.reporting.localization.Localization(__file__, 330, 8), expand_dirs_2678, *[], **kwargs_2679)
        
        
        # Call to dump_dirs(...): (line 332)
        # Processing the call arguments (line 332)
        str_2683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 23), 'str', 'post-expand_dirs()')
        # Processing the call keyword arguments (line 332)
        kwargs_2684 = {}
        # Getting the type of 'self' (line 332)
        self_2681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'self', False)
        # Obtaining the member 'dump_dirs' of a type (line 332)
        dump_dirs_2682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 8), self_2681, 'dump_dirs')
        # Calling dump_dirs(args, kwargs) (line 332)
        dump_dirs_call_result_2685 = invoke(stypy.reporting.localization.Localization(__file__, 332, 8), dump_dirs_2682, *[str_2683], **kwargs_2684)
        
        
        # Getting the type of 'self' (line 335)
        self_2686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 11), 'self')
        # Obtaining the member 'user' of a type (line 335)
        user_2687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 11), self_2686, 'user')
        # Testing the type of an if condition (line 335)
        if_condition_2688 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 335, 8), user_2687)
        # Assigning a type to the variable 'if_condition_2688' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'if_condition_2688', if_condition_2688)
        # SSA begins for if statement (line 335)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to create_home_path(...): (line 336)
        # Processing the call keyword arguments (line 336)
        kwargs_2691 = {}
        # Getting the type of 'self' (line 336)
        self_2689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'self', False)
        # Obtaining the member 'create_home_path' of a type (line 336)
        create_home_path_2690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 12), self_2689, 'create_home_path')
        # Calling create_home_path(args, kwargs) (line 336)
        create_home_path_call_result_2692 = invoke(stypy.reporting.localization.Localization(__file__, 336, 12), create_home_path_2690, *[], **kwargs_2691)
        
        # SSA join for if statement (line 335)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 342)
        # Getting the type of 'self' (line 342)
        self_2693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 11), 'self')
        # Obtaining the member 'install_lib' of a type (line 342)
        install_lib_2694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 11), self_2693, 'install_lib')
        # Getting the type of 'None' (line 342)
        None_2695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 31), 'None')
        
        (may_be_2696, more_types_in_union_2697) = may_be_none(install_lib_2694, None_2695)

        if may_be_2696:

            if more_types_in_union_2697:
                # Runtime conditional SSA (line 342)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Getting the type of 'self' (line 343)
            self_2698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 15), 'self')
            # Obtaining the member 'distribution' of a type (line 343)
            distribution_2699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 15), self_2698, 'distribution')
            # Obtaining the member 'ext_modules' of a type (line 343)
            ext_modules_2700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 15), distribution_2699, 'ext_modules')
            # Testing the type of an if condition (line 343)
            if_condition_2701 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 343, 12), ext_modules_2700)
            # Assigning a type to the variable 'if_condition_2701' (line 343)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'if_condition_2701', if_condition_2701)
            # SSA begins for if statement (line 343)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Attribute (line 344):
            
            # Assigning a Attribute to a Attribute (line 344):
            # Getting the type of 'self' (line 344)
            self_2702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 35), 'self')
            # Obtaining the member 'install_platlib' of a type (line 344)
            install_platlib_2703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 35), self_2702, 'install_platlib')
            # Getting the type of 'self' (line 344)
            self_2704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 16), 'self')
            # Setting the type of the member 'install_lib' of a type (line 344)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 16), self_2704, 'install_lib', install_platlib_2703)
            # SSA branch for the else part of an if statement (line 343)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Attribute to a Attribute (line 346):
            
            # Assigning a Attribute to a Attribute (line 346):
            # Getting the type of 'self' (line 346)
            self_2705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 35), 'self')
            # Obtaining the member 'install_purelib' of a type (line 346)
            install_purelib_2706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 35), self_2705, 'install_purelib')
            # Getting the type of 'self' (line 346)
            self_2707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 16), 'self')
            # Setting the type of the member 'install_lib' of a type (line 346)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 16), self_2707, 'install_lib', install_purelib_2706)
            # SSA join for if statement (line 343)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_2697:
                # SSA join for if statement (line 342)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to convert_paths(...): (line 351)
        # Processing the call arguments (line 351)
        str_2710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 27), 'str', 'lib')
        str_2711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 34), 'str', 'purelib')
        str_2712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 45), 'str', 'platlib')
        str_2713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 27), 'str', 'scripts')
        str_2714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 38), 'str', 'data')
        str_2715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 46), 'str', 'headers')
        str_2716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 27), 'str', 'userbase')
        str_2717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 39), 'str', 'usersite')
        # Processing the call keyword arguments (line 351)
        kwargs_2718 = {}
        # Getting the type of 'self' (line 351)
        self_2708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'self', False)
        # Obtaining the member 'convert_paths' of a type (line 351)
        convert_paths_2709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 8), self_2708, 'convert_paths')
        # Calling convert_paths(args, kwargs) (line 351)
        convert_paths_call_result_2719 = invoke(stypy.reporting.localization.Localization(__file__, 351, 8), convert_paths_2709, *[str_2710, str_2711, str_2712, str_2713, str_2714, str_2715, str_2716, str_2717], **kwargs_2718)
        
        
        # Call to handle_extra_path(...): (line 359)
        # Processing the call keyword arguments (line 359)
        kwargs_2722 = {}
        # Getting the type of 'self' (line 359)
        self_2720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'self', False)
        # Obtaining the member 'handle_extra_path' of a type (line 359)
        handle_extra_path_2721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 8), self_2720, 'handle_extra_path')
        # Calling handle_extra_path(args, kwargs) (line 359)
        handle_extra_path_call_result_2723 = invoke(stypy.reporting.localization.Localization(__file__, 359, 8), handle_extra_path_2721, *[], **kwargs_2722)
        
        
        # Assigning a Attribute to a Attribute (line 360):
        
        # Assigning a Attribute to a Attribute (line 360):
        # Getting the type of 'self' (line 360)
        self_2724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 31), 'self')
        # Obtaining the member 'install_lib' of a type (line 360)
        install_lib_2725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 31), self_2724, 'install_lib')
        # Getting the type of 'self' (line 360)
        self_2726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'self')
        # Setting the type of the member 'install_libbase' of a type (line 360)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 8), self_2726, 'install_libbase', install_lib_2725)
        
        # Assigning a Call to a Attribute (line 361):
        
        # Assigning a Call to a Attribute (line 361):
        
        # Call to join(...): (line 361)
        # Processing the call arguments (line 361)
        # Getting the type of 'self' (line 361)
        self_2730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 40), 'self', False)
        # Obtaining the member 'install_lib' of a type (line 361)
        install_lib_2731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 40), self_2730, 'install_lib')
        # Getting the type of 'self' (line 361)
        self_2732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 58), 'self', False)
        # Obtaining the member 'extra_dirs' of a type (line 361)
        extra_dirs_2733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 58), self_2732, 'extra_dirs')
        # Processing the call keyword arguments (line 361)
        kwargs_2734 = {}
        # Getting the type of 'os' (line 361)
        os_2727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 361)
        path_2728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 27), os_2727, 'path')
        # Obtaining the member 'join' of a type (line 361)
        join_2729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 27), path_2728, 'join')
        # Calling join(args, kwargs) (line 361)
        join_call_result_2735 = invoke(stypy.reporting.localization.Localization(__file__, 361, 27), join_2729, *[install_lib_2731, extra_dirs_2733], **kwargs_2734)
        
        # Getting the type of 'self' (line 361)
        self_2736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'self')
        # Setting the type of the member 'install_lib' of a type (line 361)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 8), self_2736, 'install_lib', join_call_result_2735)
        
        
        # Getting the type of 'self' (line 365)
        self_2737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 11), 'self')
        # Obtaining the member 'root' of a type (line 365)
        root_2738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 11), self_2737, 'root')
        # Getting the type of 'None' (line 365)
        None_2739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 28), 'None')
        # Applying the binary operator 'isnot' (line 365)
        result_is_not_2740 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 11), 'isnot', root_2738, None_2739)
        
        # Testing the type of an if condition (line 365)
        if_condition_2741 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 365, 8), result_is_not_2740)
        # Assigning a type to the variable 'if_condition_2741' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'if_condition_2741', if_condition_2741)
        # SSA begins for if statement (line 365)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to change_roots(...): (line 366)
        # Processing the call arguments (line 366)
        str_2744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 30), 'str', 'libbase')
        str_2745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 41), 'str', 'lib')
        str_2746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 48), 'str', 'purelib')
        str_2747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 59), 'str', 'platlib')
        str_2748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 30), 'str', 'scripts')
        str_2749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 41), 'str', 'data')
        str_2750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 49), 'str', 'headers')
        # Processing the call keyword arguments (line 366)
        kwargs_2751 = {}
        # Getting the type of 'self' (line 366)
        self_2742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'self', False)
        # Obtaining the member 'change_roots' of a type (line 366)
        change_roots_2743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 12), self_2742, 'change_roots')
        # Calling change_roots(args, kwargs) (line 366)
        change_roots_call_result_2752 = invoke(stypy.reporting.localization.Localization(__file__, 366, 12), change_roots_2743, *[str_2744, str_2745, str_2746, str_2747, str_2748, str_2749, str_2750], **kwargs_2751)
        
        # SSA join for if statement (line 365)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to dump_dirs(...): (line 369)
        # Processing the call arguments (line 369)
        str_2755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 23), 'str', 'after prepending root')
        # Processing the call keyword arguments (line 369)
        kwargs_2756 = {}
        # Getting the type of 'self' (line 369)
        self_2753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'self', False)
        # Obtaining the member 'dump_dirs' of a type (line 369)
        dump_dirs_2754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 8), self_2753, 'dump_dirs')
        # Calling dump_dirs(args, kwargs) (line 369)
        dump_dirs_call_result_2757 = invoke(stypy.reporting.localization.Localization(__file__, 369, 8), dump_dirs_2754, *[str_2755], **kwargs_2756)
        
        
        # Call to set_undefined_options(...): (line 372)
        # Processing the call arguments (line 372)
        str_2760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 35), 'str', 'build')
        
        # Obtaining an instance of the builtin type 'tuple' (line 373)
        tuple_2761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 373)
        # Adding element type (line 373)
        str_2762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 36), 'str', 'build_base')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 36), tuple_2761, str_2762)
        # Adding element type (line 373)
        str_2763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 50), 'str', 'build_base')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 36), tuple_2761, str_2763)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 374)
        tuple_2764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 374)
        # Adding element type (line 374)
        str_2765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 36), 'str', 'build_lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 36), tuple_2764, str_2765)
        # Adding element type (line 374)
        str_2766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 49), 'str', 'build_lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 36), tuple_2764, str_2766)
        
        # Processing the call keyword arguments (line 372)
        kwargs_2767 = {}
        # Getting the type of 'self' (line 372)
        self_2758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'self', False)
        # Obtaining the member 'set_undefined_options' of a type (line 372)
        set_undefined_options_2759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 8), self_2758, 'set_undefined_options')
        # Calling set_undefined_options(args, kwargs) (line 372)
        set_undefined_options_call_result_2768 = invoke(stypy.reporting.localization.Localization(__file__, 372, 8), set_undefined_options_2759, *[str_2760, tuple_2761, tuple_2764], **kwargs_2767)
        
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 240)
        stypy_return_type_2769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2769)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_2769


    @norecursion
    def dump_dirs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dump_dirs'
        module_type_store = module_type_store.open_function_context('dump_dirs', 382, 4, False)
        # Assigning a type to the variable 'self' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install.dump_dirs.__dict__.__setitem__('stypy_localization', localization)
        install.dump_dirs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install.dump_dirs.__dict__.__setitem__('stypy_type_store', module_type_store)
        install.dump_dirs.__dict__.__setitem__('stypy_function_name', 'install.dump_dirs')
        install.dump_dirs.__dict__.__setitem__('stypy_param_names_list', ['msg'])
        install.dump_dirs.__dict__.__setitem__('stypy_varargs_param_name', None)
        install.dump_dirs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install.dump_dirs.__dict__.__setitem__('stypy_call_defaults', defaults)
        install.dump_dirs.__dict__.__setitem__('stypy_call_varargs', varargs)
        install.dump_dirs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install.dump_dirs.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install.dump_dirs', ['msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dump_dirs', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dump_dirs(...)' code ##################

        
        # Getting the type of 'DEBUG' (line 383)
        DEBUG_2770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 11), 'DEBUG')
        # Testing the type of an if condition (line 383)
        if_condition_2771 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 383, 8), DEBUG_2770)
        # Assigning a type to the variable 'if_condition_2771' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'if_condition_2771', if_condition_2771)
        # SSA begins for if statement (line 383)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 384, 12))
        
        # 'from distutils.fancy_getopt import longopt_xlate' statement (line 384)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
        import_2772 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 384, 12), 'distutils.fancy_getopt')

        if (type(import_2772) is not StypyTypeError):

            if (import_2772 != 'pyd_module'):
                __import__(import_2772)
                sys_modules_2773 = sys.modules[import_2772]
                import_from_module(stypy.reporting.localization.Localization(__file__, 384, 12), 'distutils.fancy_getopt', sys_modules_2773.module_type_store, module_type_store, ['longopt_xlate'])
                nest_module(stypy.reporting.localization.Localization(__file__, 384, 12), __file__, sys_modules_2773, sys_modules_2773.module_type_store, module_type_store)
            else:
                from distutils.fancy_getopt import longopt_xlate

                import_from_module(stypy.reporting.localization.Localization(__file__, 384, 12), 'distutils.fancy_getopt', None, module_type_store, ['longopt_xlate'], [longopt_xlate])

        else:
            # Assigning a type to the variable 'distutils.fancy_getopt' (line 384)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'distutils.fancy_getopt', import_2772)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')
        
        # Getting the type of 'msg' (line 385)
        msg_2774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 18), 'msg')
        str_2775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 24), 'str', ':')
        # Applying the binary operator '+' (line 385)
        result_add_2776 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 18), '+', msg_2774, str_2775)
        
        
        # Getting the type of 'self' (line 386)
        self_2777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 23), 'self')
        # Obtaining the member 'user_options' of a type (line 386)
        user_options_2778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 23), self_2777, 'user_options')
        # Testing the type of a for loop iterable (line 386)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 386, 12), user_options_2778)
        # Getting the type of the for loop variable (line 386)
        for_loop_var_2779 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 386, 12), user_options_2778)
        # Assigning a type to the variable 'opt' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'opt', for_loop_var_2779)
        # SSA begins for a for statement (line 386)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 387):
        
        # Assigning a Subscript to a Name (line 387):
        
        # Obtaining the type of the subscript
        int_2780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 31), 'int')
        # Getting the type of 'opt' (line 387)
        opt_2781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 27), 'opt')
        # Obtaining the member '__getitem__' of a type (line 387)
        getitem___2782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 27), opt_2781, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 387)
        subscript_call_result_2783 = invoke(stypy.reporting.localization.Localization(__file__, 387, 27), getitem___2782, int_2780)
        
        # Assigning a type to the variable 'opt_name' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 16), 'opt_name', subscript_call_result_2783)
        
        
        
        # Obtaining the type of the subscript
        int_2784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 28), 'int')
        # Getting the type of 'opt_name' (line 388)
        opt_name_2785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 19), 'opt_name')
        # Obtaining the member '__getitem__' of a type (line 388)
        getitem___2786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 19), opt_name_2785, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 388)
        subscript_call_result_2787 = invoke(stypy.reporting.localization.Localization(__file__, 388, 19), getitem___2786, int_2784)
        
        str_2788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 35), 'str', '=')
        # Applying the binary operator '==' (line 388)
        result_eq_2789 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 19), '==', subscript_call_result_2787, str_2788)
        
        # Testing the type of an if condition (line 388)
        if_condition_2790 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 388, 16), result_eq_2789)
        # Assigning a type to the variable 'if_condition_2790' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 16), 'if_condition_2790', if_condition_2790)
        # SSA begins for if statement (line 388)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 389):
        
        # Assigning a Subscript to a Name (line 389):
        
        # Obtaining the type of the subscript
        int_2791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 40), 'int')
        int_2792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 42), 'int')
        slice_2793 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 389, 31), int_2791, int_2792, None)
        # Getting the type of 'opt_name' (line 389)
        opt_name_2794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 31), 'opt_name')
        # Obtaining the member '__getitem__' of a type (line 389)
        getitem___2795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 31), opt_name_2794, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 389)
        subscript_call_result_2796 = invoke(stypy.reporting.localization.Localization(__file__, 389, 31), getitem___2795, slice_2793)
        
        # Assigning a type to the variable 'opt_name' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 20), 'opt_name', subscript_call_result_2796)
        # SSA join for if statement (line 388)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'opt_name' (line 390)
        opt_name_2797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 19), 'opt_name')
        # Getting the type of 'self' (line 390)
        self_2798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 31), 'self')
        # Obtaining the member 'negative_opt' of a type (line 390)
        negative_opt_2799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 31), self_2798, 'negative_opt')
        # Applying the binary operator 'in' (line 390)
        result_contains_2800 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 19), 'in', opt_name_2797, negative_opt_2799)
        
        # Testing the type of an if condition (line 390)
        if_condition_2801 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 390, 16), result_contains_2800)
        # Assigning a type to the variable 'if_condition_2801' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 16), 'if_condition_2801', if_condition_2801)
        # SSA begins for if statement (line 390)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 391):
        
        # Assigning a Call to a Name (line 391):
        
        # Call to translate(...): (line 391)
        # Processing the call arguments (line 391)
        
        # Obtaining the type of the subscript
        # Getting the type of 'opt_name' (line 391)
        opt_name_2804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 66), 'opt_name', False)
        # Getting the type of 'self' (line 391)
        self_2805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 48), 'self', False)
        # Obtaining the member 'negative_opt' of a type (line 391)
        negative_opt_2806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 48), self_2805, 'negative_opt')
        # Obtaining the member '__getitem__' of a type (line 391)
        getitem___2807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 48), negative_opt_2806, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 391)
        subscript_call_result_2808 = invoke(stypy.reporting.localization.Localization(__file__, 391, 48), getitem___2807, opt_name_2804)
        
        # Getting the type of 'longopt_xlate' (line 392)
        longopt_xlate_2809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 48), 'longopt_xlate', False)
        # Processing the call keyword arguments (line 391)
        kwargs_2810 = {}
        # Getting the type of 'string' (line 391)
        string_2802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 31), 'string', False)
        # Obtaining the member 'translate' of a type (line 391)
        translate_2803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 31), string_2802, 'translate')
        # Calling translate(args, kwargs) (line 391)
        translate_call_result_2811 = invoke(stypy.reporting.localization.Localization(__file__, 391, 31), translate_2803, *[subscript_call_result_2808, longopt_xlate_2809], **kwargs_2810)
        
        # Assigning a type to the variable 'opt_name' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 20), 'opt_name', translate_call_result_2811)
        
        # Assigning a UnaryOp to a Name (line 393):
        
        # Assigning a UnaryOp to a Name (line 393):
        
        
        # Call to getattr(...): (line 393)
        # Processing the call arguments (line 393)
        # Getting the type of 'self' (line 393)
        self_2813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 38), 'self', False)
        # Getting the type of 'opt_name' (line 393)
        opt_name_2814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 44), 'opt_name', False)
        # Processing the call keyword arguments (line 393)
        kwargs_2815 = {}
        # Getting the type of 'getattr' (line 393)
        getattr_2812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 30), 'getattr', False)
        # Calling getattr(args, kwargs) (line 393)
        getattr_call_result_2816 = invoke(stypy.reporting.localization.Localization(__file__, 393, 30), getattr_2812, *[self_2813, opt_name_2814], **kwargs_2815)
        
        # Applying the 'not' unary operator (line 393)
        result_not__2817 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 26), 'not', getattr_call_result_2816)
        
        # Assigning a type to the variable 'val' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 20), 'val', result_not__2817)
        # SSA branch for the else part of an if statement (line 390)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 395):
        
        # Assigning a Call to a Name (line 395):
        
        # Call to translate(...): (line 395)
        # Processing the call arguments (line 395)
        # Getting the type of 'opt_name' (line 395)
        opt_name_2820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 48), 'opt_name', False)
        # Getting the type of 'longopt_xlate' (line 395)
        longopt_xlate_2821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 58), 'longopt_xlate', False)
        # Processing the call keyword arguments (line 395)
        kwargs_2822 = {}
        # Getting the type of 'string' (line 395)
        string_2818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 31), 'string', False)
        # Obtaining the member 'translate' of a type (line 395)
        translate_2819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 31), string_2818, 'translate')
        # Calling translate(args, kwargs) (line 395)
        translate_call_result_2823 = invoke(stypy.reporting.localization.Localization(__file__, 395, 31), translate_2819, *[opt_name_2820, longopt_xlate_2821], **kwargs_2822)
        
        # Assigning a type to the variable 'opt_name' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 20), 'opt_name', translate_call_result_2823)
        
        # Assigning a Call to a Name (line 396):
        
        # Assigning a Call to a Name (line 396):
        
        # Call to getattr(...): (line 396)
        # Processing the call arguments (line 396)
        # Getting the type of 'self' (line 396)
        self_2825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 34), 'self', False)
        # Getting the type of 'opt_name' (line 396)
        opt_name_2826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 40), 'opt_name', False)
        # Processing the call keyword arguments (line 396)
        kwargs_2827 = {}
        # Getting the type of 'getattr' (line 396)
        getattr_2824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 26), 'getattr', False)
        # Calling getattr(args, kwargs) (line 396)
        getattr_call_result_2828 = invoke(stypy.reporting.localization.Localization(__file__, 396, 26), getattr_2824, *[self_2825, opt_name_2826], **kwargs_2827)
        
        # Assigning a type to the variable 'val' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 20), 'val', getattr_call_result_2828)
        # SSA join for if statement (line 390)
        module_type_store = module_type_store.join_ssa_context()
        
        str_2829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 22), 'str', '  %s: %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 397)
        tuple_2830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 397)
        # Adding element type (line 397)
        # Getting the type of 'opt_name' (line 397)
        opt_name_2831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 36), 'opt_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 36), tuple_2830, opt_name_2831)
        # Adding element type (line 397)
        # Getting the type of 'val' (line 397)
        val_2832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 46), 'val')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 36), tuple_2830, val_2832)
        
        # Applying the binary operator '%' (line 397)
        result_mod_2833 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 22), '%', str_2829, tuple_2830)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 383)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'dump_dirs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dump_dirs' in the type store
        # Getting the type of 'stypy_return_type' (line 382)
        stypy_return_type_2834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2834)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dump_dirs'
        return stypy_return_type_2834


    @norecursion
    def finalize_unix(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_unix'
        module_type_store = module_type_store.open_function_context('finalize_unix', 400, 4, False)
        # Assigning a type to the variable 'self' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install.finalize_unix.__dict__.__setitem__('stypy_localization', localization)
        install.finalize_unix.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install.finalize_unix.__dict__.__setitem__('stypy_type_store', module_type_store)
        install.finalize_unix.__dict__.__setitem__('stypy_function_name', 'install.finalize_unix')
        install.finalize_unix.__dict__.__setitem__('stypy_param_names_list', [])
        install.finalize_unix.__dict__.__setitem__('stypy_varargs_param_name', None)
        install.finalize_unix.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install.finalize_unix.__dict__.__setitem__('stypy_call_defaults', defaults)
        install.finalize_unix.__dict__.__setitem__('stypy_call_varargs', varargs)
        install.finalize_unix.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install.finalize_unix.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install.finalize_unix', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'finalize_unix', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'finalize_unix(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 402)
        self_2835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 11), 'self')
        # Obtaining the member 'install_base' of a type (line 402)
        install_base_2836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 11), self_2835, 'install_base')
        # Getting the type of 'None' (line 402)
        None_2837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 36), 'None')
        # Applying the binary operator 'isnot' (line 402)
        result_is_not_2838 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 11), 'isnot', install_base_2836, None_2837)
        
        
        # Getting the type of 'self' (line 402)
        self_2839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 44), 'self')
        # Obtaining the member 'install_platbase' of a type (line 402)
        install_platbase_2840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 44), self_2839, 'install_platbase')
        # Getting the type of 'None' (line 402)
        None_2841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 73), 'None')
        # Applying the binary operator 'isnot' (line 402)
        result_is_not_2842 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 44), 'isnot', install_platbase_2840, None_2841)
        
        # Applying the binary operator 'or' (line 402)
        result_or_keyword_2843 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 11), 'or', result_is_not_2838, result_is_not_2842)
        
        # Testing the type of an if condition (line 402)
        if_condition_2844 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 402, 8), result_or_keyword_2843)
        # Assigning a type to the variable 'if_condition_2844' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'if_condition_2844', if_condition_2844)
        # SSA begins for if statement (line 402)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 403)
        self_2845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 17), 'self')
        # Obtaining the member 'install_lib' of a type (line 403)
        install_lib_2846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 17), self_2845, 'install_lib')
        # Getting the type of 'None' (line 403)
        None_2847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 37), 'None')
        # Applying the binary operator 'is' (line 403)
        result_is__2848 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 17), 'is', install_lib_2846, None_2847)
        
        
        # Getting the type of 'self' (line 404)
        self_2849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 17), 'self')
        # Obtaining the member 'install_purelib' of a type (line 404)
        install_purelib_2850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 17), self_2849, 'install_purelib')
        # Getting the type of 'None' (line 404)
        None_2851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 41), 'None')
        # Applying the binary operator 'is' (line 404)
        result_is__2852 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 17), 'is', install_purelib_2850, None_2851)
        
        # Applying the binary operator 'and' (line 403)
        result_and_keyword_2853 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 17), 'and', result_is__2848, result_is__2852)
        
        # Getting the type of 'self' (line 405)
        self_2854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 17), 'self')
        # Obtaining the member 'install_platlib' of a type (line 405)
        install_platlib_2855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 17), self_2854, 'install_platlib')
        # Getting the type of 'None' (line 405)
        None_2856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 41), 'None')
        # Applying the binary operator 'is' (line 405)
        result_is__2857 = python_operator(stypy.reporting.localization.Localization(__file__, 405, 17), 'is', install_platlib_2855, None_2856)
        
        # Applying the binary operator 'and' (line 403)
        result_and_keyword_2858 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 17), 'and', result_and_keyword_2853, result_is__2857)
        
        
        # Getting the type of 'self' (line 406)
        self_2859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 16), 'self')
        # Obtaining the member 'install_headers' of a type (line 406)
        install_headers_2860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 16), self_2859, 'install_headers')
        # Getting the type of 'None' (line 406)
        None_2861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 40), 'None')
        # Applying the binary operator 'is' (line 406)
        result_is__2862 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 16), 'is', install_headers_2860, None_2861)
        
        # Applying the binary operator 'or' (line 403)
        result_or_keyword_2863 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 16), 'or', result_and_keyword_2858, result_is__2862)
        
        # Getting the type of 'self' (line 407)
        self_2864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 16), 'self')
        # Obtaining the member 'install_scripts' of a type (line 407)
        install_scripts_2865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 16), self_2864, 'install_scripts')
        # Getting the type of 'None' (line 407)
        None_2866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 40), 'None')
        # Applying the binary operator 'is' (line 407)
        result_is__2867 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 16), 'is', install_scripts_2865, None_2866)
        
        # Applying the binary operator 'or' (line 403)
        result_or_keyword_2868 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 16), 'or', result_or_keyword_2863, result_is__2867)
        
        # Getting the type of 'self' (line 408)
        self_2869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 16), 'self')
        # Obtaining the member 'install_data' of a type (line 408)
        install_data_2870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 16), self_2869, 'install_data')
        # Getting the type of 'None' (line 408)
        None_2871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 37), 'None')
        # Applying the binary operator 'is' (line 408)
        result_is__2872 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 16), 'is', install_data_2870, None_2871)
        
        # Applying the binary operator 'or' (line 403)
        result_or_keyword_2873 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 16), 'or', result_or_keyword_2868, result_is__2872)
        
        # Testing the type of an if condition (line 403)
        if_condition_2874 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 403, 12), result_or_keyword_2873)
        # Assigning a type to the variable 'if_condition_2874' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 12), 'if_condition_2874', if_condition_2874)
        # SSA begins for if statement (line 403)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsOptionError' (line 409)
        DistutilsOptionError_2875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 22), 'DistutilsOptionError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 409, 16), DistutilsOptionError_2875, 'raise parameter', BaseException)
        # SSA join for if statement (line 403)
        module_type_store = module_type_store.join_ssa_context()
        
        # Assigning a type to the variable 'stypy_return_type' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 402)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 414)
        self_2876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 11), 'self')
        # Obtaining the member 'user' of a type (line 414)
        user_2877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 11), self_2876, 'user')
        # Testing the type of an if condition (line 414)
        if_condition_2878 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 414, 8), user_2877)
        # Assigning a type to the variable 'if_condition_2878' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'if_condition_2878', if_condition_2878)
        # SSA begins for if statement (line 414)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 415)
        # Getting the type of 'self' (line 415)
        self_2879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 15), 'self')
        # Obtaining the member 'install_userbase' of a type (line 415)
        install_userbase_2880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 15), self_2879, 'install_userbase')
        # Getting the type of 'None' (line 415)
        None_2881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 40), 'None')
        
        (may_be_2882, more_types_in_union_2883) = may_be_none(install_userbase_2880, None_2881)

        if may_be_2882:

            if more_types_in_union_2883:
                # Runtime conditional SSA (line 415)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to DistutilsPlatformError(...): (line 416)
            # Processing the call arguments (line 416)
            str_2885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 20), 'str', 'User base directory is not specified')
            # Processing the call keyword arguments (line 416)
            kwargs_2886 = {}
            # Getting the type of 'DistutilsPlatformError' (line 416)
            DistutilsPlatformError_2884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 22), 'DistutilsPlatformError', False)
            # Calling DistutilsPlatformError(args, kwargs) (line 416)
            DistutilsPlatformError_call_result_2887 = invoke(stypy.reporting.localization.Localization(__file__, 416, 22), DistutilsPlatformError_2884, *[str_2885], **kwargs_2886)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 416, 16), DistutilsPlatformError_call_result_2887, 'raise parameter', BaseException)

            if more_types_in_union_2883:
                # SSA join for if statement (line 415)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Attribute to a Attribute (line 418):
        # Getting the type of 'self' (line 418)
        self_2888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 56), 'self')
        # Obtaining the member 'install_userbase' of a type (line 418)
        install_userbase_2889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 56), self_2888, 'install_userbase')
        # Getting the type of 'self' (line 418)
        self_2890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 32), 'self')
        # Setting the type of the member 'install_platbase' of a type (line 418)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 32), self_2890, 'install_platbase', install_userbase_2889)
        
        # Assigning a Attribute to a Attribute (line 418):
        # Getting the type of 'self' (line 418)
        self_2891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 32), 'self')
        # Obtaining the member 'install_platbase' of a type (line 418)
        install_platbase_2892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 32), self_2891, 'install_platbase')
        # Getting the type of 'self' (line 418)
        self_2893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 12), 'self')
        # Setting the type of the member 'install_base' of a type (line 418)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 12), self_2893, 'install_base', install_platbase_2892)
        
        # Call to select_scheme(...): (line 419)
        # Processing the call arguments (line 419)
        str_2896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 31), 'str', 'unix_user')
        # Processing the call keyword arguments (line 419)
        kwargs_2897 = {}
        # Getting the type of 'self' (line 419)
        self_2894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 12), 'self', False)
        # Obtaining the member 'select_scheme' of a type (line 419)
        select_scheme_2895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 12), self_2894, 'select_scheme')
        # Calling select_scheme(args, kwargs) (line 419)
        select_scheme_call_result_2898 = invoke(stypy.reporting.localization.Localization(__file__, 419, 12), select_scheme_2895, *[str_2896], **kwargs_2897)
        
        # SSA branch for the else part of an if statement (line 414)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 420)
        self_2899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 13), 'self')
        # Obtaining the member 'home' of a type (line 420)
        home_2900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 13), self_2899, 'home')
        # Getting the type of 'None' (line 420)
        None_2901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 30), 'None')
        # Applying the binary operator 'isnot' (line 420)
        result_is_not_2902 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 13), 'isnot', home_2900, None_2901)
        
        # Testing the type of an if condition (line 420)
        if_condition_2903 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 420, 13), result_is_not_2902)
        # Assigning a type to the variable 'if_condition_2903' (line 420)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 13), 'if_condition_2903', if_condition_2903)
        # SSA begins for if statement (line 420)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Attribute to a Attribute (line 421):
        # Getting the type of 'self' (line 421)
        self_2904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 56), 'self')
        # Obtaining the member 'home' of a type (line 421)
        home_2905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 56), self_2904, 'home')
        # Getting the type of 'self' (line 421)
        self_2906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 32), 'self')
        # Setting the type of the member 'install_platbase' of a type (line 421)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 32), self_2906, 'install_platbase', home_2905)
        
        # Assigning a Attribute to a Attribute (line 421):
        # Getting the type of 'self' (line 421)
        self_2907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 32), 'self')
        # Obtaining the member 'install_platbase' of a type (line 421)
        install_platbase_2908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 32), self_2907, 'install_platbase')
        # Getting the type of 'self' (line 421)
        self_2909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 12), 'self')
        # Setting the type of the member 'install_base' of a type (line 421)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 12), self_2909, 'install_base', install_platbase_2908)
        
        # Call to select_scheme(...): (line 422)
        # Processing the call arguments (line 422)
        str_2912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 31), 'str', 'unix_home')
        # Processing the call keyword arguments (line 422)
        kwargs_2913 = {}
        # Getting the type of 'self' (line 422)
        self_2910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 12), 'self', False)
        # Obtaining the member 'select_scheme' of a type (line 422)
        select_scheme_2911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 12), self_2910, 'select_scheme')
        # Calling select_scheme(args, kwargs) (line 422)
        select_scheme_call_result_2914 = invoke(stypy.reporting.localization.Localization(__file__, 422, 12), select_scheme_2911, *[str_2912], **kwargs_2913)
        
        # SSA branch for the else part of an if statement (line 420)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 424)
        # Getting the type of 'self' (line 424)
        self_2915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 15), 'self')
        # Obtaining the member 'prefix' of a type (line 424)
        prefix_2916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 15), self_2915, 'prefix')
        # Getting the type of 'None' (line 424)
        None_2917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 30), 'None')
        
        (may_be_2918, more_types_in_union_2919) = may_be_none(prefix_2916, None_2917)

        if may_be_2918:

            if more_types_in_union_2919:
                # Runtime conditional SSA (line 424)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Getting the type of 'self' (line 425)
            self_2920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 19), 'self')
            # Obtaining the member 'exec_prefix' of a type (line 425)
            exec_prefix_2921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 19), self_2920, 'exec_prefix')
            # Getting the type of 'None' (line 425)
            None_2922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 43), 'None')
            # Applying the binary operator 'isnot' (line 425)
            result_is_not_2923 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 19), 'isnot', exec_prefix_2921, None_2922)
            
            # Testing the type of an if condition (line 425)
            if_condition_2924 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 425, 16), result_is_not_2923)
            # Assigning a type to the variable 'if_condition_2924' (line 425)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 16), 'if_condition_2924', if_condition_2924)
            # SSA begins for if statement (line 425)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'DistutilsOptionError' (line 426)
            DistutilsOptionError_2925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 26), 'DistutilsOptionError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 426, 20), DistutilsOptionError_2925, 'raise parameter', BaseException)
            # SSA join for if statement (line 425)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Attribute (line 429):
            
            # Assigning a Call to a Attribute (line 429):
            
            # Call to normpath(...): (line 429)
            # Processing the call arguments (line 429)
            # Getting the type of 'sys' (line 429)
            sys_2929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 47), 'sys', False)
            # Obtaining the member 'prefix' of a type (line 429)
            prefix_2930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 47), sys_2929, 'prefix')
            # Processing the call keyword arguments (line 429)
            kwargs_2931 = {}
            # Getting the type of 'os' (line 429)
            os_2926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 30), 'os', False)
            # Obtaining the member 'path' of a type (line 429)
            path_2927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 30), os_2926, 'path')
            # Obtaining the member 'normpath' of a type (line 429)
            normpath_2928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 30), path_2927, 'normpath')
            # Calling normpath(args, kwargs) (line 429)
            normpath_call_result_2932 = invoke(stypy.reporting.localization.Localization(__file__, 429, 30), normpath_2928, *[prefix_2930], **kwargs_2931)
            
            # Getting the type of 'self' (line 429)
            self_2933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 16), 'self')
            # Setting the type of the member 'prefix' of a type (line 429)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 16), self_2933, 'prefix', normpath_call_result_2932)
            
            # Assigning a Call to a Attribute (line 430):
            
            # Assigning a Call to a Attribute (line 430):
            
            # Call to normpath(...): (line 430)
            # Processing the call arguments (line 430)
            # Getting the type of 'sys' (line 430)
            sys_2937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 52), 'sys', False)
            # Obtaining the member 'exec_prefix' of a type (line 430)
            exec_prefix_2938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 52), sys_2937, 'exec_prefix')
            # Processing the call keyword arguments (line 430)
            kwargs_2939 = {}
            # Getting the type of 'os' (line 430)
            os_2934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 35), 'os', False)
            # Obtaining the member 'path' of a type (line 430)
            path_2935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 35), os_2934, 'path')
            # Obtaining the member 'normpath' of a type (line 430)
            normpath_2936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 35), path_2935, 'normpath')
            # Calling normpath(args, kwargs) (line 430)
            normpath_call_result_2940 = invoke(stypy.reporting.localization.Localization(__file__, 430, 35), normpath_2936, *[exec_prefix_2938], **kwargs_2939)
            
            # Getting the type of 'self' (line 430)
            self_2941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 16), 'self')
            # Setting the type of the member 'exec_prefix' of a type (line 430)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 16), self_2941, 'exec_prefix', normpath_call_result_2940)

            if more_types_in_union_2919:
                # Runtime conditional SSA for else branch (line 424)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_2918) or more_types_in_union_2919):
            
            # Type idiom detected: calculating its left and rigth part (line 433)
            # Getting the type of 'self' (line 433)
            self_2942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 19), 'self')
            # Obtaining the member 'exec_prefix' of a type (line 433)
            exec_prefix_2943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 19), self_2942, 'exec_prefix')
            # Getting the type of 'None' (line 433)
            None_2944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 39), 'None')
            
            (may_be_2945, more_types_in_union_2946) = may_be_none(exec_prefix_2943, None_2944)

            if may_be_2945:

                if more_types_in_union_2946:
                    # Runtime conditional SSA (line 433)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Attribute to a Attribute (line 434):
                
                # Assigning a Attribute to a Attribute (line 434):
                # Getting the type of 'self' (line 434)
                self_2947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 39), 'self')
                # Obtaining the member 'prefix' of a type (line 434)
                prefix_2948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 39), self_2947, 'prefix')
                # Getting the type of 'self' (line 434)
                self_2949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 20), 'self')
                # Setting the type of the member 'exec_prefix' of a type (line 434)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 20), self_2949, 'exec_prefix', prefix_2948)

                if more_types_in_union_2946:
                    # SSA join for if statement (line 433)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_2918 and more_types_in_union_2919):
                # SSA join for if statement (line 424)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Attribute (line 436):
        
        # Assigning a Attribute to a Attribute (line 436):
        # Getting the type of 'self' (line 436)
        self_2950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 32), 'self')
        # Obtaining the member 'prefix' of a type (line 436)
        prefix_2951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 32), self_2950, 'prefix')
        # Getting the type of 'self' (line 436)
        self_2952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'self')
        # Setting the type of the member 'install_base' of a type (line 436)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 12), self_2952, 'install_base', prefix_2951)
        
        # Assigning a Attribute to a Attribute (line 437):
        
        # Assigning a Attribute to a Attribute (line 437):
        # Getting the type of 'self' (line 437)
        self_2953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 36), 'self')
        # Obtaining the member 'exec_prefix' of a type (line 437)
        exec_prefix_2954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 36), self_2953, 'exec_prefix')
        # Getting the type of 'self' (line 437)
        self_2955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 12), 'self')
        # Setting the type of the member 'install_platbase' of a type (line 437)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 12), self_2955, 'install_platbase', exec_prefix_2954)
        
        # Call to select_scheme(...): (line 438)
        # Processing the call arguments (line 438)
        str_2958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 31), 'str', 'unix_prefix')
        # Processing the call keyword arguments (line 438)
        kwargs_2959 = {}
        # Getting the type of 'self' (line 438)
        self_2956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'self', False)
        # Obtaining the member 'select_scheme' of a type (line 438)
        select_scheme_2957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 12), self_2956, 'select_scheme')
        # Calling select_scheme(args, kwargs) (line 438)
        select_scheme_call_result_2960 = invoke(stypy.reporting.localization.Localization(__file__, 438, 12), select_scheme_2957, *[str_2958], **kwargs_2959)
        
        # SSA join for if statement (line 420)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 414)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'finalize_unix(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_unix' in the type store
        # Getting the type of 'stypy_return_type' (line 400)
        stypy_return_type_2961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2961)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_unix'
        return stypy_return_type_2961


    @norecursion
    def finalize_other(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_other'
        module_type_store = module_type_store.open_function_context('finalize_other', 443, 4, False)
        # Assigning a type to the variable 'self' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install.finalize_other.__dict__.__setitem__('stypy_localization', localization)
        install.finalize_other.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install.finalize_other.__dict__.__setitem__('stypy_type_store', module_type_store)
        install.finalize_other.__dict__.__setitem__('stypy_function_name', 'install.finalize_other')
        install.finalize_other.__dict__.__setitem__('stypy_param_names_list', [])
        install.finalize_other.__dict__.__setitem__('stypy_varargs_param_name', None)
        install.finalize_other.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install.finalize_other.__dict__.__setitem__('stypy_call_defaults', defaults)
        install.finalize_other.__dict__.__setitem__('stypy_call_varargs', varargs)
        install.finalize_other.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install.finalize_other.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install.finalize_other', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'finalize_other', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'finalize_other(...)' code ##################

        
        # Getting the type of 'self' (line 445)
        self_2962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 11), 'self')
        # Obtaining the member 'user' of a type (line 445)
        user_2963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 11), self_2962, 'user')
        # Testing the type of an if condition (line 445)
        if_condition_2964 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 445, 8), user_2963)
        # Assigning a type to the variable 'if_condition_2964' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'if_condition_2964', if_condition_2964)
        # SSA begins for if statement (line 445)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 446)
        # Getting the type of 'self' (line 446)
        self_2965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 15), 'self')
        # Obtaining the member 'install_userbase' of a type (line 446)
        install_userbase_2966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 15), self_2965, 'install_userbase')
        # Getting the type of 'None' (line 446)
        None_2967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 40), 'None')
        
        (may_be_2968, more_types_in_union_2969) = may_be_none(install_userbase_2966, None_2967)

        if may_be_2968:

            if more_types_in_union_2969:
                # Runtime conditional SSA (line 446)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to DistutilsPlatformError(...): (line 447)
            # Processing the call arguments (line 447)
            str_2971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 20), 'str', 'User base directory is not specified')
            # Processing the call keyword arguments (line 447)
            kwargs_2972 = {}
            # Getting the type of 'DistutilsPlatformError' (line 447)
            DistutilsPlatformError_2970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 22), 'DistutilsPlatformError', False)
            # Calling DistutilsPlatformError(args, kwargs) (line 447)
            DistutilsPlatformError_call_result_2973 = invoke(stypy.reporting.localization.Localization(__file__, 447, 22), DistutilsPlatformError_2970, *[str_2971], **kwargs_2972)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 447, 16), DistutilsPlatformError_call_result_2973, 'raise parameter', BaseException)

            if more_types_in_union_2969:
                # SSA join for if statement (line 446)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Attribute to a Attribute (line 449):
        # Getting the type of 'self' (line 449)
        self_2974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 56), 'self')
        # Obtaining the member 'install_userbase' of a type (line 449)
        install_userbase_2975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 56), self_2974, 'install_userbase')
        # Getting the type of 'self' (line 449)
        self_2976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 32), 'self')
        # Setting the type of the member 'install_platbase' of a type (line 449)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 32), self_2976, 'install_platbase', install_userbase_2975)
        
        # Assigning a Attribute to a Attribute (line 449):
        # Getting the type of 'self' (line 449)
        self_2977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 32), 'self')
        # Obtaining the member 'install_platbase' of a type (line 449)
        install_platbase_2978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 32), self_2977, 'install_platbase')
        # Getting the type of 'self' (line 449)
        self_2979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 12), 'self')
        # Setting the type of the member 'install_base' of a type (line 449)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 12), self_2979, 'install_base', install_platbase_2978)
        
        # Call to select_scheme(...): (line 450)
        # Processing the call arguments (line 450)
        # Getting the type of 'os' (line 450)
        os_2982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 31), 'os', False)
        # Obtaining the member 'name' of a type (line 450)
        name_2983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 31), os_2982, 'name')
        str_2984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 41), 'str', '_user')
        # Applying the binary operator '+' (line 450)
        result_add_2985 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 31), '+', name_2983, str_2984)
        
        # Processing the call keyword arguments (line 450)
        kwargs_2986 = {}
        # Getting the type of 'self' (line 450)
        self_2980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 12), 'self', False)
        # Obtaining the member 'select_scheme' of a type (line 450)
        select_scheme_2981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 12), self_2980, 'select_scheme')
        # Calling select_scheme(args, kwargs) (line 450)
        select_scheme_call_result_2987 = invoke(stypy.reporting.localization.Localization(__file__, 450, 12), select_scheme_2981, *[result_add_2985], **kwargs_2986)
        
        # SSA branch for the else part of an if statement (line 445)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 451)
        self_2988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 13), 'self')
        # Obtaining the member 'home' of a type (line 451)
        home_2989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 13), self_2988, 'home')
        # Getting the type of 'None' (line 451)
        None_2990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 30), 'None')
        # Applying the binary operator 'isnot' (line 451)
        result_is_not_2991 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 13), 'isnot', home_2989, None_2990)
        
        # Testing the type of an if condition (line 451)
        if_condition_2992 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 451, 13), result_is_not_2991)
        # Assigning a type to the variable 'if_condition_2992' (line 451)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 13), 'if_condition_2992', if_condition_2992)
        # SSA begins for if statement (line 451)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Attribute to a Attribute (line 452):
        # Getting the type of 'self' (line 452)
        self_2993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 56), 'self')
        # Obtaining the member 'home' of a type (line 452)
        home_2994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 56), self_2993, 'home')
        # Getting the type of 'self' (line 452)
        self_2995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 32), 'self')
        # Setting the type of the member 'install_platbase' of a type (line 452)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 32), self_2995, 'install_platbase', home_2994)
        
        # Assigning a Attribute to a Attribute (line 452):
        # Getting the type of 'self' (line 452)
        self_2996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 32), 'self')
        # Obtaining the member 'install_platbase' of a type (line 452)
        install_platbase_2997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 32), self_2996, 'install_platbase')
        # Getting the type of 'self' (line 452)
        self_2998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 12), 'self')
        # Setting the type of the member 'install_base' of a type (line 452)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 12), self_2998, 'install_base', install_platbase_2997)
        
        # Call to select_scheme(...): (line 453)
        # Processing the call arguments (line 453)
        str_3001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 31), 'str', 'unix_home')
        # Processing the call keyword arguments (line 453)
        kwargs_3002 = {}
        # Getting the type of 'self' (line 453)
        self_2999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 12), 'self', False)
        # Obtaining the member 'select_scheme' of a type (line 453)
        select_scheme_3000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 12), self_2999, 'select_scheme')
        # Calling select_scheme(args, kwargs) (line 453)
        select_scheme_call_result_3003 = invoke(stypy.reporting.localization.Localization(__file__, 453, 12), select_scheme_3000, *[str_3001], **kwargs_3002)
        
        # SSA branch for the else part of an if statement (line 451)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 455)
        # Getting the type of 'self' (line 455)
        self_3004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 15), 'self')
        # Obtaining the member 'prefix' of a type (line 455)
        prefix_3005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 15), self_3004, 'prefix')
        # Getting the type of 'None' (line 455)
        None_3006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 30), 'None')
        
        (may_be_3007, more_types_in_union_3008) = may_be_none(prefix_3005, None_3006)

        if may_be_3007:

            if more_types_in_union_3008:
                # Runtime conditional SSA (line 455)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 456):
            
            # Assigning a Call to a Attribute (line 456):
            
            # Call to normpath(...): (line 456)
            # Processing the call arguments (line 456)
            # Getting the type of 'sys' (line 456)
            sys_3012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 47), 'sys', False)
            # Obtaining the member 'prefix' of a type (line 456)
            prefix_3013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 47), sys_3012, 'prefix')
            # Processing the call keyword arguments (line 456)
            kwargs_3014 = {}
            # Getting the type of 'os' (line 456)
            os_3009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 30), 'os', False)
            # Obtaining the member 'path' of a type (line 456)
            path_3010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 30), os_3009, 'path')
            # Obtaining the member 'normpath' of a type (line 456)
            normpath_3011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 30), path_3010, 'normpath')
            # Calling normpath(args, kwargs) (line 456)
            normpath_call_result_3015 = invoke(stypy.reporting.localization.Localization(__file__, 456, 30), normpath_3011, *[prefix_3013], **kwargs_3014)
            
            # Getting the type of 'self' (line 456)
            self_3016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'self')
            # Setting the type of the member 'prefix' of a type (line 456)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 16), self_3016, 'prefix', normpath_call_result_3015)

            if more_types_in_union_3008:
                # SSA join for if statement (line 455)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Attribute to a Attribute (line 458):
        # Getting the type of 'self' (line 458)
        self_3017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 56), 'self')
        # Obtaining the member 'prefix' of a type (line 458)
        prefix_3018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 56), self_3017, 'prefix')
        # Getting the type of 'self' (line 458)
        self_3019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 32), 'self')
        # Setting the type of the member 'install_platbase' of a type (line 458)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 32), self_3019, 'install_platbase', prefix_3018)
        
        # Assigning a Attribute to a Attribute (line 458):
        # Getting the type of 'self' (line 458)
        self_3020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 32), 'self')
        # Obtaining the member 'install_platbase' of a type (line 458)
        install_platbase_3021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 32), self_3020, 'install_platbase')
        # Getting the type of 'self' (line 458)
        self_3022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 12), 'self')
        # Setting the type of the member 'install_base' of a type (line 458)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 12), self_3022, 'install_base', install_platbase_3021)
        
        
        # SSA begins for try-except statement (line 459)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to select_scheme(...): (line 460)
        # Processing the call arguments (line 460)
        # Getting the type of 'os' (line 460)
        os_3025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 35), 'os', False)
        # Obtaining the member 'name' of a type (line 460)
        name_3026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 35), os_3025, 'name')
        # Processing the call keyword arguments (line 460)
        kwargs_3027 = {}
        # Getting the type of 'self' (line 460)
        self_3023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 16), 'self', False)
        # Obtaining the member 'select_scheme' of a type (line 460)
        select_scheme_3024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 16), self_3023, 'select_scheme')
        # Calling select_scheme(args, kwargs) (line 460)
        select_scheme_call_result_3028 = invoke(stypy.reporting.localization.Localization(__file__, 460, 16), select_scheme_3024, *[name_3026], **kwargs_3027)
        
        # SSA branch for the except part of a try statement (line 459)
        # SSA branch for the except 'KeyError' branch of a try statement (line 459)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsPlatformError' (line 462)
        DistutilsPlatformError_3029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 22), 'DistutilsPlatformError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 462, 16), DistutilsPlatformError_3029, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 459)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 451)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 445)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'finalize_other(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_other' in the type store
        # Getting the type of 'stypy_return_type' (line 443)
        stypy_return_type_3030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3030)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_other'
        return stypy_return_type_3030


    @norecursion
    def select_scheme(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'select_scheme'
        module_type_store = module_type_store.open_function_context('select_scheme', 468, 4, False)
        # Assigning a type to the variable 'self' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install.select_scheme.__dict__.__setitem__('stypy_localization', localization)
        install.select_scheme.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install.select_scheme.__dict__.__setitem__('stypy_type_store', module_type_store)
        install.select_scheme.__dict__.__setitem__('stypy_function_name', 'install.select_scheme')
        install.select_scheme.__dict__.__setitem__('stypy_param_names_list', ['name'])
        install.select_scheme.__dict__.__setitem__('stypy_varargs_param_name', None)
        install.select_scheme.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install.select_scheme.__dict__.__setitem__('stypy_call_defaults', defaults)
        install.select_scheme.__dict__.__setitem__('stypy_call_varargs', varargs)
        install.select_scheme.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install.select_scheme.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install.select_scheme', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'select_scheme', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'select_scheme(...)' code ##################

        
        # Assigning a Subscript to a Name (line 470):
        
        # Assigning a Subscript to a Name (line 470):
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 470)
        name_3031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 33), 'name')
        # Getting the type of 'INSTALL_SCHEMES' (line 470)
        INSTALL_SCHEMES_3032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 17), 'INSTALL_SCHEMES')
        # Obtaining the member '__getitem__' of a type (line 470)
        getitem___3033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 17), INSTALL_SCHEMES_3032, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 470)
        subscript_call_result_3034 = invoke(stypy.reporting.localization.Localization(__file__, 470, 17), getitem___3033, name_3031)
        
        # Assigning a type to the variable 'scheme' (line 470)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'scheme', subscript_call_result_3034)
        
        # Getting the type of 'SCHEME_KEYS' (line 471)
        SCHEME_KEYS_3035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 19), 'SCHEME_KEYS')
        # Testing the type of a for loop iterable (line 471)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 471, 8), SCHEME_KEYS_3035)
        # Getting the type of the for loop variable (line 471)
        for_loop_var_3036 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 471, 8), SCHEME_KEYS_3035)
        # Assigning a type to the variable 'key' (line 471)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'key', for_loop_var_3036)
        # SSA begins for a for statement (line 471)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 472):
        
        # Assigning a BinOp to a Name (line 472):
        str_3037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 23), 'str', 'install_')
        # Getting the type of 'key' (line 472)
        key_3038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 36), 'key')
        # Applying the binary operator '+' (line 472)
        result_add_3039 = python_operator(stypy.reporting.localization.Localization(__file__, 472, 23), '+', str_3037, key_3038)
        
        # Assigning a type to the variable 'attrname' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 12), 'attrname', result_add_3039)
        
        # Type idiom detected: calculating its left and rigth part (line 473)
        
        # Call to getattr(...): (line 473)
        # Processing the call arguments (line 473)
        # Getting the type of 'self' (line 473)
        self_3041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 23), 'self', False)
        # Getting the type of 'attrname' (line 473)
        attrname_3042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 29), 'attrname', False)
        # Processing the call keyword arguments (line 473)
        kwargs_3043 = {}
        # Getting the type of 'getattr' (line 473)
        getattr_3040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 15), 'getattr', False)
        # Calling getattr(args, kwargs) (line 473)
        getattr_call_result_3044 = invoke(stypy.reporting.localization.Localization(__file__, 473, 15), getattr_3040, *[self_3041, attrname_3042], **kwargs_3043)
        
        # Getting the type of 'None' (line 473)
        None_3045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 42), 'None')
        
        (may_be_3046, more_types_in_union_3047) = may_be_none(getattr_call_result_3044, None_3045)

        if may_be_3046:

            if more_types_in_union_3047:
                # Runtime conditional SSA (line 473)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to setattr(...): (line 474)
            # Processing the call arguments (line 474)
            # Getting the type of 'self' (line 474)
            self_3049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 24), 'self', False)
            # Getting the type of 'attrname' (line 474)
            attrname_3050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 30), 'attrname', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'key' (line 474)
            key_3051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 47), 'key', False)
            # Getting the type of 'scheme' (line 474)
            scheme_3052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 40), 'scheme', False)
            # Obtaining the member '__getitem__' of a type (line 474)
            getitem___3053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 40), scheme_3052, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 474)
            subscript_call_result_3054 = invoke(stypy.reporting.localization.Localization(__file__, 474, 40), getitem___3053, key_3051)
            
            # Processing the call keyword arguments (line 474)
            kwargs_3055 = {}
            # Getting the type of 'setattr' (line 474)
            setattr_3048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 16), 'setattr', False)
            # Calling setattr(args, kwargs) (line 474)
            setattr_call_result_3056 = invoke(stypy.reporting.localization.Localization(__file__, 474, 16), setattr_3048, *[self_3049, attrname_3050, subscript_call_result_3054], **kwargs_3055)
            

            if more_types_in_union_3047:
                # SSA join for if statement (line 473)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'select_scheme(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'select_scheme' in the type store
        # Getting the type of 'stypy_return_type' (line 468)
        stypy_return_type_3057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3057)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'select_scheme'
        return stypy_return_type_3057


    @norecursion
    def _expand_attrs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_expand_attrs'
        module_type_store = module_type_store.open_function_context('_expand_attrs', 477, 4, False)
        # Assigning a type to the variable 'self' (line 478)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install._expand_attrs.__dict__.__setitem__('stypy_localization', localization)
        install._expand_attrs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install._expand_attrs.__dict__.__setitem__('stypy_type_store', module_type_store)
        install._expand_attrs.__dict__.__setitem__('stypy_function_name', 'install._expand_attrs')
        install._expand_attrs.__dict__.__setitem__('stypy_param_names_list', ['attrs'])
        install._expand_attrs.__dict__.__setitem__('stypy_varargs_param_name', None)
        install._expand_attrs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install._expand_attrs.__dict__.__setitem__('stypy_call_defaults', defaults)
        install._expand_attrs.__dict__.__setitem__('stypy_call_varargs', varargs)
        install._expand_attrs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install._expand_attrs.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install._expand_attrs', ['attrs'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_expand_attrs', localization, ['attrs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_expand_attrs(...)' code ##################

        
        # Getting the type of 'attrs' (line 478)
        attrs_3058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 20), 'attrs')
        # Testing the type of a for loop iterable (line 478)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 478, 8), attrs_3058)
        # Getting the type of the for loop variable (line 478)
        for_loop_var_3059 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 478, 8), attrs_3058)
        # Assigning a type to the variable 'attr' (line 478)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'attr', for_loop_var_3059)
        # SSA begins for a for statement (line 478)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 479):
        
        # Assigning a Call to a Name (line 479):
        
        # Call to getattr(...): (line 479)
        # Processing the call arguments (line 479)
        # Getting the type of 'self' (line 479)
        self_3061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 26), 'self', False)
        # Getting the type of 'attr' (line 479)
        attr_3062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 32), 'attr', False)
        # Processing the call keyword arguments (line 479)
        kwargs_3063 = {}
        # Getting the type of 'getattr' (line 479)
        getattr_3060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 18), 'getattr', False)
        # Calling getattr(args, kwargs) (line 479)
        getattr_call_result_3064 = invoke(stypy.reporting.localization.Localization(__file__, 479, 18), getattr_3060, *[self_3061, attr_3062], **kwargs_3063)
        
        # Assigning a type to the variable 'val' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'val', getattr_call_result_3064)
        
        # Type idiom detected: calculating its left and rigth part (line 480)
        # Getting the type of 'val' (line 480)
        val_3065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 12), 'val')
        # Getting the type of 'None' (line 480)
        None_3066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 26), 'None')
        
        (may_be_3067, more_types_in_union_3068) = may_not_be_none(val_3065, None_3066)

        if may_be_3067:

            if more_types_in_union_3068:
                # Runtime conditional SSA (line 480)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'os' (line 481)
            os_3069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 19), 'os')
            # Obtaining the member 'name' of a type (line 481)
            name_3070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 19), os_3069, 'name')
            str_3071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 30), 'str', 'posix')
            # Applying the binary operator '==' (line 481)
            result_eq_3072 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 19), '==', name_3070, str_3071)
            
            
            # Getting the type of 'os' (line 481)
            os_3073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 41), 'os')
            # Obtaining the member 'name' of a type (line 481)
            name_3074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 41), os_3073, 'name')
            str_3075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 52), 'str', 'nt')
            # Applying the binary operator '==' (line 481)
            result_eq_3076 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 41), '==', name_3074, str_3075)
            
            # Applying the binary operator 'or' (line 481)
            result_or_keyword_3077 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 19), 'or', result_eq_3072, result_eq_3076)
            
            # Testing the type of an if condition (line 481)
            if_condition_3078 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 481, 16), result_or_keyword_3077)
            # Assigning a type to the variable 'if_condition_3078' (line 481)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 16), 'if_condition_3078', if_condition_3078)
            # SSA begins for if statement (line 481)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 482):
            
            # Assigning a Call to a Name (line 482):
            
            # Call to expanduser(...): (line 482)
            # Processing the call arguments (line 482)
            # Getting the type of 'val' (line 482)
            val_3082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 45), 'val', False)
            # Processing the call keyword arguments (line 482)
            kwargs_3083 = {}
            # Getting the type of 'os' (line 482)
            os_3079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 26), 'os', False)
            # Obtaining the member 'path' of a type (line 482)
            path_3080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 26), os_3079, 'path')
            # Obtaining the member 'expanduser' of a type (line 482)
            expanduser_3081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 26), path_3080, 'expanduser')
            # Calling expanduser(args, kwargs) (line 482)
            expanduser_call_result_3084 = invoke(stypy.reporting.localization.Localization(__file__, 482, 26), expanduser_3081, *[val_3082], **kwargs_3083)
            
            # Assigning a type to the variable 'val' (line 482)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 20), 'val', expanduser_call_result_3084)
            # SSA join for if statement (line 481)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 483):
            
            # Assigning a Call to a Name (line 483):
            
            # Call to subst_vars(...): (line 483)
            # Processing the call arguments (line 483)
            # Getting the type of 'val' (line 483)
            val_3086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 33), 'val', False)
            # Getting the type of 'self' (line 483)
            self_3087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 38), 'self', False)
            # Obtaining the member 'config_vars' of a type (line 483)
            config_vars_3088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 38), self_3087, 'config_vars')
            # Processing the call keyword arguments (line 483)
            kwargs_3089 = {}
            # Getting the type of 'subst_vars' (line 483)
            subst_vars_3085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 22), 'subst_vars', False)
            # Calling subst_vars(args, kwargs) (line 483)
            subst_vars_call_result_3090 = invoke(stypy.reporting.localization.Localization(__file__, 483, 22), subst_vars_3085, *[val_3086, config_vars_3088], **kwargs_3089)
            
            # Assigning a type to the variable 'val' (line 483)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 16), 'val', subst_vars_call_result_3090)
            
            # Call to setattr(...): (line 484)
            # Processing the call arguments (line 484)
            # Getting the type of 'self' (line 484)
            self_3092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 24), 'self', False)
            # Getting the type of 'attr' (line 484)
            attr_3093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 30), 'attr', False)
            # Getting the type of 'val' (line 484)
            val_3094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 36), 'val', False)
            # Processing the call keyword arguments (line 484)
            kwargs_3095 = {}
            # Getting the type of 'setattr' (line 484)
            setattr_3091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 16), 'setattr', False)
            # Calling setattr(args, kwargs) (line 484)
            setattr_call_result_3096 = invoke(stypy.reporting.localization.Localization(__file__, 484, 16), setattr_3091, *[self_3092, attr_3093, val_3094], **kwargs_3095)
            

            if more_types_in_union_3068:
                # SSA join for if statement (line 480)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_expand_attrs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_expand_attrs' in the type store
        # Getting the type of 'stypy_return_type' (line 477)
        stypy_return_type_3097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3097)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_expand_attrs'
        return stypy_return_type_3097


    @norecursion
    def expand_basedirs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'expand_basedirs'
        module_type_store = module_type_store.open_function_context('expand_basedirs', 487, 4, False)
        # Assigning a type to the variable 'self' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install.expand_basedirs.__dict__.__setitem__('stypy_localization', localization)
        install.expand_basedirs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install.expand_basedirs.__dict__.__setitem__('stypy_type_store', module_type_store)
        install.expand_basedirs.__dict__.__setitem__('stypy_function_name', 'install.expand_basedirs')
        install.expand_basedirs.__dict__.__setitem__('stypy_param_names_list', [])
        install.expand_basedirs.__dict__.__setitem__('stypy_varargs_param_name', None)
        install.expand_basedirs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install.expand_basedirs.__dict__.__setitem__('stypy_call_defaults', defaults)
        install.expand_basedirs.__dict__.__setitem__('stypy_call_varargs', varargs)
        install.expand_basedirs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install.expand_basedirs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install.expand_basedirs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'expand_basedirs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'expand_basedirs(...)' code ##################

        
        # Call to _expand_attrs(...): (line 488)
        # Processing the call arguments (line 488)
        
        # Obtaining an instance of the builtin type 'list' (line 488)
        list_3100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 488)
        # Adding element type (line 488)
        str_3101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 28), 'str', 'install_base')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 488, 27), list_3100, str_3101)
        # Adding element type (line 488)
        str_3102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 28), 'str', 'install_platbase')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 488, 27), list_3100, str_3102)
        # Adding element type (line 488)
        str_3103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 28), 'str', 'root')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 488, 27), list_3100, str_3103)
        
        # Processing the call keyword arguments (line 488)
        kwargs_3104 = {}
        # Getting the type of 'self' (line 488)
        self_3098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'self', False)
        # Obtaining the member '_expand_attrs' of a type (line 488)
        _expand_attrs_3099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 8), self_3098, '_expand_attrs')
        # Calling _expand_attrs(args, kwargs) (line 488)
        _expand_attrs_call_result_3105 = invoke(stypy.reporting.localization.Localization(__file__, 488, 8), _expand_attrs_3099, *[list_3100], **kwargs_3104)
        
        
        # ################# End of 'expand_basedirs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'expand_basedirs' in the type store
        # Getting the type of 'stypy_return_type' (line 487)
        stypy_return_type_3106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3106)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'expand_basedirs'
        return stypy_return_type_3106


    @norecursion
    def expand_dirs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'expand_dirs'
        module_type_store = module_type_store.open_function_context('expand_dirs', 492, 4, False)
        # Assigning a type to the variable 'self' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install.expand_dirs.__dict__.__setitem__('stypy_localization', localization)
        install.expand_dirs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install.expand_dirs.__dict__.__setitem__('stypy_type_store', module_type_store)
        install.expand_dirs.__dict__.__setitem__('stypy_function_name', 'install.expand_dirs')
        install.expand_dirs.__dict__.__setitem__('stypy_param_names_list', [])
        install.expand_dirs.__dict__.__setitem__('stypy_varargs_param_name', None)
        install.expand_dirs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install.expand_dirs.__dict__.__setitem__('stypy_call_defaults', defaults)
        install.expand_dirs.__dict__.__setitem__('stypy_call_varargs', varargs)
        install.expand_dirs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install.expand_dirs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install.expand_dirs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'expand_dirs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'expand_dirs(...)' code ##################

        
        # Call to _expand_attrs(...): (line 493)
        # Processing the call arguments (line 493)
        
        # Obtaining an instance of the builtin type 'list' (line 493)
        list_3109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 493)
        # Adding element type (line 493)
        str_3110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 28), 'str', 'install_purelib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 27), list_3109, str_3110)
        # Adding element type (line 493)
        str_3111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 28), 'str', 'install_platlib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 27), list_3109, str_3111)
        # Adding element type (line 493)
        str_3112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 28), 'str', 'install_lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 27), list_3109, str_3112)
        # Adding element type (line 493)
        str_3113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 28), 'str', 'install_headers')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 27), list_3109, str_3113)
        # Adding element type (line 493)
        str_3114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 28), 'str', 'install_scripts')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 27), list_3109, str_3114)
        # Adding element type (line 493)
        str_3115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 28), 'str', 'install_data')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 27), list_3109, str_3115)
        
        # Processing the call keyword arguments (line 493)
        kwargs_3116 = {}
        # Getting the type of 'self' (line 493)
        self_3107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'self', False)
        # Obtaining the member '_expand_attrs' of a type (line 493)
        _expand_attrs_3108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 8), self_3107, '_expand_attrs')
        # Calling _expand_attrs(args, kwargs) (line 493)
        _expand_attrs_call_result_3117 = invoke(stypy.reporting.localization.Localization(__file__, 493, 8), _expand_attrs_3108, *[list_3109], **kwargs_3116)
        
        
        # ################# End of 'expand_dirs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'expand_dirs' in the type store
        # Getting the type of 'stypy_return_type' (line 492)
        stypy_return_type_3118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3118)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'expand_dirs'
        return stypy_return_type_3118


    @norecursion
    def convert_paths(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'convert_paths'
        module_type_store = module_type_store.open_function_context('convert_paths', 501, 4, False)
        # Assigning a type to the variable 'self' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install.convert_paths.__dict__.__setitem__('stypy_localization', localization)
        install.convert_paths.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install.convert_paths.__dict__.__setitem__('stypy_type_store', module_type_store)
        install.convert_paths.__dict__.__setitem__('stypy_function_name', 'install.convert_paths')
        install.convert_paths.__dict__.__setitem__('stypy_param_names_list', [])
        install.convert_paths.__dict__.__setitem__('stypy_varargs_param_name', 'names')
        install.convert_paths.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install.convert_paths.__dict__.__setitem__('stypy_call_defaults', defaults)
        install.convert_paths.__dict__.__setitem__('stypy_call_varargs', varargs)
        install.convert_paths.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install.convert_paths.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install.convert_paths', [], 'names', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'convert_paths', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'convert_paths(...)' code ##################

        
        # Getting the type of 'names' (line 502)
        names_3119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 20), 'names')
        # Testing the type of a for loop iterable (line 502)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 502, 8), names_3119)
        # Getting the type of the for loop variable (line 502)
        for_loop_var_3120 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 502, 8), names_3119)
        # Assigning a type to the variable 'name' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'name', for_loop_var_3120)
        # SSA begins for a for statement (line 502)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 503):
        
        # Assigning a BinOp to a Name (line 503):
        str_3121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 19), 'str', 'install_')
        # Getting the type of 'name' (line 503)
        name_3122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 32), 'name')
        # Applying the binary operator '+' (line 503)
        result_add_3123 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 19), '+', str_3121, name_3122)
        
        # Assigning a type to the variable 'attr' (line 503)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 12), 'attr', result_add_3123)
        
        # Call to setattr(...): (line 504)
        # Processing the call arguments (line 504)
        # Getting the type of 'self' (line 504)
        self_3125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 20), 'self', False)
        # Getting the type of 'attr' (line 504)
        attr_3126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 26), 'attr', False)
        
        # Call to convert_path(...): (line 504)
        # Processing the call arguments (line 504)
        
        # Call to getattr(...): (line 504)
        # Processing the call arguments (line 504)
        # Getting the type of 'self' (line 504)
        self_3129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 53), 'self', False)
        # Getting the type of 'attr' (line 504)
        attr_3130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 59), 'attr', False)
        # Processing the call keyword arguments (line 504)
        kwargs_3131 = {}
        # Getting the type of 'getattr' (line 504)
        getattr_3128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 45), 'getattr', False)
        # Calling getattr(args, kwargs) (line 504)
        getattr_call_result_3132 = invoke(stypy.reporting.localization.Localization(__file__, 504, 45), getattr_3128, *[self_3129, attr_3130], **kwargs_3131)
        
        # Processing the call keyword arguments (line 504)
        kwargs_3133 = {}
        # Getting the type of 'convert_path' (line 504)
        convert_path_3127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 32), 'convert_path', False)
        # Calling convert_path(args, kwargs) (line 504)
        convert_path_call_result_3134 = invoke(stypy.reporting.localization.Localization(__file__, 504, 32), convert_path_3127, *[getattr_call_result_3132], **kwargs_3133)
        
        # Processing the call keyword arguments (line 504)
        kwargs_3135 = {}
        # Getting the type of 'setattr' (line 504)
        setattr_3124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'setattr', False)
        # Calling setattr(args, kwargs) (line 504)
        setattr_call_result_3136 = invoke(stypy.reporting.localization.Localization(__file__, 504, 12), setattr_3124, *[self_3125, attr_3126, convert_path_call_result_3134], **kwargs_3135)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'convert_paths(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'convert_paths' in the type store
        # Getting the type of 'stypy_return_type' (line 501)
        stypy_return_type_3137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3137)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'convert_paths'
        return stypy_return_type_3137


    @norecursion
    def handle_extra_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'handle_extra_path'
        module_type_store = module_type_store.open_function_context('handle_extra_path', 507, 4, False)
        # Assigning a type to the variable 'self' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install.handle_extra_path.__dict__.__setitem__('stypy_localization', localization)
        install.handle_extra_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install.handle_extra_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        install.handle_extra_path.__dict__.__setitem__('stypy_function_name', 'install.handle_extra_path')
        install.handle_extra_path.__dict__.__setitem__('stypy_param_names_list', [])
        install.handle_extra_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        install.handle_extra_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install.handle_extra_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        install.handle_extra_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        install.handle_extra_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install.handle_extra_path.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install.handle_extra_path', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'handle_extra_path', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'handle_extra_path(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 509)
        # Getting the type of 'self' (line 509)
        self_3138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 11), 'self')
        # Obtaining the member 'extra_path' of a type (line 509)
        extra_path_3139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 11), self_3138, 'extra_path')
        # Getting the type of 'None' (line 509)
        None_3140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 30), 'None')
        
        (may_be_3141, more_types_in_union_3142) = may_be_none(extra_path_3139, None_3140)

        if may_be_3141:

            if more_types_in_union_3142:
                # Runtime conditional SSA (line 509)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Attribute (line 510):
            
            # Assigning a Attribute to a Attribute (line 510):
            # Getting the type of 'self' (line 510)
            self_3143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 30), 'self')
            # Obtaining the member 'distribution' of a type (line 510)
            distribution_3144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 30), self_3143, 'distribution')
            # Obtaining the member 'extra_path' of a type (line 510)
            extra_path_3145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 30), distribution_3144, 'extra_path')
            # Getting the type of 'self' (line 510)
            self_3146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 12), 'self')
            # Setting the type of the member 'extra_path' of a type (line 510)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 12), self_3146, 'extra_path', extra_path_3145)

            if more_types_in_union_3142:
                # SSA join for if statement (line 509)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'self' (line 512)
        self_3147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 11), 'self')
        # Obtaining the member 'extra_path' of a type (line 512)
        extra_path_3148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 11), self_3147, 'extra_path')
        # Getting the type of 'None' (line 512)
        None_3149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 34), 'None')
        # Applying the binary operator 'isnot' (line 512)
        result_is_not_3150 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 11), 'isnot', extra_path_3148, None_3149)
        
        # Testing the type of an if condition (line 512)
        if_condition_3151 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 512, 8), result_is_not_3150)
        # Assigning a type to the variable 'if_condition_3151' (line 512)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'if_condition_3151', if_condition_3151)
        # SSA begins for if statement (line 512)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to type(...): (line 513)
        # Processing the call arguments (line 513)
        # Getting the type of 'self' (line 513)
        self_3153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 20), 'self', False)
        # Obtaining the member 'extra_path' of a type (line 513)
        extra_path_3154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 20), self_3153, 'extra_path')
        # Processing the call keyword arguments (line 513)
        kwargs_3155 = {}
        # Getting the type of 'type' (line 513)
        type_3152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 15), 'type', False)
        # Calling type(args, kwargs) (line 513)
        type_call_result_3156 = invoke(stypy.reporting.localization.Localization(__file__, 513, 15), type_3152, *[extra_path_3154], **kwargs_3155)
        
        # Getting the type of 'StringType' (line 513)
        StringType_3157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 40), 'StringType')
        # Applying the binary operator 'is' (line 513)
        result_is__3158 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 15), 'is', type_call_result_3156, StringType_3157)
        
        # Testing the type of an if condition (line 513)
        if_condition_3159 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 513, 12), result_is__3158)
        # Assigning a type to the variable 'if_condition_3159' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 12), 'if_condition_3159', if_condition_3159)
        # SSA begins for if statement (line 513)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 514):
        
        # Assigning a Call to a Attribute (line 514):
        
        # Call to split(...): (line 514)
        # Processing the call arguments (line 514)
        # Getting the type of 'self' (line 514)
        self_3162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 47), 'self', False)
        # Obtaining the member 'extra_path' of a type (line 514)
        extra_path_3163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 47), self_3162, 'extra_path')
        str_3164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 64), 'str', ',')
        # Processing the call keyword arguments (line 514)
        kwargs_3165 = {}
        # Getting the type of 'string' (line 514)
        string_3160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 34), 'string', False)
        # Obtaining the member 'split' of a type (line 514)
        split_3161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 34), string_3160, 'split')
        # Calling split(args, kwargs) (line 514)
        split_call_result_3166 = invoke(stypy.reporting.localization.Localization(__file__, 514, 34), split_3161, *[extra_path_3163, str_3164], **kwargs_3165)
        
        # Getting the type of 'self' (line 514)
        self_3167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 16), 'self')
        # Setting the type of the member 'extra_path' of a type (line 514)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 16), self_3167, 'extra_path', split_call_result_3166)
        # SSA join for if statement (line 513)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to len(...): (line 516)
        # Processing the call arguments (line 516)
        # Getting the type of 'self' (line 516)
        self_3169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 19), 'self', False)
        # Obtaining the member 'extra_path' of a type (line 516)
        extra_path_3170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 19), self_3169, 'extra_path')
        # Processing the call keyword arguments (line 516)
        kwargs_3171 = {}
        # Getting the type of 'len' (line 516)
        len_3168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 15), 'len', False)
        # Calling len(args, kwargs) (line 516)
        len_call_result_3172 = invoke(stypy.reporting.localization.Localization(__file__, 516, 15), len_3168, *[extra_path_3170], **kwargs_3171)
        
        int_3173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 39), 'int')
        # Applying the binary operator '==' (line 516)
        result_eq_3174 = python_operator(stypy.reporting.localization.Localization(__file__, 516, 15), '==', len_call_result_3172, int_3173)
        
        # Testing the type of an if condition (line 516)
        if_condition_3175 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 516, 12), result_eq_3174)
        # Assigning a type to the variable 'if_condition_3175' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 12), 'if_condition_3175', if_condition_3175)
        # SSA begins for if statement (line 516)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Subscript to a Name (line 517):
        
        # Obtaining the type of the subscript
        int_3176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 57), 'int')
        # Getting the type of 'self' (line 517)
        self_3177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 41), 'self')
        # Obtaining the member 'extra_path' of a type (line 517)
        extra_path_3178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 41), self_3177, 'extra_path')
        # Obtaining the member '__getitem__' of a type (line 517)
        getitem___3179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 41), extra_path_3178, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 517)
        subscript_call_result_3180 = invoke(stypy.reporting.localization.Localization(__file__, 517, 41), getitem___3179, int_3176)
        
        # Assigning a type to the variable 'extra_dirs' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 28), 'extra_dirs', subscript_call_result_3180)
        
        # Assigning a Name to a Name (line 517):
        # Getting the type of 'extra_dirs' (line 517)
        extra_dirs_3181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 28), 'extra_dirs')
        # Assigning a type to the variable 'path_file' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 16), 'path_file', extra_dirs_3181)
        # SSA branch for the else part of an if statement (line 516)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to len(...): (line 518)
        # Processing the call arguments (line 518)
        # Getting the type of 'self' (line 518)
        self_3183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 21), 'self', False)
        # Obtaining the member 'extra_path' of a type (line 518)
        extra_path_3184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 21), self_3183, 'extra_path')
        # Processing the call keyword arguments (line 518)
        kwargs_3185 = {}
        # Getting the type of 'len' (line 518)
        len_3182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 17), 'len', False)
        # Calling len(args, kwargs) (line 518)
        len_call_result_3186 = invoke(stypy.reporting.localization.Localization(__file__, 518, 17), len_3182, *[extra_path_3184], **kwargs_3185)
        
        int_3187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 41), 'int')
        # Applying the binary operator '==' (line 518)
        result_eq_3188 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 17), '==', len_call_result_3186, int_3187)
        
        # Testing the type of an if condition (line 518)
        if_condition_3189 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 518, 17), result_eq_3188)
        # Assigning a type to the variable 'if_condition_3189' (line 518)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 17), 'if_condition_3189', if_condition_3189)
        # SSA begins for if statement (line 518)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Tuple (line 519):
        
        # Assigning a Subscript to a Name (line 519):
        
        # Obtaining the type of the subscript
        int_3190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 16), 'int')
        # Getting the type of 'self' (line 519)
        self_3191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 42), 'self')
        # Obtaining the member 'extra_path' of a type (line 519)
        extra_path_3192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 42), self_3191, 'extra_path')
        # Obtaining the member '__getitem__' of a type (line 519)
        getitem___3193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 16), extra_path_3192, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 519)
        subscript_call_result_3194 = invoke(stypy.reporting.localization.Localization(__file__, 519, 16), getitem___3193, int_3190)
        
        # Assigning a type to the variable 'tuple_var_assignment_2304' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 16), 'tuple_var_assignment_2304', subscript_call_result_3194)
        
        # Assigning a Subscript to a Name (line 519):
        
        # Obtaining the type of the subscript
        int_3195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 16), 'int')
        # Getting the type of 'self' (line 519)
        self_3196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 42), 'self')
        # Obtaining the member 'extra_path' of a type (line 519)
        extra_path_3197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 42), self_3196, 'extra_path')
        # Obtaining the member '__getitem__' of a type (line 519)
        getitem___3198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 16), extra_path_3197, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 519)
        subscript_call_result_3199 = invoke(stypy.reporting.localization.Localization(__file__, 519, 16), getitem___3198, int_3195)
        
        # Assigning a type to the variable 'tuple_var_assignment_2305' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 16), 'tuple_var_assignment_2305', subscript_call_result_3199)
        
        # Assigning a Name to a Name (line 519):
        # Getting the type of 'tuple_var_assignment_2304' (line 519)
        tuple_var_assignment_2304_3200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 16), 'tuple_var_assignment_2304')
        # Assigning a type to the variable 'path_file' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 17), 'path_file', tuple_var_assignment_2304_3200)
        
        # Assigning a Name to a Name (line 519):
        # Getting the type of 'tuple_var_assignment_2305' (line 519)
        tuple_var_assignment_2305_3201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 16), 'tuple_var_assignment_2305')
        # Assigning a type to the variable 'extra_dirs' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 28), 'extra_dirs', tuple_var_assignment_2305_3201)
        # SSA branch for the else part of an if statement (line 518)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'DistutilsOptionError' (line 521)
        DistutilsOptionError_3202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 22), 'DistutilsOptionError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 521, 16), DistutilsOptionError_3202, 'raise parameter', BaseException)
        # SSA join for if statement (line 518)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 516)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 527):
        
        # Assigning a Call to a Name (line 527):
        
        # Call to convert_path(...): (line 527)
        # Processing the call arguments (line 527)
        # Getting the type of 'extra_dirs' (line 527)
        extra_dirs_3204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 38), 'extra_dirs', False)
        # Processing the call keyword arguments (line 527)
        kwargs_3205 = {}
        # Getting the type of 'convert_path' (line 527)
        convert_path_3203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 25), 'convert_path', False)
        # Calling convert_path(args, kwargs) (line 527)
        convert_path_call_result_3206 = invoke(stypy.reporting.localization.Localization(__file__, 527, 25), convert_path_3203, *[extra_dirs_3204], **kwargs_3205)
        
        # Assigning a type to the variable 'extra_dirs' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 12), 'extra_dirs', convert_path_call_result_3206)
        # SSA branch for the else part of an if statement (line 512)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 530):
        
        # Assigning a Name to a Name (line 530):
        # Getting the type of 'None' (line 530)
        None_3207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 24), 'None')
        # Assigning a type to the variable 'path_file' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'path_file', None_3207)
        
        # Assigning a Str to a Name (line 531):
        
        # Assigning a Str to a Name (line 531):
        str_3208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 25), 'str', '')
        # Assigning a type to the variable 'extra_dirs' (line 531)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 12), 'extra_dirs', str_3208)
        # SSA join for if statement (line 512)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 535):
        
        # Assigning a Name to a Attribute (line 535):
        # Getting the type of 'path_file' (line 535)
        path_file_3209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 25), 'path_file')
        # Getting the type of 'self' (line 535)
        self_3210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'self')
        # Setting the type of the member 'path_file' of a type (line 535)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 8), self_3210, 'path_file', path_file_3209)
        
        # Assigning a Name to a Attribute (line 536):
        
        # Assigning a Name to a Attribute (line 536):
        # Getting the type of 'extra_dirs' (line 536)
        extra_dirs_3211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 26), 'extra_dirs')
        # Getting the type of 'self' (line 536)
        self_3212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'self')
        # Setting the type of the member 'extra_dirs' of a type (line 536)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 8), self_3212, 'extra_dirs', extra_dirs_3211)
        
        # ################# End of 'handle_extra_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'handle_extra_path' in the type store
        # Getting the type of 'stypy_return_type' (line 507)
        stypy_return_type_3213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3213)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'handle_extra_path'
        return stypy_return_type_3213


    @norecursion
    def change_roots(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'change_roots'
        module_type_store = module_type_store.open_function_context('change_roots', 541, 4, False)
        # Assigning a type to the variable 'self' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install.change_roots.__dict__.__setitem__('stypy_localization', localization)
        install.change_roots.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install.change_roots.__dict__.__setitem__('stypy_type_store', module_type_store)
        install.change_roots.__dict__.__setitem__('stypy_function_name', 'install.change_roots')
        install.change_roots.__dict__.__setitem__('stypy_param_names_list', [])
        install.change_roots.__dict__.__setitem__('stypy_varargs_param_name', 'names')
        install.change_roots.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install.change_roots.__dict__.__setitem__('stypy_call_defaults', defaults)
        install.change_roots.__dict__.__setitem__('stypy_call_varargs', varargs)
        install.change_roots.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install.change_roots.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install.change_roots', [], 'names', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'change_roots', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'change_roots(...)' code ##################

        
        # Getting the type of 'names' (line 542)
        names_3214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 20), 'names')
        # Testing the type of a for loop iterable (line 542)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 542, 8), names_3214)
        # Getting the type of the for loop variable (line 542)
        for_loop_var_3215 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 542, 8), names_3214)
        # Assigning a type to the variable 'name' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'name', for_loop_var_3215)
        # SSA begins for a for statement (line 542)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 543):
        
        # Assigning a BinOp to a Name (line 543):
        str_3216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 19), 'str', 'install_')
        # Getting the type of 'name' (line 543)
        name_3217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 32), 'name')
        # Applying the binary operator '+' (line 543)
        result_add_3218 = python_operator(stypy.reporting.localization.Localization(__file__, 543, 19), '+', str_3216, name_3217)
        
        # Assigning a type to the variable 'attr' (line 543)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 12), 'attr', result_add_3218)
        
        # Call to setattr(...): (line 544)
        # Processing the call arguments (line 544)
        # Getting the type of 'self' (line 544)
        self_3220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 20), 'self', False)
        # Getting the type of 'attr' (line 544)
        attr_3221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 26), 'attr', False)
        
        # Call to change_root(...): (line 544)
        # Processing the call arguments (line 544)
        # Getting the type of 'self' (line 544)
        self_3223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 44), 'self', False)
        # Obtaining the member 'root' of a type (line 544)
        root_3224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 44), self_3223, 'root')
        
        # Call to getattr(...): (line 544)
        # Processing the call arguments (line 544)
        # Getting the type of 'self' (line 544)
        self_3226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 63), 'self', False)
        # Getting the type of 'attr' (line 544)
        attr_3227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 69), 'attr', False)
        # Processing the call keyword arguments (line 544)
        kwargs_3228 = {}
        # Getting the type of 'getattr' (line 544)
        getattr_3225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 55), 'getattr', False)
        # Calling getattr(args, kwargs) (line 544)
        getattr_call_result_3229 = invoke(stypy.reporting.localization.Localization(__file__, 544, 55), getattr_3225, *[self_3226, attr_3227], **kwargs_3228)
        
        # Processing the call keyword arguments (line 544)
        kwargs_3230 = {}
        # Getting the type of 'change_root' (line 544)
        change_root_3222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 32), 'change_root', False)
        # Calling change_root(args, kwargs) (line 544)
        change_root_call_result_3231 = invoke(stypy.reporting.localization.Localization(__file__, 544, 32), change_root_3222, *[root_3224, getattr_call_result_3229], **kwargs_3230)
        
        # Processing the call keyword arguments (line 544)
        kwargs_3232 = {}
        # Getting the type of 'setattr' (line 544)
        setattr_3219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 12), 'setattr', False)
        # Calling setattr(args, kwargs) (line 544)
        setattr_call_result_3233 = invoke(stypy.reporting.localization.Localization(__file__, 544, 12), setattr_3219, *[self_3220, attr_3221, change_root_call_result_3231], **kwargs_3232)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'change_roots(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'change_roots' in the type store
        # Getting the type of 'stypy_return_type' (line 541)
        stypy_return_type_3234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3234)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'change_roots'
        return stypy_return_type_3234


    @norecursion
    def create_home_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_home_path'
        module_type_store = module_type_store.open_function_context('create_home_path', 546, 4, False)
        # Assigning a type to the variable 'self' (line 547)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install.create_home_path.__dict__.__setitem__('stypy_localization', localization)
        install.create_home_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install.create_home_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        install.create_home_path.__dict__.__setitem__('stypy_function_name', 'install.create_home_path')
        install.create_home_path.__dict__.__setitem__('stypy_param_names_list', [])
        install.create_home_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        install.create_home_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install.create_home_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        install.create_home_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        install.create_home_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install.create_home_path.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install.create_home_path', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_home_path', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_home_path(...)' code ##################

        str_3235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, (-1)), 'str', 'Create directories under ~\n        ')
        
        
        # Getting the type of 'self' (line 549)
        self_3236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 15), 'self')
        # Obtaining the member 'user' of a type (line 549)
        user_3237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 15), self_3236, 'user')
        # Applying the 'not' unary operator (line 549)
        result_not__3238 = python_operator(stypy.reporting.localization.Localization(__file__, 549, 11), 'not', user_3237)
        
        # Testing the type of an if condition (line 549)
        if_condition_3239 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 549, 8), result_not__3238)
        # Assigning a type to the variable 'if_condition_3239' (line 549)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 8), 'if_condition_3239', if_condition_3239)
        # SSA begins for if statement (line 549)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 549)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 551):
        
        # Assigning a Call to a Name (line 551):
        
        # Call to convert_path(...): (line 551)
        # Processing the call arguments (line 551)
        
        # Call to expanduser(...): (line 551)
        # Processing the call arguments (line 551)
        str_3244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 47), 'str', '~')
        # Processing the call keyword arguments (line 551)
        kwargs_3245 = {}
        # Getting the type of 'os' (line 551)
        os_3241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 28), 'os', False)
        # Obtaining the member 'path' of a type (line 551)
        path_3242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 28), os_3241, 'path')
        # Obtaining the member 'expanduser' of a type (line 551)
        expanduser_3243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 28), path_3242, 'expanduser')
        # Calling expanduser(args, kwargs) (line 551)
        expanduser_call_result_3246 = invoke(stypy.reporting.localization.Localization(__file__, 551, 28), expanduser_3243, *[str_3244], **kwargs_3245)
        
        # Processing the call keyword arguments (line 551)
        kwargs_3247 = {}
        # Getting the type of 'convert_path' (line 551)
        convert_path_3240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 15), 'convert_path', False)
        # Calling convert_path(args, kwargs) (line 551)
        convert_path_call_result_3248 = invoke(stypy.reporting.localization.Localization(__file__, 551, 15), convert_path_3240, *[expanduser_call_result_3246], **kwargs_3247)
        
        # Assigning a type to the variable 'home' (line 551)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), 'home', convert_path_call_result_3248)
        
        
        # Call to iteritems(...): (line 552)
        # Processing the call keyword arguments (line 552)
        kwargs_3252 = {}
        # Getting the type of 'self' (line 552)
        self_3249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 26), 'self', False)
        # Obtaining the member 'config_vars' of a type (line 552)
        config_vars_3250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 26), self_3249, 'config_vars')
        # Obtaining the member 'iteritems' of a type (line 552)
        iteritems_3251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 26), config_vars_3250, 'iteritems')
        # Calling iteritems(args, kwargs) (line 552)
        iteritems_call_result_3253 = invoke(stypy.reporting.localization.Localization(__file__, 552, 26), iteritems_3251, *[], **kwargs_3252)
        
        # Testing the type of a for loop iterable (line 552)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 552, 8), iteritems_call_result_3253)
        # Getting the type of the for loop variable (line 552)
        for_loop_var_3254 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 552, 8), iteritems_call_result_3253)
        # Assigning a type to the variable 'name' (line 552)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 552, 8), for_loop_var_3254))
        # Assigning a type to the variable 'path' (line 552)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 8), 'path', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 552, 8), for_loop_var_3254))
        # SSA begins for a for statement (line 552)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Call to startswith(...): (line 553)
        # Processing the call arguments (line 553)
        # Getting the type of 'home' (line 553)
        home_3257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 31), 'home', False)
        # Processing the call keyword arguments (line 553)
        kwargs_3258 = {}
        # Getting the type of 'path' (line 553)
        path_3255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 15), 'path', False)
        # Obtaining the member 'startswith' of a type (line 553)
        startswith_3256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 15), path_3255, 'startswith')
        # Calling startswith(args, kwargs) (line 553)
        startswith_call_result_3259 = invoke(stypy.reporting.localization.Localization(__file__, 553, 15), startswith_3256, *[home_3257], **kwargs_3258)
        
        
        
        # Call to isdir(...): (line 553)
        # Processing the call arguments (line 553)
        # Getting the type of 'path' (line 553)
        path_3263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 59), 'path', False)
        # Processing the call keyword arguments (line 553)
        kwargs_3264 = {}
        # Getting the type of 'os' (line 553)
        os_3260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 45), 'os', False)
        # Obtaining the member 'path' of a type (line 553)
        path_3261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 45), os_3260, 'path')
        # Obtaining the member 'isdir' of a type (line 553)
        isdir_3262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 45), path_3261, 'isdir')
        # Calling isdir(args, kwargs) (line 553)
        isdir_call_result_3265 = invoke(stypy.reporting.localization.Localization(__file__, 553, 45), isdir_3262, *[path_3263], **kwargs_3264)
        
        # Applying the 'not' unary operator (line 553)
        result_not__3266 = python_operator(stypy.reporting.localization.Localization(__file__, 553, 41), 'not', isdir_call_result_3265)
        
        # Applying the binary operator 'and' (line 553)
        result_and_keyword_3267 = python_operator(stypy.reporting.localization.Localization(__file__, 553, 15), 'and', startswith_call_result_3259, result_not__3266)
        
        # Testing the type of an if condition (line 553)
        if_condition_3268 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 553, 12), result_and_keyword_3267)
        # Assigning a type to the variable 'if_condition_3268' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 12), 'if_condition_3268', if_condition_3268)
        # SSA begins for if statement (line 553)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to debug_print(...): (line 554)
        # Processing the call arguments (line 554)
        str_3271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 33), 'str', "os.makedirs('%s', 0700)")
        # Getting the type of 'path' (line 554)
        path_3272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 61), 'path', False)
        # Applying the binary operator '%' (line 554)
        result_mod_3273 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 33), '%', str_3271, path_3272)
        
        # Processing the call keyword arguments (line 554)
        kwargs_3274 = {}
        # Getting the type of 'self' (line 554)
        self_3269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 16), 'self', False)
        # Obtaining the member 'debug_print' of a type (line 554)
        debug_print_3270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 16), self_3269, 'debug_print')
        # Calling debug_print(args, kwargs) (line 554)
        debug_print_call_result_3275 = invoke(stypy.reporting.localization.Localization(__file__, 554, 16), debug_print_3270, *[result_mod_3273], **kwargs_3274)
        
        
        # Call to makedirs(...): (line 555)
        # Processing the call arguments (line 555)
        # Getting the type of 'path' (line 555)
        path_3278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 28), 'path', False)
        int_3279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 34), 'int')
        # Processing the call keyword arguments (line 555)
        kwargs_3280 = {}
        # Getting the type of 'os' (line 555)
        os_3276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 16), 'os', False)
        # Obtaining the member 'makedirs' of a type (line 555)
        makedirs_3277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 16), os_3276, 'makedirs')
        # Calling makedirs(args, kwargs) (line 555)
        makedirs_call_result_3281 = invoke(stypy.reporting.localization.Localization(__file__, 555, 16), makedirs_3277, *[path_3278, int_3279], **kwargs_3280)
        
        # SSA join for if statement (line 553)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'create_home_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_home_path' in the type store
        # Getting the type of 'stypy_return_type' (line 546)
        stypy_return_type_3282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3282)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_home_path'
        return stypy_return_type_3282


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 559, 4, False)
        # Assigning a type to the variable 'self' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install.run.__dict__.__setitem__('stypy_localization', localization)
        install.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        install.run.__dict__.__setitem__('stypy_function_name', 'install.run')
        install.run.__dict__.__setitem__('stypy_param_names_list', [])
        install.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        install.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        install.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        install.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install.run', [], None, None, defaults, varargs, kwargs)

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

        
        
        # Getting the type of 'self' (line 562)
        self_3283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 15), 'self')
        # Obtaining the member 'skip_build' of a type (line 562)
        skip_build_3284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 15), self_3283, 'skip_build')
        # Applying the 'not' unary operator (line 562)
        result_not__3285 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 11), 'not', skip_build_3284)
        
        # Testing the type of an if condition (line 562)
        if_condition_3286 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 562, 8), result_not__3285)
        # Assigning a type to the variable 'if_condition_3286' (line 562)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'if_condition_3286', if_condition_3286)
        # SSA begins for if statement (line 562)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to run_command(...): (line 563)
        # Processing the call arguments (line 563)
        str_3289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 29), 'str', 'build')
        # Processing the call keyword arguments (line 563)
        kwargs_3290 = {}
        # Getting the type of 'self' (line 563)
        self_3287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 12), 'self', False)
        # Obtaining the member 'run_command' of a type (line 563)
        run_command_3288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 12), self_3287, 'run_command')
        # Calling run_command(args, kwargs) (line 563)
        run_command_call_result_3291 = invoke(stypy.reporting.localization.Localization(__file__, 563, 12), run_command_3288, *[str_3289], **kwargs_3290)
        
        
        # Assigning a Attribute to a Name (line 565):
        
        # Assigning a Attribute to a Name (line 565):
        
        # Call to get_command_obj(...): (line 565)
        # Processing the call arguments (line 565)
        str_3295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 59), 'str', 'build')
        # Processing the call keyword arguments (line 565)
        kwargs_3296 = {}
        # Getting the type of 'self' (line 565)
        self_3292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 25), 'self', False)
        # Obtaining the member 'distribution' of a type (line 565)
        distribution_3293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 25), self_3292, 'distribution')
        # Obtaining the member 'get_command_obj' of a type (line 565)
        get_command_obj_3294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 25), distribution_3293, 'get_command_obj')
        # Calling get_command_obj(args, kwargs) (line 565)
        get_command_obj_call_result_3297 = invoke(stypy.reporting.localization.Localization(__file__, 565, 25), get_command_obj_3294, *[str_3295], **kwargs_3296)
        
        # Obtaining the member 'plat_name' of a type (line 565)
        plat_name_3298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 25), get_command_obj_call_result_3297, 'plat_name')
        # Assigning a type to the variable 'build_plat' (line 565)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 12), 'build_plat', plat_name_3298)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 569)
        self_3299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 15), 'self')
        # Obtaining the member 'warn_dir' of a type (line 569)
        warn_dir_3300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 15), self_3299, 'warn_dir')
        
        # Getting the type of 'build_plat' (line 569)
        build_plat_3301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 33), 'build_plat')
        
        # Call to get_platform(...): (line 569)
        # Processing the call keyword arguments (line 569)
        kwargs_3303 = {}
        # Getting the type of 'get_platform' (line 569)
        get_platform_3302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 47), 'get_platform', False)
        # Calling get_platform(args, kwargs) (line 569)
        get_platform_call_result_3304 = invoke(stypy.reporting.localization.Localization(__file__, 569, 47), get_platform_3302, *[], **kwargs_3303)
        
        # Applying the binary operator '!=' (line 569)
        result_ne_3305 = python_operator(stypy.reporting.localization.Localization(__file__, 569, 33), '!=', build_plat_3301, get_platform_call_result_3304)
        
        # Applying the binary operator 'and' (line 569)
        result_and_keyword_3306 = python_operator(stypy.reporting.localization.Localization(__file__, 569, 15), 'and', warn_dir_3300, result_ne_3305)
        
        # Testing the type of an if condition (line 569)
        if_condition_3307 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 569, 12), result_and_keyword_3306)
        # Assigning a type to the variable 'if_condition_3307' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 12), 'if_condition_3307', if_condition_3307)
        # SSA begins for if statement (line 569)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to DistutilsPlatformError(...): (line 570)
        # Processing the call arguments (line 570)
        str_3309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 45), 'str', "Can't install when cross-compiling")
        # Processing the call keyword arguments (line 570)
        kwargs_3310 = {}
        # Getting the type of 'DistutilsPlatformError' (line 570)
        DistutilsPlatformError_3308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 22), 'DistutilsPlatformError', False)
        # Calling DistutilsPlatformError(args, kwargs) (line 570)
        DistutilsPlatformError_call_result_3311 = invoke(stypy.reporting.localization.Localization(__file__, 570, 22), DistutilsPlatformError_3308, *[str_3309], **kwargs_3310)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 570, 16), DistutilsPlatformError_call_result_3311, 'raise parameter', BaseException)
        # SSA join for if statement (line 569)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 562)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to get_sub_commands(...): (line 574)
        # Processing the call keyword arguments (line 574)
        kwargs_3314 = {}
        # Getting the type of 'self' (line 574)
        self_3312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 24), 'self', False)
        # Obtaining the member 'get_sub_commands' of a type (line 574)
        get_sub_commands_3313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 24), self_3312, 'get_sub_commands')
        # Calling get_sub_commands(args, kwargs) (line 574)
        get_sub_commands_call_result_3315 = invoke(stypy.reporting.localization.Localization(__file__, 574, 24), get_sub_commands_3313, *[], **kwargs_3314)
        
        # Testing the type of a for loop iterable (line 574)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 574, 8), get_sub_commands_call_result_3315)
        # Getting the type of the for loop variable (line 574)
        for_loop_var_3316 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 574, 8), get_sub_commands_call_result_3315)
        # Assigning a type to the variable 'cmd_name' (line 574)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'cmd_name', for_loop_var_3316)
        # SSA begins for a for statement (line 574)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to run_command(...): (line 575)
        # Processing the call arguments (line 575)
        # Getting the type of 'cmd_name' (line 575)
        cmd_name_3319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 29), 'cmd_name', False)
        # Processing the call keyword arguments (line 575)
        kwargs_3320 = {}
        # Getting the type of 'self' (line 575)
        self_3317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 12), 'self', False)
        # Obtaining the member 'run_command' of a type (line 575)
        run_command_3318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 12), self_3317, 'run_command')
        # Calling run_command(args, kwargs) (line 575)
        run_command_call_result_3321 = invoke(stypy.reporting.localization.Localization(__file__, 575, 12), run_command_3318, *[cmd_name_3319], **kwargs_3320)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 577)
        self_3322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 11), 'self')
        # Obtaining the member 'path_file' of a type (line 577)
        path_file_3323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 11), self_3322, 'path_file')
        # Testing the type of an if condition (line 577)
        if_condition_3324 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 577, 8), path_file_3323)
        # Assigning a type to the variable 'if_condition_3324' (line 577)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 8), 'if_condition_3324', if_condition_3324)
        # SSA begins for if statement (line 577)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to create_path_file(...): (line 578)
        # Processing the call keyword arguments (line 578)
        kwargs_3327 = {}
        # Getting the type of 'self' (line 578)
        self_3325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 12), 'self', False)
        # Obtaining the member 'create_path_file' of a type (line 578)
        create_path_file_3326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 12), self_3325, 'create_path_file')
        # Calling create_path_file(args, kwargs) (line 578)
        create_path_file_call_result_3328 = invoke(stypy.reporting.localization.Localization(__file__, 578, 12), create_path_file_3326, *[], **kwargs_3327)
        
        # SSA join for if statement (line 577)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 581)
        self_3329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 11), 'self')
        # Obtaining the member 'record' of a type (line 581)
        record_3330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 11), self_3329, 'record')
        # Testing the type of an if condition (line 581)
        if_condition_3331 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 581, 8), record_3330)
        # Assigning a type to the variable 'if_condition_3331' (line 581)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'if_condition_3331', if_condition_3331)
        # SSA begins for if statement (line 581)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 582):
        
        # Assigning a Call to a Name (line 582):
        
        # Call to get_outputs(...): (line 582)
        # Processing the call keyword arguments (line 582)
        kwargs_3334 = {}
        # Getting the type of 'self' (line 582)
        self_3332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 22), 'self', False)
        # Obtaining the member 'get_outputs' of a type (line 582)
        get_outputs_3333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 22), self_3332, 'get_outputs')
        # Calling get_outputs(args, kwargs) (line 582)
        get_outputs_call_result_3335 = invoke(stypy.reporting.localization.Localization(__file__, 582, 22), get_outputs_3333, *[], **kwargs_3334)
        
        # Assigning a type to the variable 'outputs' (line 582)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 12), 'outputs', get_outputs_call_result_3335)
        
        # Getting the type of 'self' (line 583)
        self_3336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 15), 'self')
        # Obtaining the member 'root' of a type (line 583)
        root_3337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 15), self_3336, 'root')
        # Testing the type of an if condition (line 583)
        if_condition_3338 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 583, 12), root_3337)
        # Assigning a type to the variable 'if_condition_3338' (line 583)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 12), 'if_condition_3338', if_condition_3338)
        # SSA begins for if statement (line 583)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 584):
        
        # Assigning a Call to a Name (line 584):
        
        # Call to len(...): (line 584)
        # Processing the call arguments (line 584)
        # Getting the type of 'self' (line 584)
        self_3340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 31), 'self', False)
        # Obtaining the member 'root' of a type (line 584)
        root_3341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 31), self_3340, 'root')
        # Processing the call keyword arguments (line 584)
        kwargs_3342 = {}
        # Getting the type of 'len' (line 584)
        len_3339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 27), 'len', False)
        # Calling len(args, kwargs) (line 584)
        len_call_result_3343 = invoke(stypy.reporting.localization.Localization(__file__, 584, 27), len_3339, *[root_3341], **kwargs_3342)
        
        # Assigning a type to the variable 'root_len' (line 584)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 16), 'root_len', len_call_result_3343)
        
        
        # Call to xrange(...): (line 585)
        # Processing the call arguments (line 585)
        
        # Call to len(...): (line 585)
        # Processing the call arguments (line 585)
        # Getting the type of 'outputs' (line 585)
        outputs_3346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 42), 'outputs', False)
        # Processing the call keyword arguments (line 585)
        kwargs_3347 = {}
        # Getting the type of 'len' (line 585)
        len_3345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 38), 'len', False)
        # Calling len(args, kwargs) (line 585)
        len_call_result_3348 = invoke(stypy.reporting.localization.Localization(__file__, 585, 38), len_3345, *[outputs_3346], **kwargs_3347)
        
        # Processing the call keyword arguments (line 585)
        kwargs_3349 = {}
        # Getting the type of 'xrange' (line 585)
        xrange_3344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 31), 'xrange', False)
        # Calling xrange(args, kwargs) (line 585)
        xrange_call_result_3350 = invoke(stypy.reporting.localization.Localization(__file__, 585, 31), xrange_3344, *[len_call_result_3348], **kwargs_3349)
        
        # Testing the type of a for loop iterable (line 585)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 585, 16), xrange_call_result_3350)
        # Getting the type of the for loop variable (line 585)
        for_loop_var_3351 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 585, 16), xrange_call_result_3350)
        # Assigning a type to the variable 'counter' (line 585)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 16), 'counter', for_loop_var_3351)
        # SSA begins for a for statement (line 585)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Subscript (line 586):
        
        # Assigning a Subscript to a Subscript (line 586):
        
        # Obtaining the type of the subscript
        # Getting the type of 'root_len' (line 586)
        root_len_3352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 56), 'root_len')
        slice_3353 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 586, 39), root_len_3352, None, None)
        
        # Obtaining the type of the subscript
        # Getting the type of 'counter' (line 586)
        counter_3354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 47), 'counter')
        # Getting the type of 'outputs' (line 586)
        outputs_3355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 39), 'outputs')
        # Obtaining the member '__getitem__' of a type (line 586)
        getitem___3356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 39), outputs_3355, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 586)
        subscript_call_result_3357 = invoke(stypy.reporting.localization.Localization(__file__, 586, 39), getitem___3356, counter_3354)
        
        # Obtaining the member '__getitem__' of a type (line 586)
        getitem___3358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 39), subscript_call_result_3357, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 586)
        subscript_call_result_3359 = invoke(stypy.reporting.localization.Localization(__file__, 586, 39), getitem___3358, slice_3353)
        
        # Getting the type of 'outputs' (line 586)
        outputs_3360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 20), 'outputs')
        # Getting the type of 'counter' (line 586)
        counter_3361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 28), 'counter')
        # Storing an element on a container (line 586)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 586, 20), outputs_3360, (counter_3361, subscript_call_result_3359))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 583)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to execute(...): (line 587)
        # Processing the call arguments (line 587)
        # Getting the type of 'write_file' (line 587)
        write_file_3364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 25), 'write_file', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 588)
        tuple_3365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 588)
        # Adding element type (line 588)
        # Getting the type of 'self' (line 588)
        self_3366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 26), 'self', False)
        # Obtaining the member 'record' of a type (line 588)
        record_3367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 26), self_3366, 'record')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 588, 26), tuple_3365, record_3367)
        # Adding element type (line 588)
        # Getting the type of 'outputs' (line 588)
        outputs_3368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 39), 'outputs', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 588, 26), tuple_3365, outputs_3368)
        
        str_3369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 25), 'str', "writing list of installed files to '%s'")
        # Getting the type of 'self' (line 590)
        self_3370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 25), 'self', False)
        # Obtaining the member 'record' of a type (line 590)
        record_3371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 25), self_3370, 'record')
        # Applying the binary operator '%' (line 589)
        result_mod_3372 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 25), '%', str_3369, record_3371)
        
        # Processing the call keyword arguments (line 587)
        kwargs_3373 = {}
        # Getting the type of 'self' (line 587)
        self_3362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 12), 'self', False)
        # Obtaining the member 'execute' of a type (line 587)
        execute_3363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 12), self_3362, 'execute')
        # Calling execute(args, kwargs) (line 587)
        execute_call_result_3374 = invoke(stypy.reporting.localization.Localization(__file__, 587, 12), execute_3363, *[write_file_3364, tuple_3365, result_mod_3372], **kwargs_3373)
        
        # SSA join for if statement (line 581)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 592):
        
        # Assigning a Call to a Name (line 592):
        
        # Call to map(...): (line 592)
        # Processing the call arguments (line 592)
        # Getting the type of 'os' (line 592)
        os_3376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 592)
        path_3377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 23), os_3376, 'path')
        # Obtaining the member 'normpath' of a type (line 592)
        normpath_3378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 23), path_3377, 'normpath')
        # Getting the type of 'sys' (line 592)
        sys_3379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 41), 'sys', False)
        # Obtaining the member 'path' of a type (line 592)
        path_3380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 41), sys_3379, 'path')
        # Processing the call keyword arguments (line 592)
        kwargs_3381 = {}
        # Getting the type of 'map' (line 592)
        map_3375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 19), 'map', False)
        # Calling map(args, kwargs) (line 592)
        map_call_result_3382 = invoke(stypy.reporting.localization.Localization(__file__, 592, 19), map_3375, *[normpath_3378, path_3380], **kwargs_3381)
        
        # Assigning a type to the variable 'sys_path' (line 592)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'sys_path', map_call_result_3382)
        
        # Assigning a Call to a Name (line 593):
        
        # Assigning a Call to a Name (line 593):
        
        # Call to map(...): (line 593)
        # Processing the call arguments (line 593)
        # Getting the type of 'os' (line 593)
        os_3384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 593)
        path_3385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 23), os_3384, 'path')
        # Obtaining the member 'normcase' of a type (line 593)
        normcase_3386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 23), path_3385, 'normcase')
        # Getting the type of 'sys_path' (line 593)
        sys_path_3387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 41), 'sys_path', False)
        # Processing the call keyword arguments (line 593)
        kwargs_3388 = {}
        # Getting the type of 'map' (line 593)
        map_3383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 19), 'map', False)
        # Calling map(args, kwargs) (line 593)
        map_call_result_3389 = invoke(stypy.reporting.localization.Localization(__file__, 593, 19), map_3383, *[normcase_3386, sys_path_3387], **kwargs_3388)
        
        # Assigning a type to the variable 'sys_path' (line 593)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'sys_path', map_call_result_3389)
        
        # Assigning a Call to a Name (line 594):
        
        # Assigning a Call to a Name (line 594):
        
        # Call to normcase(...): (line 594)
        # Processing the call arguments (line 594)
        
        # Call to normpath(...): (line 594)
        # Processing the call arguments (line 594)
        # Getting the type of 'self' (line 594)
        self_3396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 56), 'self', False)
        # Obtaining the member 'install_lib' of a type (line 594)
        install_lib_3397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 56), self_3396, 'install_lib')
        # Processing the call keyword arguments (line 594)
        kwargs_3398 = {}
        # Getting the type of 'os' (line 594)
        os_3393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 39), 'os', False)
        # Obtaining the member 'path' of a type (line 594)
        path_3394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 39), os_3393, 'path')
        # Obtaining the member 'normpath' of a type (line 594)
        normpath_3395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 39), path_3394, 'normpath')
        # Calling normpath(args, kwargs) (line 594)
        normpath_call_result_3399 = invoke(stypy.reporting.localization.Localization(__file__, 594, 39), normpath_3395, *[install_lib_3397], **kwargs_3398)
        
        # Processing the call keyword arguments (line 594)
        kwargs_3400 = {}
        # Getting the type of 'os' (line 594)
        os_3390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 22), 'os', False)
        # Obtaining the member 'path' of a type (line 594)
        path_3391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 22), os_3390, 'path')
        # Obtaining the member 'normcase' of a type (line 594)
        normcase_3392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 22), path_3391, 'normcase')
        # Calling normcase(args, kwargs) (line 594)
        normcase_call_result_3401 = invoke(stypy.reporting.localization.Localization(__file__, 594, 22), normcase_3392, *[normpath_call_result_3399], **kwargs_3400)
        
        # Assigning a type to the variable 'install_lib' (line 594)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'install_lib', normcase_call_result_3401)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 595)
        self_3402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 12), 'self')
        # Obtaining the member 'warn_dir' of a type (line 595)
        warn_dir_3403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 12), self_3402, 'warn_dir')
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 596)
        self_3404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 17), 'self')
        # Obtaining the member 'path_file' of a type (line 596)
        path_file_3405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 17), self_3404, 'path_file')
        # Getting the type of 'self' (line 596)
        self_3406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 36), 'self')
        # Obtaining the member 'install_path_file' of a type (line 596)
        install_path_file_3407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 36), self_3406, 'install_path_file')
        # Applying the binary operator 'and' (line 596)
        result_and_keyword_3408 = python_operator(stypy.reporting.localization.Localization(__file__, 596, 17), 'and', path_file_3405, install_path_file_3407)
        
        # Applying the 'not' unary operator (line 596)
        result_not__3409 = python_operator(stypy.reporting.localization.Localization(__file__, 596, 12), 'not', result_and_keyword_3408)
        
        # Applying the binary operator 'and' (line 595)
        result_and_keyword_3410 = python_operator(stypy.reporting.localization.Localization(__file__, 595, 12), 'and', warn_dir_3403, result_not__3409)
        
        # Getting the type of 'install_lib' (line 597)
        install_lib_3411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 12), 'install_lib')
        # Getting the type of 'sys_path' (line 597)
        sys_path_3412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 31), 'sys_path')
        # Applying the binary operator 'notin' (line 597)
        result_contains_3413 = python_operator(stypy.reporting.localization.Localization(__file__, 597, 12), 'notin', install_lib_3411, sys_path_3412)
        
        # Applying the binary operator 'and' (line 595)
        result_and_keyword_3414 = python_operator(stypy.reporting.localization.Localization(__file__, 595, 12), 'and', result_and_keyword_3410, result_contains_3413)
        
        # Testing the type of an if condition (line 595)
        if_condition_3415 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 595, 8), result_and_keyword_3414)
        # Assigning a type to the variable 'if_condition_3415' (line 595)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 8), 'if_condition_3415', if_condition_3415)
        # SSA begins for if statement (line 595)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to debug(...): (line 598)
        # Processing the call arguments (line 598)
        str_3418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 23), 'str', "modules installed to '%s', which is not in Python's module search path (sys.path) -- you'll have to change the search path yourself")
        # Getting the type of 'self' (line 601)
        self_3419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 23), 'self', False)
        # Obtaining the member 'install_lib' of a type (line 601)
        install_lib_3420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 23), self_3419, 'install_lib')
        # Processing the call keyword arguments (line 598)
        kwargs_3421 = {}
        # Getting the type of 'log' (line 598)
        log_3416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 12), 'log', False)
        # Obtaining the member 'debug' of a type (line 598)
        debug_3417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 12), log_3416, 'debug')
        # Calling debug(args, kwargs) (line 598)
        debug_call_result_3422 = invoke(stypy.reporting.localization.Localization(__file__, 598, 12), debug_3417, *[str_3418, install_lib_3420], **kwargs_3421)
        
        # SSA join for if statement (line 595)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 559)
        stypy_return_type_3423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3423)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_3423


    @norecursion
    def create_path_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_path_file'
        module_type_store = module_type_store.open_function_context('create_path_file', 605, 4, False)
        # Assigning a type to the variable 'self' (line 606)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install.create_path_file.__dict__.__setitem__('stypy_localization', localization)
        install.create_path_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install.create_path_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        install.create_path_file.__dict__.__setitem__('stypy_function_name', 'install.create_path_file')
        install.create_path_file.__dict__.__setitem__('stypy_param_names_list', [])
        install.create_path_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        install.create_path_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install.create_path_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        install.create_path_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        install.create_path_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install.create_path_file.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install.create_path_file', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_path_file', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_path_file(...)' code ##################

        
        # Assigning a Call to a Name (line 606):
        
        # Assigning a Call to a Name (line 606):
        
        # Call to join(...): (line 606)
        # Processing the call arguments (line 606)
        # Getting the type of 'self' (line 606)
        self_3427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 32), 'self', False)
        # Obtaining the member 'install_libbase' of a type (line 606)
        install_libbase_3428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 32), self_3427, 'install_libbase')
        # Getting the type of 'self' (line 607)
        self_3429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 32), 'self', False)
        # Obtaining the member 'path_file' of a type (line 607)
        path_file_3430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 32), self_3429, 'path_file')
        str_3431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 49), 'str', '.pth')
        # Applying the binary operator '+' (line 607)
        result_add_3432 = python_operator(stypy.reporting.localization.Localization(__file__, 607, 32), '+', path_file_3430, str_3431)
        
        # Processing the call keyword arguments (line 606)
        kwargs_3433 = {}
        # Getting the type of 'os' (line 606)
        os_3424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 606)
        path_3425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 19), os_3424, 'path')
        # Obtaining the member 'join' of a type (line 606)
        join_3426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 19), path_3425, 'join')
        # Calling join(args, kwargs) (line 606)
        join_call_result_3434 = invoke(stypy.reporting.localization.Localization(__file__, 606, 19), join_3426, *[install_libbase_3428, result_add_3432], **kwargs_3433)
        
        # Assigning a type to the variable 'filename' (line 606)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 8), 'filename', join_call_result_3434)
        
        # Getting the type of 'self' (line 608)
        self_3435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 11), 'self')
        # Obtaining the member 'install_path_file' of a type (line 608)
        install_path_file_3436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 11), self_3435, 'install_path_file')
        # Testing the type of an if condition (line 608)
        if_condition_3437 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 608, 8), install_path_file_3436)
        # Assigning a type to the variable 'if_condition_3437' (line 608)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 8), 'if_condition_3437', if_condition_3437)
        # SSA begins for if statement (line 608)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to execute(...): (line 609)
        # Processing the call arguments (line 609)
        # Getting the type of 'write_file' (line 609)
        write_file_3440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 25), 'write_file', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 610)
        tuple_3441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 610)
        # Adding element type (line 610)
        # Getting the type of 'filename' (line 610)
        filename_3442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 26), 'filename', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 610, 26), tuple_3441, filename_3442)
        # Adding element type (line 610)
        
        # Obtaining an instance of the builtin type 'list' (line 610)
        list_3443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 610)
        # Adding element type (line 610)
        # Getting the type of 'self' (line 610)
        self_3444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 37), 'self', False)
        # Obtaining the member 'extra_dirs' of a type (line 610)
        extra_dirs_3445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 37), self_3444, 'extra_dirs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 610, 36), list_3443, extra_dirs_3445)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 610, 26), tuple_3441, list_3443)
        
        str_3446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 25), 'str', 'creating %s')
        # Getting the type of 'filename' (line 611)
        filename_3447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 41), 'filename', False)
        # Applying the binary operator '%' (line 611)
        result_mod_3448 = python_operator(stypy.reporting.localization.Localization(__file__, 611, 25), '%', str_3446, filename_3447)
        
        # Processing the call keyword arguments (line 609)
        kwargs_3449 = {}
        # Getting the type of 'self' (line 609)
        self_3438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 12), 'self', False)
        # Obtaining the member 'execute' of a type (line 609)
        execute_3439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 12), self_3438, 'execute')
        # Calling execute(args, kwargs) (line 609)
        execute_call_result_3450 = invoke(stypy.reporting.localization.Localization(__file__, 609, 12), execute_3439, *[write_file_3440, tuple_3441, result_mod_3448], **kwargs_3449)
        
        # SSA branch for the else part of an if statement (line 608)
        module_type_store.open_ssa_branch('else')
        
        # Call to warn(...): (line 613)
        # Processing the call arguments (line 613)
        str_3453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 22), 'str', "path file '%s' not created")
        # Getting the type of 'filename' (line 613)
        filename_3454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 53), 'filename', False)
        # Applying the binary operator '%' (line 613)
        result_mod_3455 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 22), '%', str_3453, filename_3454)
        
        # Processing the call keyword arguments (line 613)
        kwargs_3456 = {}
        # Getting the type of 'self' (line 613)
        self_3451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 12), 'self', False)
        # Obtaining the member 'warn' of a type (line 613)
        warn_3452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 12), self_3451, 'warn')
        # Calling warn(args, kwargs) (line 613)
        warn_call_result_3457 = invoke(stypy.reporting.localization.Localization(__file__, 613, 12), warn_3452, *[result_mod_3455], **kwargs_3456)
        
        # SSA join for if statement (line 608)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'create_path_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_path_file' in the type store
        # Getting the type of 'stypy_return_type' (line 605)
        stypy_return_type_3458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3458)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_path_file'
        return stypy_return_type_3458


    @norecursion
    def get_outputs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_outputs'
        module_type_store = module_type_store.open_function_context('get_outputs', 618, 4, False)
        # Assigning a type to the variable 'self' (line 619)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install.get_outputs.__dict__.__setitem__('stypy_localization', localization)
        install.get_outputs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install.get_outputs.__dict__.__setitem__('stypy_type_store', module_type_store)
        install.get_outputs.__dict__.__setitem__('stypy_function_name', 'install.get_outputs')
        install.get_outputs.__dict__.__setitem__('stypy_param_names_list', [])
        install.get_outputs.__dict__.__setitem__('stypy_varargs_param_name', None)
        install.get_outputs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install.get_outputs.__dict__.__setitem__('stypy_call_defaults', defaults)
        install.get_outputs.__dict__.__setitem__('stypy_call_varargs', varargs)
        install.get_outputs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install.get_outputs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install.get_outputs', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a List to a Name (line 620):
        
        # Assigning a List to a Name (line 620):
        
        # Obtaining an instance of the builtin type 'list' (line 620)
        list_3459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 620)
        
        # Assigning a type to the variable 'outputs' (line 620)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 8), 'outputs', list_3459)
        
        
        # Call to get_sub_commands(...): (line 621)
        # Processing the call keyword arguments (line 621)
        kwargs_3462 = {}
        # Getting the type of 'self' (line 621)
        self_3460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 24), 'self', False)
        # Obtaining the member 'get_sub_commands' of a type (line 621)
        get_sub_commands_3461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 24), self_3460, 'get_sub_commands')
        # Calling get_sub_commands(args, kwargs) (line 621)
        get_sub_commands_call_result_3463 = invoke(stypy.reporting.localization.Localization(__file__, 621, 24), get_sub_commands_3461, *[], **kwargs_3462)
        
        # Testing the type of a for loop iterable (line 621)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 621, 8), get_sub_commands_call_result_3463)
        # Getting the type of the for loop variable (line 621)
        for_loop_var_3464 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 621, 8), get_sub_commands_call_result_3463)
        # Assigning a type to the variable 'cmd_name' (line 621)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 8), 'cmd_name', for_loop_var_3464)
        # SSA begins for a for statement (line 621)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 622):
        
        # Assigning a Call to a Name (line 622):
        
        # Call to get_finalized_command(...): (line 622)
        # Processing the call arguments (line 622)
        # Getting the type of 'cmd_name' (line 622)
        cmd_name_3467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 45), 'cmd_name', False)
        # Processing the call keyword arguments (line 622)
        kwargs_3468 = {}
        # Getting the type of 'self' (line 622)
        self_3465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 18), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 622)
        get_finalized_command_3466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 18), self_3465, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 622)
        get_finalized_command_call_result_3469 = invoke(stypy.reporting.localization.Localization(__file__, 622, 18), get_finalized_command_3466, *[cmd_name_3467], **kwargs_3468)
        
        # Assigning a type to the variable 'cmd' (line 622)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 12), 'cmd', get_finalized_command_call_result_3469)
        
        
        # Call to get_outputs(...): (line 625)
        # Processing the call keyword arguments (line 625)
        kwargs_3472 = {}
        # Getting the type of 'cmd' (line 625)
        cmd_3470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 28), 'cmd', False)
        # Obtaining the member 'get_outputs' of a type (line 625)
        get_outputs_3471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 28), cmd_3470, 'get_outputs')
        # Calling get_outputs(args, kwargs) (line 625)
        get_outputs_call_result_3473 = invoke(stypy.reporting.localization.Localization(__file__, 625, 28), get_outputs_3471, *[], **kwargs_3472)
        
        # Testing the type of a for loop iterable (line 625)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 625, 12), get_outputs_call_result_3473)
        # Getting the type of the for loop variable (line 625)
        for_loop_var_3474 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 625, 12), get_outputs_call_result_3473)
        # Assigning a type to the variable 'filename' (line 625)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 12), 'filename', for_loop_var_3474)
        # SSA begins for a for statement (line 625)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'filename' (line 626)
        filename_3475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 19), 'filename')
        # Getting the type of 'outputs' (line 626)
        outputs_3476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 35), 'outputs')
        # Applying the binary operator 'notin' (line 626)
        result_contains_3477 = python_operator(stypy.reporting.localization.Localization(__file__, 626, 19), 'notin', filename_3475, outputs_3476)
        
        # Testing the type of an if condition (line 626)
        if_condition_3478 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 626, 16), result_contains_3477)
        # Assigning a type to the variable 'if_condition_3478' (line 626)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 16), 'if_condition_3478', if_condition_3478)
        # SSA begins for if statement (line 626)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 627)
        # Processing the call arguments (line 627)
        # Getting the type of 'filename' (line 627)
        filename_3481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 35), 'filename', False)
        # Processing the call keyword arguments (line 627)
        kwargs_3482 = {}
        # Getting the type of 'outputs' (line 627)
        outputs_3479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 20), 'outputs', False)
        # Obtaining the member 'append' of a type (line 627)
        append_3480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 20), outputs_3479, 'append')
        # Calling append(args, kwargs) (line 627)
        append_call_result_3483 = invoke(stypy.reporting.localization.Localization(__file__, 627, 20), append_3480, *[filename_3481], **kwargs_3482)
        
        # SSA join for if statement (line 626)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 629)
        self_3484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 11), 'self')
        # Obtaining the member 'path_file' of a type (line 629)
        path_file_3485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 11), self_3484, 'path_file')
        # Getting the type of 'self' (line 629)
        self_3486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 30), 'self')
        # Obtaining the member 'install_path_file' of a type (line 629)
        install_path_file_3487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 30), self_3486, 'install_path_file')
        # Applying the binary operator 'and' (line 629)
        result_and_keyword_3488 = python_operator(stypy.reporting.localization.Localization(__file__, 629, 11), 'and', path_file_3485, install_path_file_3487)
        
        # Testing the type of an if condition (line 629)
        if_condition_3489 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 629, 8), result_and_keyword_3488)
        # Assigning a type to the variable 'if_condition_3489' (line 629)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 8), 'if_condition_3489', if_condition_3489)
        # SSA begins for if statement (line 629)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 630)
        # Processing the call arguments (line 630)
        
        # Call to join(...): (line 630)
        # Processing the call arguments (line 630)
        # Getting the type of 'self' (line 630)
        self_3495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 40), 'self', False)
        # Obtaining the member 'install_libbase' of a type (line 630)
        install_libbase_3496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 40), self_3495, 'install_libbase')
        # Getting the type of 'self' (line 631)
        self_3497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 40), 'self', False)
        # Obtaining the member 'path_file' of a type (line 631)
        path_file_3498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 40), self_3497, 'path_file')
        str_3499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 57), 'str', '.pth')
        # Applying the binary operator '+' (line 631)
        result_add_3500 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 40), '+', path_file_3498, str_3499)
        
        # Processing the call keyword arguments (line 630)
        kwargs_3501 = {}
        # Getting the type of 'os' (line 630)
        os_3492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 630)
        path_3493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 27), os_3492, 'path')
        # Obtaining the member 'join' of a type (line 630)
        join_3494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 27), path_3493, 'join')
        # Calling join(args, kwargs) (line 630)
        join_call_result_3502 = invoke(stypy.reporting.localization.Localization(__file__, 630, 27), join_3494, *[install_libbase_3496, result_add_3500], **kwargs_3501)
        
        # Processing the call keyword arguments (line 630)
        kwargs_3503 = {}
        # Getting the type of 'outputs' (line 630)
        outputs_3490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 12), 'outputs', False)
        # Obtaining the member 'append' of a type (line 630)
        append_3491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 12), outputs_3490, 'append')
        # Calling append(args, kwargs) (line 630)
        append_call_result_3504 = invoke(stypy.reporting.localization.Localization(__file__, 630, 12), append_3491, *[join_call_result_3502], **kwargs_3503)
        
        # SSA join for if statement (line 629)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'outputs' (line 633)
        outputs_3505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 15), 'outputs')
        # Assigning a type to the variable 'stypy_return_type' (line 633)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 8), 'stypy_return_type', outputs_3505)
        
        # ################# End of 'get_outputs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_outputs' in the type store
        # Getting the type of 'stypy_return_type' (line 618)
        stypy_return_type_3506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3506)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_outputs'
        return stypy_return_type_3506


    @norecursion
    def get_inputs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_inputs'
        module_type_store = module_type_store.open_function_context('get_inputs', 635, 4, False)
        # Assigning a type to the variable 'self' (line 636)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install.get_inputs.__dict__.__setitem__('stypy_localization', localization)
        install.get_inputs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install.get_inputs.__dict__.__setitem__('stypy_type_store', module_type_store)
        install.get_inputs.__dict__.__setitem__('stypy_function_name', 'install.get_inputs')
        install.get_inputs.__dict__.__setitem__('stypy_param_names_list', [])
        install.get_inputs.__dict__.__setitem__('stypy_varargs_param_name', None)
        install.get_inputs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install.get_inputs.__dict__.__setitem__('stypy_call_defaults', defaults)
        install.get_inputs.__dict__.__setitem__('stypy_call_varargs', varargs)
        install.get_inputs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install.get_inputs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install.get_inputs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_inputs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_inputs(...)' code ##################

        
        # Assigning a List to a Name (line 637):
        
        # Assigning a List to a Name (line 637):
        
        # Obtaining an instance of the builtin type 'list' (line 637)
        list_3507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 637)
        
        # Assigning a type to the variable 'inputs' (line 637)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 8), 'inputs', list_3507)
        
        
        # Call to get_sub_commands(...): (line 638)
        # Processing the call keyword arguments (line 638)
        kwargs_3510 = {}
        # Getting the type of 'self' (line 638)
        self_3508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 24), 'self', False)
        # Obtaining the member 'get_sub_commands' of a type (line 638)
        get_sub_commands_3509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 24), self_3508, 'get_sub_commands')
        # Calling get_sub_commands(args, kwargs) (line 638)
        get_sub_commands_call_result_3511 = invoke(stypy.reporting.localization.Localization(__file__, 638, 24), get_sub_commands_3509, *[], **kwargs_3510)
        
        # Testing the type of a for loop iterable (line 638)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 638, 8), get_sub_commands_call_result_3511)
        # Getting the type of the for loop variable (line 638)
        for_loop_var_3512 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 638, 8), get_sub_commands_call_result_3511)
        # Assigning a type to the variable 'cmd_name' (line 638)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 8), 'cmd_name', for_loop_var_3512)
        # SSA begins for a for statement (line 638)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 639):
        
        # Assigning a Call to a Name (line 639):
        
        # Call to get_finalized_command(...): (line 639)
        # Processing the call arguments (line 639)
        # Getting the type of 'cmd_name' (line 639)
        cmd_name_3515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 45), 'cmd_name', False)
        # Processing the call keyword arguments (line 639)
        kwargs_3516 = {}
        # Getting the type of 'self' (line 639)
        self_3513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 18), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 639)
        get_finalized_command_3514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 18), self_3513, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 639)
        get_finalized_command_call_result_3517 = invoke(stypy.reporting.localization.Localization(__file__, 639, 18), get_finalized_command_3514, *[cmd_name_3515], **kwargs_3516)
        
        # Assigning a type to the variable 'cmd' (line 639)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 12), 'cmd', get_finalized_command_call_result_3517)
        
        # Call to extend(...): (line 640)
        # Processing the call arguments (line 640)
        
        # Call to get_inputs(...): (line 640)
        # Processing the call keyword arguments (line 640)
        kwargs_3522 = {}
        # Getting the type of 'cmd' (line 640)
        cmd_3520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 26), 'cmd', False)
        # Obtaining the member 'get_inputs' of a type (line 640)
        get_inputs_3521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 26), cmd_3520, 'get_inputs')
        # Calling get_inputs(args, kwargs) (line 640)
        get_inputs_call_result_3523 = invoke(stypy.reporting.localization.Localization(__file__, 640, 26), get_inputs_3521, *[], **kwargs_3522)
        
        # Processing the call keyword arguments (line 640)
        kwargs_3524 = {}
        # Getting the type of 'inputs' (line 640)
        inputs_3518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 12), 'inputs', False)
        # Obtaining the member 'extend' of a type (line 640)
        extend_3519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 12), inputs_3518, 'extend')
        # Calling extend(args, kwargs) (line 640)
        extend_call_result_3525 = invoke(stypy.reporting.localization.Localization(__file__, 640, 12), extend_3519, *[get_inputs_call_result_3523], **kwargs_3524)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'inputs' (line 642)
        inputs_3526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 15), 'inputs')
        # Assigning a type to the variable 'stypy_return_type' (line 642)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 8), 'stypy_return_type', inputs_3526)
        
        # ################# End of 'get_inputs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_inputs' in the type store
        # Getting the type of 'stypy_return_type' (line 635)
        stypy_return_type_3527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3527)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_inputs'
        return stypy_return_type_3527


    @norecursion
    def has_lib(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_lib'
        module_type_store = module_type_store.open_function_context('has_lib', 647, 4, False)
        # Assigning a type to the variable 'self' (line 648)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install.has_lib.__dict__.__setitem__('stypy_localization', localization)
        install.has_lib.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install.has_lib.__dict__.__setitem__('stypy_type_store', module_type_store)
        install.has_lib.__dict__.__setitem__('stypy_function_name', 'install.has_lib')
        install.has_lib.__dict__.__setitem__('stypy_param_names_list', [])
        install.has_lib.__dict__.__setitem__('stypy_varargs_param_name', None)
        install.has_lib.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install.has_lib.__dict__.__setitem__('stypy_call_defaults', defaults)
        install.has_lib.__dict__.__setitem__('stypy_call_varargs', varargs)
        install.has_lib.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install.has_lib.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install.has_lib', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_lib', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_lib(...)' code ##################

        str_3528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, (-1)), 'str', 'Return true if the current distribution has any Python\n        modules to install.')
        
        # Evaluating a boolean operation
        
        # Call to has_pure_modules(...): (line 650)
        # Processing the call keyword arguments (line 650)
        kwargs_3532 = {}
        # Getting the type of 'self' (line 650)
        self_3529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 16), 'self', False)
        # Obtaining the member 'distribution' of a type (line 650)
        distribution_3530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 16), self_3529, 'distribution')
        # Obtaining the member 'has_pure_modules' of a type (line 650)
        has_pure_modules_3531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 16), distribution_3530, 'has_pure_modules')
        # Calling has_pure_modules(args, kwargs) (line 650)
        has_pure_modules_call_result_3533 = invoke(stypy.reporting.localization.Localization(__file__, 650, 16), has_pure_modules_3531, *[], **kwargs_3532)
        
        
        # Call to has_ext_modules(...): (line 651)
        # Processing the call keyword arguments (line 651)
        kwargs_3537 = {}
        # Getting the type of 'self' (line 651)
        self_3534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 16), 'self', False)
        # Obtaining the member 'distribution' of a type (line 651)
        distribution_3535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 16), self_3534, 'distribution')
        # Obtaining the member 'has_ext_modules' of a type (line 651)
        has_ext_modules_3536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 16), distribution_3535, 'has_ext_modules')
        # Calling has_ext_modules(args, kwargs) (line 651)
        has_ext_modules_call_result_3538 = invoke(stypy.reporting.localization.Localization(__file__, 651, 16), has_ext_modules_3536, *[], **kwargs_3537)
        
        # Applying the binary operator 'or' (line 650)
        result_or_keyword_3539 = python_operator(stypy.reporting.localization.Localization(__file__, 650, 16), 'or', has_pure_modules_call_result_3533, has_ext_modules_call_result_3538)
        
        # Assigning a type to the variable 'stypy_return_type' (line 650)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 8), 'stypy_return_type', result_or_keyword_3539)
        
        # ################# End of 'has_lib(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_lib' in the type store
        # Getting the type of 'stypy_return_type' (line 647)
        stypy_return_type_3540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3540)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_lib'
        return stypy_return_type_3540


    @norecursion
    def has_headers(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_headers'
        module_type_store = module_type_store.open_function_context('has_headers', 653, 4, False)
        # Assigning a type to the variable 'self' (line 654)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install.has_headers.__dict__.__setitem__('stypy_localization', localization)
        install.has_headers.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install.has_headers.__dict__.__setitem__('stypy_type_store', module_type_store)
        install.has_headers.__dict__.__setitem__('stypy_function_name', 'install.has_headers')
        install.has_headers.__dict__.__setitem__('stypy_param_names_list', [])
        install.has_headers.__dict__.__setitem__('stypy_varargs_param_name', None)
        install.has_headers.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install.has_headers.__dict__.__setitem__('stypy_call_defaults', defaults)
        install.has_headers.__dict__.__setitem__('stypy_call_varargs', varargs)
        install.has_headers.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install.has_headers.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install.has_headers', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_headers', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_headers(...)' code ##################

        
        # Call to has_headers(...): (line 654)
        # Processing the call keyword arguments (line 654)
        kwargs_3544 = {}
        # Getting the type of 'self' (line 654)
        self_3541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 15), 'self', False)
        # Obtaining the member 'distribution' of a type (line 654)
        distribution_3542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 654, 15), self_3541, 'distribution')
        # Obtaining the member 'has_headers' of a type (line 654)
        has_headers_3543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 654, 15), distribution_3542, 'has_headers')
        # Calling has_headers(args, kwargs) (line 654)
        has_headers_call_result_3545 = invoke(stypy.reporting.localization.Localization(__file__, 654, 15), has_headers_3543, *[], **kwargs_3544)
        
        # Assigning a type to the variable 'stypy_return_type' (line 654)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 8), 'stypy_return_type', has_headers_call_result_3545)
        
        # ################# End of 'has_headers(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_headers' in the type store
        # Getting the type of 'stypy_return_type' (line 653)
        stypy_return_type_3546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3546)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_headers'
        return stypy_return_type_3546


    @norecursion
    def has_scripts(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_scripts'
        module_type_store = module_type_store.open_function_context('has_scripts', 656, 4, False)
        # Assigning a type to the variable 'self' (line 657)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install.has_scripts.__dict__.__setitem__('stypy_localization', localization)
        install.has_scripts.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install.has_scripts.__dict__.__setitem__('stypy_type_store', module_type_store)
        install.has_scripts.__dict__.__setitem__('stypy_function_name', 'install.has_scripts')
        install.has_scripts.__dict__.__setitem__('stypy_param_names_list', [])
        install.has_scripts.__dict__.__setitem__('stypy_varargs_param_name', None)
        install.has_scripts.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install.has_scripts.__dict__.__setitem__('stypy_call_defaults', defaults)
        install.has_scripts.__dict__.__setitem__('stypy_call_varargs', varargs)
        install.has_scripts.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install.has_scripts.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install.has_scripts', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_scripts', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_scripts(...)' code ##################

        
        # Call to has_scripts(...): (line 657)
        # Processing the call keyword arguments (line 657)
        kwargs_3550 = {}
        # Getting the type of 'self' (line 657)
        self_3547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 15), 'self', False)
        # Obtaining the member 'distribution' of a type (line 657)
        distribution_3548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 15), self_3547, 'distribution')
        # Obtaining the member 'has_scripts' of a type (line 657)
        has_scripts_3549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 15), distribution_3548, 'has_scripts')
        # Calling has_scripts(args, kwargs) (line 657)
        has_scripts_call_result_3551 = invoke(stypy.reporting.localization.Localization(__file__, 657, 15), has_scripts_3549, *[], **kwargs_3550)
        
        # Assigning a type to the variable 'stypy_return_type' (line 657)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 8), 'stypy_return_type', has_scripts_call_result_3551)
        
        # ################# End of 'has_scripts(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_scripts' in the type store
        # Getting the type of 'stypy_return_type' (line 656)
        stypy_return_type_3552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3552)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_scripts'
        return stypy_return_type_3552


    @norecursion
    def has_data(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_data'
        module_type_store = module_type_store.open_function_context('has_data', 659, 4, False)
        # Assigning a type to the variable 'self' (line 660)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install.has_data.__dict__.__setitem__('stypy_localization', localization)
        install.has_data.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install.has_data.__dict__.__setitem__('stypy_type_store', module_type_store)
        install.has_data.__dict__.__setitem__('stypy_function_name', 'install.has_data')
        install.has_data.__dict__.__setitem__('stypy_param_names_list', [])
        install.has_data.__dict__.__setitem__('stypy_varargs_param_name', None)
        install.has_data.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install.has_data.__dict__.__setitem__('stypy_call_defaults', defaults)
        install.has_data.__dict__.__setitem__('stypy_call_varargs', varargs)
        install.has_data.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install.has_data.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install.has_data', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_data', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_data(...)' code ##################

        
        # Call to has_data_files(...): (line 660)
        # Processing the call keyword arguments (line 660)
        kwargs_3556 = {}
        # Getting the type of 'self' (line 660)
        self_3553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 15), 'self', False)
        # Obtaining the member 'distribution' of a type (line 660)
        distribution_3554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 15), self_3553, 'distribution')
        # Obtaining the member 'has_data_files' of a type (line 660)
        has_data_files_3555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 15), distribution_3554, 'has_data_files')
        # Calling has_data_files(args, kwargs) (line 660)
        has_data_files_call_result_3557 = invoke(stypy.reporting.localization.Localization(__file__, 660, 15), has_data_files_3555, *[], **kwargs_3556)
        
        # Assigning a type to the variable 'stypy_return_type' (line 660)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 8), 'stypy_return_type', has_data_files_call_result_3557)
        
        # ################# End of 'has_data(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_data' in the type store
        # Getting the type of 'stypy_return_type' (line 659)
        stypy_return_type_3558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3558)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_data'
        return stypy_return_type_3558

    
    # Assigning a List to a Name (line 665):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 94, 0, False)
        # Assigning a type to the variable 'self' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'install' (line 94)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'install', install)

# Assigning a Str to a Name (line 96):
str_3559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 18), 'str', 'install everything from build directory')
# Getting the type of 'install'
install_3560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_3560, 'description', str_3559)

# Assigning a List to a Name (line 98):

# Obtaining an instance of the builtin type 'list' (line 98)
list_3561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 98)
# Adding element type (line 98)

# Obtaining an instance of the builtin type 'tuple' (line 100)
tuple_3562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 100)
# Adding element type (line 100)
str_3563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 9), 'str', 'prefix=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 9), tuple_3562, str_3563)
# Adding element type (line 100)
# Getting the type of 'None' (line 100)
None_3564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 20), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 9), tuple_3562, None_3564)
# Adding element type (line 100)
str_3565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 9), 'str', 'installation prefix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 9), tuple_3562, str_3565)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 19), list_3561, tuple_3562)
# Adding element type (line 98)

# Obtaining an instance of the builtin type 'tuple' (line 102)
tuple_3566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 102)
# Adding element type (line 102)
str_3567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 9), 'str', 'exec-prefix=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 9), tuple_3566, str_3567)
# Adding element type (line 102)
# Getting the type of 'None' (line 102)
None_3568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 25), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 9), tuple_3566, None_3568)
# Adding element type (line 102)
str_3569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 9), 'str', '(Unix only) prefix for platform-specific files')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 9), tuple_3566, str_3569)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 19), list_3561, tuple_3566)
# Adding element type (line 98)

# Obtaining an instance of the builtin type 'tuple' (line 104)
tuple_3570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 104)
# Adding element type (line 104)
str_3571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 9), 'str', 'home=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 9), tuple_3570, str_3571)
# Adding element type (line 104)
# Getting the type of 'None' (line 104)
None_3572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 18), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 9), tuple_3570, None_3572)
# Adding element type (line 104)
str_3573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 9), 'str', '(Unix only) home directory to install under')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 9), tuple_3570, str_3573)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 19), list_3561, tuple_3570)
# Adding element type (line 98)

# Obtaining an instance of the builtin type 'tuple' (line 106)
tuple_3574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 106)
# Adding element type (line 106)
str_3575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 9), 'str', 'user')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 9), tuple_3574, str_3575)
# Adding element type (line 106)
# Getting the type of 'None' (line 106)
None_3576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 17), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 9), tuple_3574, None_3576)
# Adding element type (line 106)
str_3577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 9), 'str', "install in user site-package '%s'")
# Getting the type of 'USER_SITE' (line 107)
USER_SITE_3578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 47), 'USER_SITE')
# Applying the binary operator '%' (line 107)
result_mod_3579 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 9), '%', str_3577, USER_SITE_3578)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 9), tuple_3574, result_mod_3579)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 19), list_3561, tuple_3574)
# Adding element type (line 98)

# Obtaining an instance of the builtin type 'tuple' (line 110)
tuple_3580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 110)
# Adding element type (line 110)
str_3581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 9), 'str', 'install-base=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 9), tuple_3580, str_3581)
# Adding element type (line 110)
# Getting the type of 'None' (line 110)
None_3582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 9), tuple_3580, None_3582)
# Adding element type (line 110)
str_3583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 9), 'str', 'base installation directory (instead of --prefix or --home)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 9), tuple_3580, str_3583)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 19), list_3561, tuple_3580)
# Adding element type (line 98)

# Obtaining an instance of the builtin type 'tuple' (line 112)
tuple_3584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 112)
# Adding element type (line 112)
str_3585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 9), 'str', 'install-platbase=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 9), tuple_3584, str_3585)
# Adding element type (line 112)
# Getting the type of 'None' (line 112)
None_3586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 30), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 9), tuple_3584, None_3586)
# Adding element type (line 112)
str_3587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 9), 'str', 'base installation directory for platform-specific files ')
str_3588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 9), 'str', '(instead of --exec-prefix or --home)')
# Applying the binary operator '+' (line 113)
result_add_3589 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 9), '+', str_3587, str_3588)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 9), tuple_3584, result_add_3589)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 19), list_3561, tuple_3584)
# Adding element type (line 98)

# Obtaining an instance of the builtin type 'tuple' (line 115)
tuple_3590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 115)
# Adding element type (line 115)
str_3591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 9), 'str', 'root=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 9), tuple_3590, str_3591)
# Adding element type (line 115)
# Getting the type of 'None' (line 115)
None_3592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 18), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 9), tuple_3590, None_3592)
# Adding element type (line 115)
str_3593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 9), 'str', 'install everything relative to this alternate root directory')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 9), tuple_3590, str_3593)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 19), list_3561, tuple_3590)
# Adding element type (line 98)

# Obtaining an instance of the builtin type 'tuple' (line 119)
tuple_3594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 119)
# Adding element type (line 119)
str_3595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 9), 'str', 'install-purelib=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 9), tuple_3594, str_3595)
# Adding element type (line 119)
# Getting the type of 'None' (line 119)
None_3596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 29), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 9), tuple_3594, None_3596)
# Adding element type (line 119)
str_3597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 9), 'str', 'installation directory for pure Python module distributions')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 9), tuple_3594, str_3597)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 19), list_3561, tuple_3594)
# Adding element type (line 98)

# Obtaining an instance of the builtin type 'tuple' (line 121)
tuple_3598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 121)
# Adding element type (line 121)
str_3599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 9), 'str', 'install-platlib=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 9), tuple_3598, str_3599)
# Adding element type (line 121)
# Getting the type of 'None' (line 121)
None_3600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 29), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 9), tuple_3598, None_3600)
# Adding element type (line 121)
str_3601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 9), 'str', 'installation directory for non-pure module distributions')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 9), tuple_3598, str_3601)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 19), list_3561, tuple_3598)
# Adding element type (line 98)

# Obtaining an instance of the builtin type 'tuple' (line 123)
tuple_3602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 123)
# Adding element type (line 123)
str_3603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 9), 'str', 'install-lib=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 9), tuple_3602, str_3603)
# Adding element type (line 123)
# Getting the type of 'None' (line 123)
None_3604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 25), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 9), tuple_3602, None_3604)
# Adding element type (line 123)
str_3605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 9), 'str', 'installation directory for all module distributions ')
str_3606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 9), 'str', '(overrides --install-purelib and --install-platlib)')
# Applying the binary operator '+' (line 124)
result_add_3607 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 9), '+', str_3605, str_3606)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 9), tuple_3602, result_add_3607)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 19), list_3561, tuple_3602)
# Adding element type (line 98)

# Obtaining an instance of the builtin type 'tuple' (line 127)
tuple_3608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 127)
# Adding element type (line 127)
str_3609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 9), 'str', 'install-headers=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 9), tuple_3608, str_3609)
# Adding element type (line 127)
# Getting the type of 'None' (line 127)
None_3610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 29), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 9), tuple_3608, None_3610)
# Adding element type (line 127)
str_3611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 9), 'str', 'installation directory for C/C++ headers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 9), tuple_3608, str_3611)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 19), list_3561, tuple_3608)
# Adding element type (line 98)

# Obtaining an instance of the builtin type 'tuple' (line 129)
tuple_3612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 129)
# Adding element type (line 129)
str_3613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 9), 'str', 'install-scripts=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 9), tuple_3612, str_3613)
# Adding element type (line 129)
# Getting the type of 'None' (line 129)
None_3614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 29), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 9), tuple_3612, None_3614)
# Adding element type (line 129)
str_3615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 9), 'str', 'installation directory for Python scripts')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 9), tuple_3612, str_3615)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 19), list_3561, tuple_3612)
# Adding element type (line 98)

# Obtaining an instance of the builtin type 'tuple' (line 131)
tuple_3616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 131)
# Adding element type (line 131)
str_3617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 9), 'str', 'install-data=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 9), tuple_3616, str_3617)
# Adding element type (line 131)
# Getting the type of 'None' (line 131)
None_3618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 9), tuple_3616, None_3618)
# Adding element type (line 131)
str_3619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 9), 'str', 'installation directory for data files')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 9), tuple_3616, str_3619)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 19), list_3561, tuple_3616)
# Adding element type (line 98)

# Obtaining an instance of the builtin type 'tuple' (line 137)
tuple_3620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 137)
# Adding element type (line 137)
str_3621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 9), 'str', 'compile')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 9), tuple_3620, str_3621)
# Adding element type (line 137)
str_3622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 20), 'str', 'c')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 9), tuple_3620, str_3622)
# Adding element type (line 137)
str_3623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 25), 'str', 'compile .py to .pyc [default]')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 9), tuple_3620, str_3623)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 19), list_3561, tuple_3620)
# Adding element type (line 98)

# Obtaining an instance of the builtin type 'tuple' (line 138)
tuple_3624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 138)
# Adding element type (line 138)
str_3625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 9), 'str', 'no-compile')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 9), tuple_3624, str_3625)
# Adding element type (line 138)
# Getting the type of 'None' (line 138)
None_3626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 23), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 9), tuple_3624, None_3626)
# Adding element type (line 138)
str_3627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 29), 'str', "don't compile .py files")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 9), tuple_3624, str_3627)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 19), list_3561, tuple_3624)
# Adding element type (line 98)

# Obtaining an instance of the builtin type 'tuple' (line 139)
tuple_3628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 139)
# Adding element type (line 139)
str_3629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 9), 'str', 'optimize=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 9), tuple_3628, str_3629)
# Adding element type (line 139)
str_3630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 22), 'str', 'O')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 9), tuple_3628, str_3630)
# Adding element type (line 139)
str_3631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 9), 'str', 'also compile with optimization: -O1 for "python -O", -O2 for "python -OO", and -O0 to disable [default: -O0]')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 9), tuple_3628, str_3631)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 19), list_3561, tuple_3628)
# Adding element type (line 98)

# Obtaining an instance of the builtin type 'tuple' (line 144)
tuple_3632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 144)
# Adding element type (line 144)
str_3633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 9), 'str', 'force')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 9), tuple_3632, str_3633)
# Adding element type (line 144)
str_3634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 18), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 9), tuple_3632, str_3634)
# Adding element type (line 144)
str_3635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 9), 'str', 'force installation (overwrite any existing files)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 9), tuple_3632, str_3635)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 19), list_3561, tuple_3632)
# Adding element type (line 98)

# Obtaining an instance of the builtin type 'tuple' (line 146)
tuple_3636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 146)
# Adding element type (line 146)
str_3637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 9), 'str', 'skip-build')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 9), tuple_3636, str_3637)
# Adding element type (line 146)
# Getting the type of 'None' (line 146)
None_3638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 23), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 9), tuple_3636, None_3638)
# Adding element type (line 146)
str_3639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 9), 'str', 'skip rebuilding everything (for testing/debugging)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 9), tuple_3636, str_3639)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 19), list_3561, tuple_3636)
# Adding element type (line 98)

# Obtaining an instance of the builtin type 'tuple' (line 155)
tuple_3640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 155)
# Adding element type (line 155)
str_3641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 9), 'str', 'record=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 9), tuple_3640, str_3641)
# Adding element type (line 155)
# Getting the type of 'None' (line 155)
None_3642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 20), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 9), tuple_3640, None_3642)
# Adding element type (line 155)
str_3643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 9), 'str', 'filename in which to record list of installed files')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 9), tuple_3640, str_3643)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 19), list_3561, tuple_3640)

# Getting the type of 'install'
install_3644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_3644, 'user_options', list_3561)

# Assigning a List to a Name (line 159):

# Obtaining an instance of the builtin type 'list' (line 159)
list_3645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 159)
# Adding element type (line 159)
str_3646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 23), 'str', 'compile')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 22), list_3645, str_3646)
# Adding element type (line 159)
str_3647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 34), 'str', 'force')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 22), list_3645, str_3647)
# Adding element type (line 159)
str_3648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 43), 'str', 'skip-build')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 22), list_3645, str_3648)
# Adding element type (line 159)
str_3649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 57), 'str', 'user')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 22), list_3645, str_3649)

# Getting the type of 'install'
install_3650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install')
# Setting the type of the member 'boolean_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_3650, 'boolean_options', list_3645)

# Assigning a Dict to a Name (line 160):

# Obtaining an instance of the builtin type 'dict' (line 160)
dict_3651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 160)
# Adding element type (key, value) (line 160)
str_3652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 20), 'str', 'no-compile')
str_3653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 35), 'str', 'compile')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 19), dict_3651, (str_3652, str_3653))

# Getting the type of 'install'
install_3654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install')
# Setting the type of the member 'negative_opt' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_3654, 'negative_opt', dict_3651)

# Assigning a List to a Name (line 665):

# Obtaining an instance of the builtin type 'list' (line 665)
list_3655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 665)
# Adding element type (line 665)

# Obtaining an instance of the builtin type 'tuple' (line 665)
tuple_3656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 665)
# Adding element type (line 665)
str_3657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 21), 'str', 'install_lib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 665, 21), tuple_3656, str_3657)
# Adding element type (line 665)
# Getting the type of 'install'
install_3658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install')
# Obtaining the member 'has_lib' of a type
has_lib_3659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_3658, 'has_lib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 665, 21), tuple_3656, has_lib_3659)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 665, 19), list_3655, tuple_3656)
# Adding element type (line 665)

# Obtaining an instance of the builtin type 'tuple' (line 666)
tuple_3660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 666)
# Adding element type (line 666)
str_3661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 21), 'str', 'install_headers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 666, 21), tuple_3660, str_3661)
# Adding element type (line 666)
# Getting the type of 'install'
install_3662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install')
# Obtaining the member 'has_headers' of a type
has_headers_3663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_3662, 'has_headers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 666, 21), tuple_3660, has_headers_3663)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 665, 19), list_3655, tuple_3660)
# Adding element type (line 665)

# Obtaining an instance of the builtin type 'tuple' (line 667)
tuple_3664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 667)
# Adding element type (line 667)
str_3665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 21), 'str', 'install_scripts')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 21), tuple_3664, str_3665)
# Adding element type (line 667)
# Getting the type of 'install'
install_3666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install')
# Obtaining the member 'has_scripts' of a type
has_scripts_3667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_3666, 'has_scripts')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 21), tuple_3664, has_scripts_3667)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 665, 19), list_3655, tuple_3664)
# Adding element type (line 665)

# Obtaining an instance of the builtin type 'tuple' (line 668)
tuple_3668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 668)
# Adding element type (line 668)
str_3669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 21), 'str', 'install_data')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 668, 21), tuple_3668, str_3669)
# Adding element type (line 668)
# Getting the type of 'install'
install_3670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install')
# Obtaining the member 'has_data' of a type
has_data_3671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_3670, 'has_data')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 668, 21), tuple_3668, has_data_3671)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 665, 19), list_3655, tuple_3668)
# Adding element type (line 665)

# Obtaining an instance of the builtin type 'tuple' (line 669)
tuple_3672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 669)
# Adding element type (line 669)
str_3673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 21), 'str', 'install_egg_info')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 669, 21), tuple_3672, str_3673)
# Adding element type (line 669)

@norecursion
def _stypy_temp_lambda_2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_2'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_2', 669, 41, True)
    # Passed parameters checking function
    _stypy_temp_lambda_2.stypy_localization = localization
    _stypy_temp_lambda_2.stypy_type_of_self = None
    _stypy_temp_lambda_2.stypy_type_store = module_type_store
    _stypy_temp_lambda_2.stypy_function_name = '_stypy_temp_lambda_2'
    _stypy_temp_lambda_2.stypy_param_names_list = ['self']
    _stypy_temp_lambda_2.stypy_varargs_param_name = None
    _stypy_temp_lambda_2.stypy_kwargs_param_name = None
    _stypy_temp_lambda_2.stypy_call_defaults = defaults
    _stypy_temp_lambda_2.stypy_call_varargs = varargs
    _stypy_temp_lambda_2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_2', ['self'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_2', ['self'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    # Getting the type of 'True' (line 669)
    True_3674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 53), 'True')
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 669)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 41), 'stypy_return_type', True_3674)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_2' in the type store
    # Getting the type of 'stypy_return_type' (line 669)
    stypy_return_type_3675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 41), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3675)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_2'
    return stypy_return_type_3675

# Assigning a type to the variable '_stypy_temp_lambda_2' (line 669)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 41), '_stypy_temp_lambda_2', _stypy_temp_lambda_2)
# Getting the type of '_stypy_temp_lambda_2' (line 669)
_stypy_temp_lambda_2_3676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 41), '_stypy_temp_lambda_2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 669, 21), tuple_3672, _stypy_temp_lambda_2_3676)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 665, 19), list_3655, tuple_3672)

# Getting the type of 'install'
install_3677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install')
# Setting the type of the member 'sub_commands' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_3677, 'sub_commands', list_3655)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
